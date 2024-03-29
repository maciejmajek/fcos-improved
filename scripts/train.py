import os
import tqdm
import wandb
import torch
import torch.nn as nn
import gin.torch
from src.fcos import FCOS
from src.loss import IOULoss, FocalLoss, MulticlassFocalLoss
from src.dataset import BDD100K
from src.dataset import cat_to_num
import src.utils as architecture
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import MulticlassAUROC
import torchvision.transforms as T


device = "cuda"
train_iterations = 0
test_iterations = 0
scaler = torch.cuda.amp.GradScaler()


@torch.no_grad()
def get_auc_roc(prediction, target, thresholds=10):
    if prediction.shape[1] > 1:
        metric = MulticlassAUROC(
            num_classes=prediction.shape[1],
            thresholds=thresholds,
            average="weighted",
            ignore_index=0,
        )
    else:
        metric = BinaryAUROC(thresholds=thresholds)
        prediction = prediction.squeeze(1)

    metric = metric.to(device)
    score = metric(prediction, target.long())

    return score


def train_step(
    model,
    img,
    maps_cls,
    maps_reg,
    maps_cnt,
    maps_da,
    loader,
    strides,
    loss_fn_cls,
    loss_fn_reg,
    loss_fn_cnt,
    loss_fn_da,
    scheduler,
):
    cls_pred, reg_pred, cnt_pred, da_pred = model(img)
    loss = {
        "cls": 0,
        "reg": 0,
        "cnt": 0,
        "da": 0,
    }
    auc_rocs = {
        "cls": [],
        "da": 0,
    }

    maps_cls = [maps_cls[int(stride)] for stride in strides]
    maps_reg = [maps_reg[int(stride)] for stride in strides]
    maps_cnt = [maps_cnt[int(stride)] for stride in strides]

    for t_cls, t_reg, t_cnt, p_cls, p_reg, p_cnt in zip(
        maps_cls, maps_reg, maps_cnt, cls_pred, reg_pred, cnt_pred
    ):
        t_cls, t_reg, t_cnt = [
            x.to(device, non_blocking=True) for x in [t_cls, t_reg, t_cnt]
        ]
        loss["cls"] += loss_fn_cls(p_cls.squeeze(1), t_cls)
        if t_cls.sum() > 0:
            loss["reg"] += (loss_fn_reg(p_reg, t_reg) * (t_cls > 0)).sum() / (
                t_cls > 0
            ).sum()
        else:
            loss["reg"] += torch.tensor(0.0).to(device)

        loss["cnt"] += loss_fn_cnt(p_cnt.squeeze(1), t_cnt)
        if train_iterations % 10 == 0:
            auc_rocs["cls"].append(get_auc_roc(p_cls, t_cls))

    if train_iterations % 10 == 0:
        auc_rocs["da"] = get_auc_roc(da_pred, maps_da.squeeze(1))
    loss["da"] = loss_fn_da(da_pred.squeeze(1), maps_da.squeeze(1).float())
    loss["cls"] = loss["cls"] / loader.batch_size
    loss["reg"] = loss["reg"] / loader.batch_size
    loss["cnt"] = loss["cnt"] / loader.batch_size

    return loss, auc_rocs


@torch.no_grad()
def test_step(
    model,
    img,
    maps_cls,
    maps_reg,
    maps_cnt,
    maps_da,
    loader,
    strides,
    loss_fn_cls,
    loss_fn_reg,
    loss_fn_cnt,
    loss_fn_da,
):
    cls_pred, reg_pred, cnt_pred, da_pred = model(img)
    loss = {
        "cls": 0,
        "reg": 0,
        "cnt": 0,
        "da": 0,
    }
    auc_rocs = {
        "cls": [],
        "da": 0,
    }

    maps_cls = [maps_cls[int(stride)] for stride in strides]
    maps_reg = [maps_reg[int(stride)] for stride in strides]
    maps_cnt = [maps_cnt[int(stride)] for stride in strides]

    for t_cls, t_reg, t_cnt, p_cls, p_reg, p_cnt in zip(
        maps_cls, maps_reg, maps_cnt, cls_pred, reg_pred, cnt_pred
    ):
        t_cls, t_reg, t_cnt = [
            x.to(device, non_blocking=True) for x in [t_cls, t_reg, t_cnt]
        ]
        loss["cls"] += loss_fn_cls(p_cls.squeeze(1), t_cls)
        if t_cls.sum() > 0:
            loss["reg"] += (loss_fn_reg(p_reg, t_reg)).sum() / t_cls.sum()
        else:
            loss["reg"] += torch.tensor(0.0).to(device)
        loss["cnt"] += loss_fn_cnt(p_cnt.squeeze(1), t_cnt)

        auc_rocs["cls"].append(get_auc_roc(p_cls, t_cls))

    auc_rocs["da"] = get_auc_roc(da_pred, maps_da.squeeze(1))
    loss["da"] = loss_fn_da(da_pred.squeeze(1), maps_da.squeeze(1).float())
    loss["cls"] = loss["cls"] / loader.batch_size
    loss["reg"] = loss["reg"] / loader.batch_size
    loss["cnt"] = loss["cnt"] / loader.batch_size
    return loss, auc_rocs


def train_epoch(
    model,
    loader,
    loss_cls,
    loss_reg,
    loss_cnt,
    loss_da,
    optimizer,
    scheduler=None,
    transforms=None,
):
    model.train()
    epoch_loss = 0
    global train_iterations
    for img, maps_cls, maps_reg, maps_cnt, maps_da in tqdm.tqdm(loader):
        optimizer.zero_grad()
        img, maps_da = img.to(device), maps_da.to(device)

        if transforms:
            img = (255 * img).type(torch.uint8)
            img = transforms(img)
            img = img / 255.0
        with torch.cuda.amp.autocast():
            loss, auc_rocs = train_step(
                model,
                img,
                maps_cls,
                maps_reg,
                maps_cnt,
                maps_da,
                loader,
                model.strides,
                loss_cls,
                loss_reg,
                loss_cnt,
                loss_da,
                scheduler,
            )
        log = {"Train/Loss/" + k: v.item() for k, v in loss.items()}
        if auc_rocs["da"] != 0:
            for i, f1 in enumerate(auc_rocs["cls"]):
                log[f"Train/ROCAUC/cls_{model.strides[i]}"] = f1.item()
            log["Train/ROCAUC/da"] = auc_rocs["da"]
        log["iterations/train"] = train_iterations
        if scheduler:
            log["lr"] = scheduler.get_last_lr()[-1]
        wandb.log(log)

        loss = loss["cls"] + loss["reg"] + loss["cnt"] + loss["da"]
        epoch_loss += (loss).item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        train_iterations += loader.batch_size

    return epoch_loss / len(loader)


@torch.no_grad()
def test_epoch(model, loader, loss_cls, loss_reg, loss_cnt, loss_da):
    model.eval()
    epoch_loss = 0
    global test_iterations
    for img, maps_cls, maps_reg, maps_cnt, maps_da in tqdm.tqdm(loader):

        img, maps_da = img.to(device), maps_da.to(device)
        loss, auc_rocs = test_step(
            model,
            img,
            maps_cls,
            maps_reg,
            maps_cnt,
            maps_da,
            loader,
            model.strides,
            loss_cls,
            loss_reg,
            loss_cnt,
            loss_da,
        )
        log = {}
        for i, f1 in enumerate(auc_rocs["cls"]):
            log[f"Test/ROCAUC/cls_{model.strides[i]}"] = f1.item()
        log["Test/ROCAUC/da"] = auc_rocs["da"]
        log["iterations/test"] = test_iterations
        wandb.log(log)

        loss = loss["cls"] + loss["reg"] + loss["cnt"] + loss["da"]
        epoch_loss += (loss).item()
        test_iterations += loader.batch_size

    return epoch_loss / len(loader)


@gin.configurable
def get_optimizer(model, optimizer, lr):
    return getattr(torch.optim, optimizer)(
        model.parameters(), lr, weight_decay=0.0001, momentum=0.9
    )


@gin.configurable
def freeze_backbone(model, freeze=False):
    if freeze:
        for param in model.backbone_fpn.backbone.parameters():
            param.requires_grad = False

    return model


@gin.configurable
def save_model(model, i, prefix="model_", suffix="None", folder_name=None):
    if folder_name:
        folder_name = "zoo-" + folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        torch.save(model.state_dict(), f"{folder_name}/{prefix}{i}{suffix}.pth")
    else:
        torch.save(model.state_dict(), f"{prefix}{i}{suffix}.pth")


@gin.configurable
def build_model(
    backbone_name, fpn_channels, tower_depth, num_classes_det, freeze_backbone
):
    backbone = getattr(architecture, backbone_name)
    model = FCOS(
        backbone=backbone,
        num_classes=num_classes_det,
        fpn_channels=fpn_channels,
        tower_depth=tower_depth,
    )

    if freeze_backbone:
        for param in model.backbone_fpn.backbone.parameters():
            param.requires_grad = False

    return model


@gin.configurable
def training_configuration(num_epochs, batch_size):
    return num_epochs, batch_size


@gin.configurable
def build_datasets(root, size_limit_train, size_limit_val):
    train_dataset = BDD100K(
        root, split="train", return_drivable_area=True, size=size_limit_train
    )
    test_dataset = BDD100K(
        root, split="val", return_drivable_area=True, size=size_limit_val
    )
    return train_dataset, test_dataset


def main():
    # misc #
    cfg = gin.parse_config_file("scripts/config.gin")
    torch.backends.cudnn.benchmark = True

    # dataset #
    epochs, batch_size = training_configuration()
    train_dataset, test_dataset = build_datasets()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=14,
        drop_last=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8, drop_last=True
    )

    # model #
    model = build_model().to(device)
    optim = get_optimizer(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim, 0.04, epochs * len(train_loader), div_factor=5000,
    )

    num_classes = len(set(cat_to_num.values()))
    if num_classes > 2:
        loss_cls = MulticlassFocalLoss(reduction="sum")
    else:
        loss_cls = FocalLoss(reduction="sum")
    loss_cnt = nn.BCEWithLogitsLoss().to(device)
    loss_reg = IOULoss("giou")
    loss_da = nn.BCEWithLogitsLoss()

    config = {
        "model_size": sum([p.numel() for p in model.parameters()]),
        "num_classes": num_classes,
        "batch_size": batch_size,
        "optimizer": optim,
        "scheduler": scheduler.__str__(),
        "epochs": epochs,
        "loss_cnt": loss_cnt,
        "loss_reg": loss_reg,
        "loss_da": loss_da,
        "loss_cls": loss_cls,
    }

    print(config)
    wandb.init(project="INZ", entity="maciejeg1337", config=config)
    wandb.watch(model)

    for i in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            loss_cls,
            loss_reg,
            loss_cnt,
            loss_da,
            optim,
            scheduler,
        )
        wandb.log({"Train/Loss": train_loss})
        save_model(model, i, folder_name=wandb.run.name)
        save_model(optim, i, prefix="optim_", folder_name=wandb.run.name)
        test_loss = test_epoch(
            model, test_loader, loss_cls, loss_reg, loss_cnt, loss_da
        )
        wandb.log({"Test/Loss": test_loss})


if __name__ == "__main__":
    main()
