import tqdm
import torch
import torch.nn as nn
from src.fcos import FCOS
from src.loss import IOULoss, FocalLoss
from src.dataset import BDD100K
from torchmetrics.classification import BinaryF1Score
from torch.utils.tensorboard import SummaryWriter

device = "cuda"
train_iterations = 0
test_iterations = 0

f1 = BinaryF1Score(threshold=0.1).to(device)


def get_f1(prediction, target):
    score = f1(prediction.flatten(), target.flatten())
    return score


def train_step(
    model,
    img,
    maps_cls,
    maps_reg,
    maps_cnt,
    loader,
    strides,
    loss_fn_cls,
    loss_fn_reg,
    loss_fn_cnt,
):
    cls_pred, reg_pred, cnt_pred = model(img)
    loss = {
        "cls": 0,
        "reg": 0,
        "cnt": 0,
    }
    f1s = {
        "cls": [],
        "cnt": [],
    }

    maps_cls = [maps_cls[int(stride)] for stride in strides]
    maps_reg = [maps_reg[int(stride)] for stride in strides]
    maps_cnt = [maps_cnt[int(stride)] for stride in strides]
    pos_cls = 0

    for t_cls, t_reg, t_cnt, p_cls, p_reg, p_cnt in zip(
        maps_cls, maps_reg, maps_cnt, cls_pred, reg_pred, cnt_pred
    ):
        t_cls, t_reg, t_cnt = [
            x.to(device, non_blocking=True) for x in [t_cls, t_reg, t_cnt]
        ]
        loss["cls"] += loss_fn_cls(p_cls.squeeze(1), t_cls)
        loss["reg"] += loss_fn_reg(p_reg, t_reg)
        loss["cnt"] += loss_fn_cnt(p_cnt.squeeze(1), t_cnt)
        f1s["cls"].append(get_f1(p_cls, t_cls))
        pos_cls += (t_cls != 0).sum()

    loss["cls"] = loss["cls"] / loader.batch_size
    loss["reg"] = loss["reg"] / loader.batch_size
    loss["cnt"] = loss["cnt"] / loader.batch_size
    return loss, f1s


@torch.no_grad()
def test_step(
    model,
    img,
    maps_cls,
    maps_reg,
    maps_cnt,
    loader,
    strides,
    loss_fn_cls,
    loss_fn_reg,
    loss_fn_cnt,
):
    cls_pred, reg_pred, cnt_pred = model(img)
    loss = {
        "cls": 0,
        "reg": 0,
        "cnt": 0,
    }
    f1s = {
        "cls": [],
        "cnt": [],
    }

    maps_cls = [maps_cls[int(stride)] for stride in strides]
    maps_reg = [maps_reg[int(stride)] for stride in strides]
    maps_cnt = [maps_cnt[int(stride)] for stride in strides]
    pos_cls = 0

    for t_cls, t_reg, t_cnt, p_cls, p_reg, p_cnt in zip(
        maps_cls, maps_reg, maps_cnt, cls_pred, reg_pred, cnt_pred
    ):
        t_cls, t_reg, t_cnt = [
            x.to(device, non_blocking=True) for x in [t_cls, t_reg, t_cnt]
        ]
        loss["cls"] += loss_fn_cls(p_cls.squeeze(1), t_cls)
        loss["reg"] += loss_fn_reg(p_reg, t_reg)
        loss["cnt"] += loss_fn_cnt(p_cnt.squeeze(1), t_cnt)
        f1s["cls"].append(get_f1(p_cls, t_cls))
        pos_cls += (t_cls != 0).sum()

    loss["cls"] = loss["cls"] / loader.batch_size
    loss["reg"] = loss["reg"] / loader.batch_size
    loss["cnt"] = loss["cnt"] / loader.batch_size
    return loss, f1s


def train_epoch(model, loader, loss_cls, loss_reg, loss_cnt, optimizer, writer):
    model.train()
    epoch_loss = 0
    global train_iterations
    for img, maps_cls, maps_reg, maps_cnt in tqdm.tqdm(loader):

        img = img.to(device)

        loss, f1s = train_step(
            model,
            img,
            maps_cls,
            maps_reg,
            maps_cnt,
            loader,
            model.strides,
            loss_cls,
            loss_reg,
            loss_cnt,
        )
        writer.add_scalar("Train/Loss/cls", loss["cls"].item(), train_iterations)
        writer.add_scalar("Train/Loss/reg", loss["reg"].item(), train_iterations)
        writer.add_scalar("Train/Loss/cnt", loss["cnt"].item(), train_iterations)
        for i, f1 in enumerate(f1s["cls"]):
            writer.add_scalar(
                f"Train/F1/cls_{model.strides[i]}", f1.item(), train_iterations
            )

        loss = loss["cls"] + loss["reg"] + loss["cnt"]
        epoch_loss += (loss).item()
        loss.backward()
        optimizer.step()
        train_iterations += loader.batch_size

    return epoch_loss / len(loader)


@torch.no_grad()
def test_epoch(model, loader, loss_cls, loss_reg, loss_cnt, writer):
    model.eval()
    epoch_loss = 0
    global test_iterations
    for img, maps_cls, maps_reg, maps_cnt in tqdm.tqdm(loader):

        img = img.to(device)

        loss, f1s = test_step(
            model,
            img,
            maps_cls,
            maps_reg,
            maps_cnt,
            loader,
            model.strides,
            loss_cls,
            loss_reg,
            loss_cnt,
        )
        writer.add_scalar("Test/Loss/cls", loss["cls"].item(), test_iterations)
        writer.add_scalar("Test/Loss/reg", loss["reg"].item(), test_iterations)
        writer.add_scalar("Test/Loss/cnt", loss["cnt"].item(), test_iterations)
        for i, f1 in enumerate(f1s["cls"]):
            writer.add_scalar(
                f"Test/F1/cls_{model.strides[i]}", f1.item(), test_iterations
            )

        loss = loss["cls"] + loss["reg"] + loss["cnt"]
        epoch_loss += (loss).item()
        test_iterations += loader.batch_size

    return epoch_loss / len(loader)


def main():
    torch.backends.cudnn.benchmark = True
    # misc #
    writer = SummaryWriter()
    # dataset #
    root = "/media/muzg/D8F26982F269662A/bdd100k/bdd100k/"
    train_dataset = BDD100K(root, split="train",size=100)
    test_dataset = BDD100K(root, split="val",size=100)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=22, num_workers=5, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=22, num_workers=5, drop_last=True
    )
    # model #
    model = FCOS().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    for param in model.backbone_fpn.backbone.parameters():
        param.requires_grad = False

    loss_cls = FocalLoss(reduction="mean")
    loss_cnt = nn.BCEWithLogitsLoss()
    loss_reg = IOULoss()
    for i in range(10):
        train_loss = train_epoch(
            model, train_loader, loss_cls, loss_reg, loss_cnt, optim, writer
        )
        writer.add_scalar("Train/Loss", train_loss, train_iterations)
        torch.save(model.state_dict(),f"model_epoch_{i}.pth")
        test_loss = test_epoch(model, test_loader, loss_cls, loss_reg, loss_cnt, writer)
        writer.add_scalar("Test/Loss", test_loss, test_iterations)


if __name__ == "__main__":
    main()
