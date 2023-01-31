import torch
import torch.nn as nn
import gin.torch
from src.utils import BackboneFPN
from src.utils import SegmentationHead


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOS(torch.nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=1,
        fpn_channels=128,
        tower_depth=4,
        strides=[8, 16, 32, 64, 128],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.tower_depth = tower_depth
        self.strides = torch.tensor(strides)
        self.backbone_fpn = BackboneFPN(
            backbone=backbone, depth=self.fpn_channels, return_list=True
        )
        self.cls_tower = nn.ModuleList()
        self.bbox_tower = nn.ModuleList()
        self.segmentation_head = SegmentationHead(self.fpn_channels, self.tower_depth)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        for _ in range(tower_depth):
            self.cls_tower.append(
                nn.Conv2d(
                    self.backbone_fpn.depth, self.backbone_fpn.depth, 3, padding=1
                )
            )
            self.cls_tower.append(nn.GroupNorm(32, self.backbone_fpn.depth))
            self.cls_tower.append(nn.GELU())
            self.bbox_tower.append(
                nn.Conv2d(
                    self.backbone_fpn.depth, self.backbone_fpn.depth, 3, padding=1
                )
            )
            self.bbox_tower.append(nn.GroupNorm(32, self.backbone_fpn.depth))
            self.bbox_tower.append(nn.GELU())

        self.cls_tower = nn.Sequential(*self.cls_tower)
        self.bbox_tower = nn.Sequential(*self.bbox_tower)
        if self.num_classes == 1:
            self.cls = nn.Sequential(
                nn.Conv2d(self.backbone_fpn.depth, self.num_classes, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.cls = nn.Sequential(
                nn.Conv2d(self.backbone_fpn.depth, self.num_classes, 3, padding=1),
            )
        self.cnt = nn.Conv2d(self.backbone_fpn.depth, 1, 3, padding=1)
        self.reg = nn.Conv2d(self.backbone_fpn.depth, 4, 3, padding=1)

        for modules in [self.cls_tower, self.bbox_tower, self.reg]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        for modules in [self.cls, self.cnt]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(
                        l.bias, -torch.log(torch.tensor((1 - 0.01) / 0.01))
                    )

    def forward(self, x):
        x = self.backbone_fpn(x)
        da = self.segmentation_head(x[0])

        cls = []
        reg = []
        cnt = []

        for i, fpn_map in enumerate(x):
            cls_tower_map = self.cls_tower(fpn_map)
            reg_tower_map = self.bbox_tower(fpn_map)

            cls.append(self.cls(cls_tower_map))
            cnt.append(self.cnt(cls_tower_map))
            reg.append(self.scales[i](torch.exp(self.reg(reg_tower_map) - 1.0)))

        return cls, reg, cnt, da
