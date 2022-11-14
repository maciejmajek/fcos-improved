import torch
import torch.nn as nn
import gin.torch
from src.utils import BackboneFPN


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@gin.configurable
class FCOS(torch.nn.Module):
    def __init__(self, backbone_depth=128, tower_depth=4):
        super().__init__()
        self.backbone_depth = backbone_depth
        self.tower_depth = tower_depth
        self.strides = torch.tensor([8, 16, 32, 64, 128])
        self.backbone_fpn = BackboneFPN(depth=self.backbone_depth, return_list=True)
        self.cls_tower = nn.ModuleList()
        self.bbox_tower = nn.ModuleList()

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        for _ in range(tower_depth):
            self.cls_tower.append(
                nn.Conv2d(
                    self.backbone_fpn.depth, self.backbone_fpn.depth, 3, padding=1
                )
            )
            self.cls_tower.append(nn.GroupNorm(32, self.backbone_fpn.depth))
            self.cls_tower.append(nn.ReLU())
            self.bbox_tower.append(
                nn.Conv2d(
                    self.backbone_fpn.depth, self.backbone_fpn.depth, 3, padding=1
                )
            )
            self.bbox_tower.append(nn.GroupNorm(32, self.backbone_fpn.depth))
            self.bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*self.cls_tower)
        self.bbox_tower = nn.Sequential(*self.bbox_tower)

        self.cls = nn.Sequential(
            nn.Conv2d(self.backbone_fpn.depth, 1, 3, padding=1), nn.Sigmoid()
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
        cls = []
        reg = []
        cnt = []

        for i, l in enumerate([self.cls_tower(i) for i in x]):
            cls.append(self.cls(l))
            cnt.append(self.cnt(l))

        for i, l in enumerate([self.bbox_tower(i) for i in x]):
            box_pred = self.scales[i](self.reg(l))
            reg.append(torch.exp(box_pred))

        for i in range(len(self.strides)):
            assert (
                reg[i].min() >= 0.0
            ), f"Min is {reg[i].min()} Max is {reg[i].max()} stride: {self.strides[i]} p: {self.scales[i].scale}"
            assert (
                cls[i].min() >= 0.0
            ), f"Min is {cls[i].min()} Max is {cls[i].max()} stride: {self.strides[i]} p: {self.scales[i].scale}"
            assert (
                cls[i].max() <= 1.0
            ), f"Min is {cls[i].min()} Max is {cls[i].max()} stride: {self.strides[i]} p: {self.scales[i].scale}"

        return cls, reg, cnt
