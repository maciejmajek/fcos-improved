import torch
import torch.nn as nn
import gin.torch
from src.utils import BackboneFPN

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
        self.exp_params = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(8))) for i in range(5)]
        )

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
        for modules in [self.cls_tower, self.bbox_tower, self.cls, self.reg, self.cnt]:

            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = self.backbone_fpn(x)
        cls = []
        reg = []
        cnt = []

        for i, l in enumerate([self.cls_tower(i) for i in x]):
            cls.append(self.cls(l))
            cnt.append(self.cnt(l))

        for i, l in enumerate([self.bbox_tower(i) for i in x]):
            reg.append(torch.exp(self.reg(l) * self.exp_params[i]))

        return cls, reg, cnt
