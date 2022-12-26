import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torchvision.models.mobilenetv3 import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.convnext import convnext_tiny, ConvNeXt_Tiny_Weights

INF = 10 ** 10
h, w = 6 * 128, 9 * 128
overlap = 0.5  # Not implemented

object_sizes_of_interest = [
    [-1, 64 * overlap],
    [overlap * 64, overlap * 128],
    [overlap * 128, overlap * 256],
    [overlap * 256, overlap * 512],
    [overlap * 512, overlap * INF],
]


def get_level(size, sizes):
    i = 0
    for i, l in enumerate(sizes):
        if l[0] <= size < l[1]:
            return i
    return -1


def get_levels(dx, dy, object_sizes_of_interest):
    levels = []
    for val in torch.stack((dx, dy)).min(dim=0).values:
        levels.append(get_level(val, object_sizes_of_interest))
    return levels


def locations_inside_box(box):
    x = torch.arange(box[0], box[2])
    y = torch.arange(box[1], box[3])
    return torch.cartesian_prod(x, y)


def calculate_centerness(l, t, r, b, sign="multiplication"):
    if sign == "addition":
        return torch.sqrt(
            torch.min(l, r) / torch.max(l, r) + torch.min(t, b) / torch.max(t, b)
        )
    if sign == "multiplication":
        return torch.sqrt(
            torch.min(l, r) / torch.max(l, r) * torch.min(t, b) / torch.max(t, b)
        )
    raise "Not implemented"


def prepare_box(box, stride):
    box = box / stride
    return box


def extend_box(box):
    box = box.round()
    box = box.int()
    return box


def get_cls_target(boxes, strides, box_target_stride, labels, device):
    maps_cls = {
        int(stride): torch.zeros((int(h / stride), int(w / stride)), device=device)
        for stride in strides
    }
    if boxes == None:
        return maps_cls
    for ij, (box, stride) in enumerate(zip(boxes.bbox, box_target_stride)):
        box = prepare_box(box, stride)
        box = extend_box(box)

        locations = locations_inside_box(box)
        if locations == None:
            continue
        for i, j in locations:
            if (
                j < maps_cls[int(stride)].shape[0]
                and i < maps_cls[int(stride)].shape[1]
            ):
                maps_cls[int(stride)][j, i] = labels[ij]

        assert maps_cls[int(stride)].min() >= 0.0
        assert torch.isnan(maps_cls[int(stride)]).any() == False, "NaN detected"
    return maps_cls


def get_reg_target(boxes, strides, box_target_stride, device):
    maps_reg = {
        int(stride): torch.zeros((4, int(h / stride), int(w / stride)), device=device)
        for stride in strides
    }
    if boxes == None:
        return maps_reg
    for ij, (box, stride) in enumerate(zip(boxes.bbox, box_target_stride)):
        box = prepare_box(box, stride)
        extended_box = extend_box(box.clone())

        locations = locations_inside_box(extended_box)

        if locations == None:
            continue

        for i, j in locations:
            l = i - box[0]
            r = box[2] - i
            t = j - box[1]
            b = box[3] - j
            l = torch.max(torch.tensor(0.0), l)
            t = torch.max(torch.tensor(0.0), t)
            r = torch.max(torch.tensor(0.0), r)
            b = torch.max(torch.tensor(0.0), b)

            try:
                maps_reg[int(stride)][:, j, i] = torch.tensor([l, t, r, b])
            except IndexError as e:
                pass  # print(e)

        assert maps_reg[int(stride)].min() >= 0.0
        assert torch.isnan(maps_reg[int(stride)]).any() == False, "NaN detected"

    return maps_reg


def get_cnt_target(boxes, strides, maps_reg, device):
    maps_cnt = {
        int(stride): torch.zeros((int(h / stride), int(w / stride)), device=device)
        for stride in strides
    }
    if boxes == None:
        return maps_cnt
    for key, value in maps_reg.items():
        l, t, r, b = value
        maps_cnt[key] = calculate_centerness(l, t, r, b)
        maps_cnt[key][maps_cnt[key].isnan()] = 0
        maps_cnt[key][maps_cnt[key].isinf()] = 0
        assert maps_cnt[key].max() <= 1.0, f"Max is {maps_cnt[key].max()}"
        assert maps_cnt[key].min() >= 0.0, f"Min is {maps_cnt[key].min()}"
        assert torch.isnan(maps_cnt[key]).any() == False, "NaN detected"
    return maps_cnt


def criterium(boxes, strides, sizes):
    x1, y1 = boxes.bbox[:, 0], boxes.bbox[:, 1]
    x2, y2 = boxes.bbox[:, 2], boxes.bbox[:, 3]
    dx = x2 - x1
    dy = y2 - y1
    levels = get_levels(dx, dy, sizes)
    return strides[levels]


def get_targets(boxes, strides, sizes, device):
    if boxes == None:
        maps_cls = get_cls_target(boxes, strides, [], [], device)
        maps_reg = get_reg_target(boxes, strides, [], device)
        maps_cnt = get_cnt_target(boxes, strides, maps_reg, device)
        return maps_cls, maps_reg, maps_cnt

    labels = boxes.get_field("labels")
    box_target_stride = criterium(boxes, strides, sizes)

    maps_cls = get_cls_target(boxes, strides, box_target_stride, labels, device)
    maps_reg = get_reg_target(boxes, strides, box_target_stride, device)
    maps_cnt = get_cnt_target(boxes, strides, maps_reg, device)
    for key in maps_cls:
        maps_cnt[key] = maps_cnt[key] * (maps_cls[key] != 0)
        reg = torch.zeros_like(maps_reg[key])
        reg[:, maps_cls[key].bool()] = maps_reg[key][:, maps_cls[key].bool()]
        maps_reg[key] = reg
    return maps_cls, maps_reg, maps_cnt


FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xyxy"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        if bbox.size(-1) != 4:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh"):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.size, mode=mode)
        else:
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.size, mode=mode)
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_height, ratio_width = ratios[::-1]
        xmin, ymin, xmax, ymax = self._split_into_xyxy()

        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_xmin = image_width - xmax - TO_REMOVE
            transposed_xmax = image_width - xmin - TO_REMOVE
            transposed_ymin = ymin
            transposed_ymax = ymax
        elif method == FLIP_TOP_BOTTOM:
            transposed_xmin = xmin
            transposed_xmax = xmax
            transposed_ymin = image_height - ymax
            transposed_ymax = image_height - ymin

        transposed_boxes = torch.cat(
            (transposed_xmin, transposed_ymin, transposed_xmax, transposed_ymax), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.size, mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
        cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
        cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
        cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
        )
        bbox = BoxList(cropped_box, (w, h), mode="xyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # Tensor-like methods

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        if remove_empty:
            box = self.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (
                box[:, 3] - box[:, 1] + TO_REMOVE
            )
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


class Resnet18Bacbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.pre = torch.nn.Sequential(
            self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
        )
        self.l1 = self.model.layer1
        self.l2 = self.model.layer2
        self.l3 = self.model.layer3
        self.l4 = self.model.layer4
        self.depth_channels = [128, 256, 512]

    def forward(self, x):
        x = self.pre(x)
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        output = OrderedDict()
        output["feat0"] = x2
        output["feat1"] = x3
        output["feat2"] = x4
        return output


class Resnet50Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.pre = torch.nn.Sequential(
            self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
        )
        self.l1 = self.model.layer1
        self.l2 = self.model.layer2
        self.l3 = self.model.layer3
        self.l4 = self.model.layer4
        self.depth_channels = [512, 1024, 2048]

    def forward(self, x):
        x = self.pre(x)
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        output = OrderedDict()
        output["feat0"] = x2
        output["feat1"] = x3
        output["feat2"] = x4
        return output


class MobileNetBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.l1 = self.model.features[0:4]
        self.l2 = self.model.features[4:9]
        self.l3 = self.model.features[9:13]

        self.depth_channels = [24, 48, 576]

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        output = OrderedDict()
        output["feat0"] = x1
        output["feat1"] = x2
        output["feat2"] = x3
        return output


class ConvNextBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        del self.model.classifier
        self.pre = self.model.features[0:2]
        self.l1 = self.model.features[2:4]
        self.l2 = self.model.features[4:6]
        self.l3 = self.model.features[6:8]

        self.depth_channels = [192, 384, 768]

    def forward(self, x):
        x = self.pre(x)
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        output = OrderedDict()
        output["feat0"] = x1
        output["feat1"] = x2
        output["feat2"] = x3
        return output


class FPN_P6P7(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.p6 = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1, stride=2),
            torch.nn.GELU(),
        )
        self.p7 = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1, stride=2),
            torch.nn.GELU(),
        )

    def forward(self, x):
        p = OrderedDict()
        p["feat3"] = self.p6(x)
        p["feat4"] = self.p7(p["feat3"])
        return p


class BackboneFPN(torch.nn.Module):
    def __init__(self, backbone, depth, return_list=False):
        super().__init__()
        self.depth = depth
        self.return_list = return_list
        self.backbone = backbone()
        self.fpn = FeaturePyramidNetwork(self.backbone.depth_channels, self.depth)
        self.fpn_top = FPN_P6P7(self.depth)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        x1 = self.fpn_top(x["feat2"])
        x["feat3"] = x1["feat3"]
        x["feat4"] = x1["feat4"]
        if self.return_list:
            return list(x.values())
        return x


class SegmentationHead(nn.Module):
    def __init__(self, fpn_depth=128, tower_depth=4, num_classes=1):
        super().__init__()
        self.fpn_depth = fpn_depth
        self.tower_depth = tower_depth
        self.num_classes = num_classes
        self.head_tower = nn.ModuleList()
        self.cls = nn.Conv2d(self.fpn_depth, self.num_classes, 3, padding=1)
        for _ in range(self.tower_depth):
            self.head_tower.append(
                nn.Conv2d(self.fpn_depth, self.fpn_depth, 3, padding=1)
            )
            self.head_tower.append(nn.GroupNorm(32, self.fpn_depth))
            self.head_tower.append(nn.GELU())

    def forward(self, x):
        for layer in self.head_tower:
            x = layer(x)
        x = self.cls(x)
        return x
