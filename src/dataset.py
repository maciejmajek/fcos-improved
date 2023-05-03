import os
import json
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.transforms.functional import InterpolationMode
from src.utils import BoxList
from src.utils import get_targets
from src.utils import object_sizes_of_interest


cat_to_num = {
    "pedestrian": 1,
    "rider": 0,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 0,
    "motorcycle": 5,
    "bicycle": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "other vehicle": 0,
    "other person": 0,
    "trailer": 0,
}


def get_labels(det, name):
    try:
        return det[name]["labels"]
    except KeyError:
        return None


def get_boxes(labels):
    boxes = []
    for box in labels:
        x1, y1, x2, y2 = (
            box["box2d"]["x1"],
            box["box2d"]["y1"],
            box["box2d"]["x2"],
            box["box2d"]["y2"],
        )
        boxes.append([x1, y1, x2, y2])
    return boxes


def get_categories(labels):
    categories = []
    for box in labels:
        categories.append(cat_to_num[box["category"]])
    return categories


def get_box_list(boxes, h=720, w=1280):
    box_list = BoxList(boxes, (w, h))
    return box_list


def get_trainable_targets(data, name, sort=False, reverse=True):
    labels = get_labels(data, name)
    if labels == None:
        return None
    boxes = get_boxes(labels)
    if sort:
        boxes_sorted = sorted(
            boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=reverse
        )
        indexes = [boxes.index(value) for value in boxes_sorted]
        labels = [labels[idx] for idx in indexes]
        boxes = boxes_sorted

    boxes = get_box_list(boxes)
    categories = get_categories(labels)

    h, w = 6 * 128, 9 * 128
    boxes = boxes.resize((w, h))
    boxes.extra_fields["labels"] = torch.tensor(categories)
    return boxes


bad_ids = [
    6879,
    15376,
]


class BDD100K(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split,
        size=100_000,
        return_detection=True,
        return_drivable_area=False,
        return_bboxes=False,
        test_mode=False,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.size = size
        self.return_detection = return_detection
        self.return_drivable = return_drivable_area
        self.return_bboxes = return_bboxes
        self.test_mode = test_mode
        self.det = json.load(
            open(f"{self.root}/labels/det_20/det_{self.split}.json", "r")
        )
        self.images = os.listdir(f"{self.root}/images/100k/{self.split}")
        self.images_path = f"{self.root}/images/100k/{self.split}"
        self.drivable_path = f"{self.root}/labels/drivable/masks/{self.split}"
        self._preprocess_det()
        self._preprocess_images()

    def _preprocess_det(self):
        det_temp = {}
        for sample in self.det:
            name = sample.pop("name")
            det_temp[name] = sample
        self.det = det_temp

    def _preprocess_images(self):
        if self.split != "train":
            return
        for index in sorted(bad_ids, reverse=True):
            del self.images[index]

    def _get_final(self, data, idx):
        strides = torch.tensor([8, 16, 32, 64, 128])
        h, w = 6 * 128, 9 * 128
        resize = T.Resize((h, w))
        resize_no_interp = T.Resize(
            (h // strides[0], w // strides[0]), interpolation=InterpolationMode.NEAREST
        )
        img = read_image(self.images_path + "/" + self.images[idx])
        img = resize(img)
        img = img / 255.0
        returns = [img]
        if self.test_mode:
            box_list = get_trainable_targets(
                data, self.images[idx], sort=True, reverse=False
            )
            returns.append(box_list)
            return returns
        box_list = None
        if self.return_detection:
            box_list = get_trainable_targets(
                data, self.images[idx], sort=True, reverse=False
            )
            maps_cls, maps_reg, maps_cnt = get_targets(
                box_list, strides, object_sizes_of_interest, "cpu"
            )
            returns += [maps_cls, maps_reg, maps_cnt]

        if self.return_drivable:
            drivable_area = read_image(
                f"{self.drivable_path}/{self.images[idx].replace('jpg', 'png')}"
            )
            drivable_area[drivable_area != 2] = 1
            drivable_area[drivable_area == 2] = 0
            drivable_area = resize_no_interp(drivable_area)
            returns.append(drivable_area.long())

        if self.return_bboxes and box_list != None:
            returns.append(box_list)
        return returns

    def __getitem__(self, idx):
        return self._get_final(self.det, idx)

    def __len__(self):
        return min(self.size, len(self.images))
