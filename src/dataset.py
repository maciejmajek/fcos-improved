import os
import json
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from src.utils import BoxList
from src.utils import get_targets
from src.utils import object_sizes_of_interest


categories = {
    1: "pedestrian",
    2: "rider",
    3: "car",
    4: "truck",
    5: "bus",
    6: "train",
    7: "motorcycle",
    8: "bicycle",
    9: "traffic light",
    10: "traffic sign",
    0: "other vehicle",
}
cat_to_num = {v: k for k, v in categories.items()}

cat_to_num = {
    "pedestrian": 1,
    "rider": 1,
    "car": 1,
    "truck": 1,
    "bus": 1,
    "train": 1,
    "motorcycle": 1,
    "bicycle": 1,
    "traffic light": 1,
    "traffic sign": 1,
    "other vehicle": 1,
    "other person": 1,
    "trailer": 1,
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
    box_list = BoxList(boxes, (h, w))
    return box_list


def get_trainable_targets(data, name):
    labels = get_labels(data, name)
    if labels == None:
        return None
    boxes = get_boxes(labels)
    boxes = get_box_list(boxes)
    categories = get_categories(labels)
    h, w = 6 * 128, 9 * 128
    boxes = boxes.resize((h, w))
    boxes.extra_fields["labels"] = torch.tensor(categories)
    return boxes


bad_ids = [
    6879,
    15376,
    15763,
    19525,
    19705,
    28516,
    28852,
    31942,
    33295,
    33967,
    34052,
    34274,
    34410,
    34531,
    34562,
    34651,
    34669,
    35833,
    35890,
    35927,
    35967,
    35971,
    36039,
    36290,
    36421,
    36489,
    36497,
    36513,
    36564,
    36614,
    36616,
    36888,
    36920,
    36923,
    37078,
    37094,
    37235,
    37245,
    37330,
    37331,
    37336,
    37587,
    37682,
    37818,
    37864,
    37898,
    37935,
    37944,
    38177,
    38186,
    38264,
    38339,
    38544,
    38591,
    38783,
    39109,
    39211,
    39243,
    39387,
    39437,
    39476,
    39604,
    39687,
    39767,
    39825,
    39860,
    40128,
    40153,
    40180,
    40291,
    40345,
    40380,
    40454,
    40462,
    40610,
    40636,
    40666,
    40667,
    40804,
    41020,
    41049,
    41060,
    41139,
    41263,
    41298,
    41485,
    41624,
    41692,
    41746,
    41942,
    42221,
    42694,
    42802,
    42942,
    43088,
    43298,
    43484,
    43638,
    43781,
    43808,
    43829,
    44108,
    44220,
    44511,
    44572,
    44692,
    45161,
    45193,
    45288,
    45302,
    45435,
    45710,
    45755,
    45816,
    46006,
    46178,
    46240,
    46257,
    46440,
    46537,
    46570,
    46751,
    46818,
    46845,
    46999,
    47032,
    47104,
    47168,
    47385,
    47578,
    47641,
    47734,
    47771,
    47830,
    47939,
    47993,
    48060,
    48181,
    48205,
    48257,
    48264,
    48821,
    49006,
    49123,
    49307,
    49390,
    49422,
]


class BDD100K(torch.utils.data.Dataset):
    def __init__(self, root, split, size=100_000):
        super().__init__()
        self.root = root
        self.split = split
        self.size = size
        self.det = json.load(
            open(f"{self.root}/labels/det_20/det_{self.split}.json", "r")
        )
        self.images = os.listdir(f"{self.root}/images/100k/{self.split}")
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

        box_list = get_trainable_targets(data, self.images[idx])
        maps_cls, maps_reg, maps_cnt = get_targets(
            box_list, strides, object_sizes_of_interest, "cpu"
        )

        h, w = 6 * 128, 9 * 128
        resize = T.Resize((h, w))
        img = read_image(
            self.root + r"images/100k/" + self.split + "/" + self.images[idx]
        )
        img = resize(img)
        img = img / 255.0
        return img, maps_cls, maps_reg, maps_cnt

    def __getitem__(self, idx):
        return self._get_final(self.det, idx)

    def __len__(self):
        return min(self.size, len(self.images))
