import cv2
import torch
import torch.nn as nn
from torchvision.ops import nms


def box_to_bb(center, ltrb):
    i, j = center[1], center[0]
    l, t, r, b = ltrb
    box = [-1, -1, -1, -1]
    box[0] = i - l
    box[1] = j - t
    box[2] = r + i
    box[3] = b + j
    return box


def get_predicted_boxes(cls_pred, cnt_pred, reg_pred, confidence):
    strides = [8, 16, 32, 64, 128]
    boxes = []
    all_labels = []
    all_scores = []
    i = 0
    for cls, cnt, reg in zip(cls_pred, cnt_pred, reg_pred):
        cls = nn.Softmax(dim=1)(cls)
        cnt = nn.Sigmoid()(cnt)

        out = cls * cnt
        scores, labels = out.max(1)
        labels = labels[scores > confidence]
        regressions = reg.permute(0, 2, 3, 1)[scores > confidence]
        regressions_indexes = torch.stack(
            torch.meshgrid(
                torch.arange(0, cls.shape[2]), torch.arange(0, cls.shape[3])
            ),
            dim=2,
        ).unsqueeze(0)
        regressions_indexes = regressions_indexes[scores > confidence]
        scores = scores[scores > confidence]

        scores = scores[labels != 0]
        regressions = regressions[labels != 0]
        regressions_indexes = regressions_indexes[labels != 0]
        labels = labels[labels != 0]

        all_labels.append(labels)
        all_scores.append(scores)
        boxes.append(
            predictions_to_boxes(
                labels,
                scores,
                regressions * strides[i],
                regressions_indexes * strides[i],
            )
        )
        i += 1

    return torch.cat(boxes, dim=0), torch.cat(all_labels), torch.cat(all_scores)


def predictions_to_boxes(labels, scores, regressions, regressions_indexes):
    bb = box_to_bb(regressions_indexes.permute(1, 0), regressions.permute(1, 0))
    bb = torch.stack(bb).permute(1, 0)

    return bb


def get_boxes_from_predictions(cls_pred, cnt_pred, reg_pred, confidence, iou_threshold):
    bbs, labels, scores = get_predicted_boxes(cls_pred, cnt_pred, reg_pred, confidence)
    indices = nms(bbs, scores, iou_threshold)
    bbs, labels, scores = bbs[indices], labels[indices], scores[indices]
    return bbs, labels, scores


def visualize_boxes(img, bbs, labels, scores, colors):
    img = img.permute(1, 2, 0).numpy().copy()
    for box, label, score in zip(bbs, labels, scores):
        box = box.int().numpy()
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
        img = cv2.putText(
            img,
            str(int(label)),
            (box[0], box[3]),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 0),
            1,
        )
        img = cv2.putText(
            img,
            f"{score:.2f}",
            (box[0], box[1] + 12),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 0, 0),
            1,
        )
    return img
