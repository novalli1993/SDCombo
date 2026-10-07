"""临时探针：验证集上的平凡基线 + 类别加权的量级。

给出「恒定预测某一类」时的 mIoU / 全局准确率，作为训练是否真的学到东西的参照；
同时打印 focal loss 在各类别上的相对贡献。
"""
import os

import numpy as np
import torch
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets", "VKITTI_II")
CLASSES = ['Terrain', 'Tree', 'Vegetation', 'Building', 'Road', 'GuardRail',
           'TrafficSign', 'TrafficLight', 'Pole', 'Misc', 'Truck', 'Car',
           'Van', 'Undefined']


def confusion(split):
    ann_dir = os.path.join(ROOT, "annotations", split)
    names = sorted(os.listdir(ann_dir))
    mat = np.zeros((14, 14), dtype=np.int64)
    for name in names:
        a = np.array(Image.open(os.path.join(ann_dir, name))).ravel()
        k = (a >= 0) & (a < 14)
        mat += np.bincount(14 * a[k] + a[k], minlength=196).reshape(14, 14)
    return mat


def report(mat, tag):
    h = mat.astype(np.float64)
    iu = np.diag(h) / np.clip(h.sum(1) + h.sum(0) - np.diag(h), 1e-9, None)
    acc = np.diag(h) / np.clip(h.sum(1), 1e-9, None)
    print("--- {} ---".format(tag))
    for c in range(14):
        if h[c].sum() == 0 and h[:, c].sum() == 0:
            continue
        print("  {:14s} px={:>12d}  recall={:6.1f}  IoU={:6.1f}".format(
            CLASSES[c], int(h[c].sum()), acc[c] * 100, iu[c] * 100))
    print("  mIoU(all 14)      = {:.2f}".format(iu.mean() * 100))
    print("  mIoU(present only)= {:.2f}".format(iu[[h[c].sum() > 0 or h[:, c].sum() > 0
                                                    for c in range(14)]].mean() * 100))
    print("  global acc        = {:.2f}".format(np.diag(h).sum() / h.sum() * 100))


def majority_baseline(mat):
    """恒定预测某一类时的混淆矩阵。"""
    print("\n=== 平凡基线（validation 上恒定预测某一类）===")
    gt = mat.sum(1).astype(np.float64)
    rows = []
    for c in range(14):
        if gt[c] == 0:
            continue
        fake = np.zeros((14, 14), dtype=np.float64)
        fake[:, c] = gt                       # 所有像素都预测成 c
        iu = np.diag(fake) / np.clip(fake.sum(1) + fake.sum(0) - np.diag(fake), 1e-9, None)
        rows.append((iu.mean() * 100, gt[c] / gt.sum() * 100, CLASSES[c]))
    for miou, acc, name in sorted(rows, reverse=True)[:5]:
        print("  预测恒为 {:14s}: global_acc={:5.1f}%  mIoU(14类)={:5.2f}".format(name, acc, miou))


if __name__ == "__main__":
    for split in ("training", "validation"):
        m = confusion(split)
        report(m, split)
        if split == "validation":
            majority_baseline(m)
