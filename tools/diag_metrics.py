"""诊断评估指标问题：验证集类别直方图 + 用混淆矩阵解释 nan 的来源。"""
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "datasets", "VKITTI_II")
CLASSES = ["Terrain", "Tree", "Vegetation", "Building", "Road", "GuardRail",
           "TrafficSign", "TrafficLight", "Pole", "Misc", "Truck", "Car",
           "Van", "Undefined", "未标注/背景"]

for split in ("training", "validation"):
    d = os.path.join(ROOT, "annotations", split)
    files = sorted(os.listdir(d))
    hist = np.zeros(15, dtype=np.int64)
    for f in files:
        a = np.asarray(Image.open(os.path.join(d, f)))
        hist += np.bincount(a.ravel(), minlength=15)[:15]
    tot = hist.sum()
    print("=" * 74)
    print(f"{split}  共 {len(files):,} 帧, {tot:,} 像素")
    print("=" * 74)
    print(f"{'类':>3} {'名称':<16} {'像素数':>18} {'占比':>9}  出现?")
    for i in range(15):
        mark = "是" if hist[i] > 0 else "★从未出现★"
        print(f"{i:>3} {CLASSES[i]:<16} {hist[i]:>18,} {hist[i]/tot:>8.3%}  {mark}")
    missing = [i for i in range(15) if hist[i] == 0]
    print(f"\n  从未在 GT 中出现的类别: {missing}")
    if missing:
        print("  -> 这些类别的 IoU = 0/0 = nan，会让 mean IoU 变成 nan")
    print(f"  类别 0 (Terrain) 占比: {hist[0]/tot:.2%}  <- 被 ignore_index=0 排除，不参与损失")

print()
print("=" * 74)
print("说明：Utils/distributed_utils.py 的 ConfusionMatrix.compute() 里")
print("  iu = diag / (row + col - diag)")
print("对从未出现的类别，分子分母同为 0 -> 0/0 -> nan，")
print("再取 mean 就把整个 mean IoU 变成 nan。这是指标实现的缺陷，不是训练失败。")
print("=" * 74)
