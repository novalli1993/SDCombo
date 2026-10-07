"""抽查数据集：三目录对齐、标签取值、深度是否为 16bit、以及原图分辨率。"""
import os
import random
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "datasets", "VKITTI_II")
random.seed(0)
bad = 0

for split in ("training", "validation"):
    print("=" * 66)
    print(f"split = {split}")
    print("=" * 66)
    dirs = {k: os.path.join(ROOT, k, split) for k in ("images", "annotations", "depth")}
    stems = {k: {os.path.splitext(f)[0] for f in os.listdir(v)} for k, v in dirs.items()}
    print(f"文件数: images={len(stems['images'])} annotations={len(stems['annotations'])} "
          f"depth={len(stems['depth'])}")
    same = stems["images"] == stems["annotations"] == stems["depth"]
    print(f"主文件名集合一致: {same}")
    if not same:
        bad += 1

    samples = random.sample(sorted(stems["images"]), 5)
    print(f"\n{'帧':<30} {'原图':<14} {'标签':<12} {'深度(mode/max)':<22}")
    print("-" * 84)
    for st in samples:
        rgb_p = os.path.join(dirs["images"], st + ".jpg")
        if not os.path.exists(rgb_p):
            cand = [f for f in os.listdir(dirs["images"]) if f.startswith(st)]
            rgb_p = os.path.join(dirs["images"], cand[0])
        rgb = Image.open(rgb_p)
        ann = np.array(Image.open(os.path.join(dirs["annotations"], st + ".png")))
        dep_img = Image.open(os.path.join(dirs["depth"], st + ".png"))
        dep = np.array(dep_img)

        shape_ok = (ann.shape == dep.shape == (rgb.size[1], rgb.size[0]))
        vals_ok = ann.min() >= 0 and ann.max() <= 14 and ann.ndim == 2
        dep_ok = dep.dtype == np.uint16 or dep.max() > 255
        if not (shape_ok and vals_ok and dep_ok):
            bad += 1
        flag = "" if (shape_ok and vals_ok and dep_ok) else "  ★"
        print(f"{st:<30} {str(rgb.size):<14} {str(ann.shape):<12} "
              f"{dep_img.mode}/{dep.max():<10} {dep.dtype}{flag}")
        if len(samples) <= 3:
            print(f"    标签直方图: {np.bincount(ann.ravel(), minlength=15).tolist()}")

    # 深度一次性统计（抽样 30 张）
    mx = []
    for st in random.sample(sorted(stems["depth"]), min(30, len(stems["depth"]))):
        mx.append(int(np.array(Image.open(os.path.join(dirs["depth"], st + ".png"))).max()))
    print(f"\n深度抽样 30 张 max: min={min(mx)} median={int(np.median(mx))} max={max(mx)}")
    print(f"  （16bit 值域，1 单位 = 1cm；若都是 <=255 说明被截成了 8bit）")

print()
print("=" * 66)
print("抽查通过" if bad == 0 else f"抽查发现 {bad} 处问题")
print("=" * 66)
sys.exit(1 if bad else 0)
