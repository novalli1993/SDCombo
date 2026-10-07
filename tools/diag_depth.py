"""检查深度数据分布，为 SDCHead 的深度归一化选一个物理上合理的尺度。

背景：VKITTI 2 的深度是 16bit，1 单位 = 1cm，远平面 655.35m（=65535，被裁剪）。
原实现用 torch.max(depth) 归一化，而几乎每张图都有 65535 的裁剪像素，
导致真实深度（如 50m 路面 = 5000）被压到 0.076，深度分支实际上近乎常数。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for split in ("training", "validation"):
    d = os.path.join(ROOT, "datasets", "VKITTI_II", "depth", split)
    files = sorted(os.listdir(d))[::max(1, len(os.listdir(d)) // 60)][:60]
    samples = [np.asarray(Image.open(os.path.join(d, f))).astype(np.int64) for f in files]
    allpx = np.concatenate([s.ravel() for s in samples])
    print("=" * 72)
    print(f"{split}: 抽样 {len(files)} 帧, {allpx.size:,} 像素")
    print("=" * 72)
    pcts = [0, 1, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    vals = np.percentile(allpx, pcts)
    print(f"{'百分位':>8} {'原始(1cm)':>12} {'米':>10}")
    for p, v in zip(pcts, vals):
        print(f"{p:>8} {v:>12.0f} {v/100:>10.1f}")
    print(f"\n  每帧最大值: min={min(int(s.max()) for s in samples)}, "
          f"等于 65535 的帧数={sum(1 for s in samples if s.max() >= 65535)}/{len(samples)}")
    print(f"  -> 用 torch.max 归一化时，分母几乎恒为 65535")
    med = np.median(allpx)
    print(f"\n  中位深度 {med:.0f} (={med/100:.1f} m)")
    print(f"  若用 max 归一化，中位像素被压到 {med/65535:.4f}")
    print(f"  若用 655.35m 归一化，中位像素为 {med/65535:.4f}"
          f"（尺度相同，但至少数值稳定、不依赖 .item()）")
    print(f"  若用 99.9 百分位 ({vals[pcts.index(99.9)]:.0f}) 归一化，"
          f"中位像素为 {med/vals[pcts.index(99.9)]:.4f}")
