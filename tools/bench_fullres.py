"""测量全分辨率推理的显存与速度，用于判断评估阶段是否还受显存限制。

VKITTI 2 原图为 1242x375；现有 PipelineEval 只裁 256x256（约 13.5% 像素）。
本脚本直接跑整图，看 24GB 卡是否吃得下。
"""
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from Joint.model import SDCombo  # noqa: E402

device = "cuda"
model = SDCombo(15).to(device).eval()
print(f"GPU: {torch.cuda.get_device_name(0)}  "
      f"total={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M\n")

print(f"{'输入尺寸':>14} | {'像素占比':>8} | {'峰值显存':>10} | {'耗时':>8}")
print("-" * 52)
BASE = 1242 * 375

for shape in [(1, 3, 256, 256), (1, 3, 375, 1242), (1, 3, 512, 512), (1, 3, 750, 1242)]:
    h, w = shape[-2], shape[-1]
    img = torch.randn(*shape, device=device)
    dep = torch.randint(1, 65535, (shape[0], h, w), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(img, dep)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"{str(h) + 'x' + str(w):>14} | {h * w / BASE:>7.0%} | "
              f"{peak:>8.2f} GiB | {dt:>6.2f} s")
    except RuntimeError as exc:
        print(f"{str(h) + 'x' + str(w):>14} | {h * w / BASE:>7.0%} | "
              f"OOM -> {str(exc)[:40]}")
    del img, dep

# 训练态（含反向）在 512 crop 下能吃多大 batch
# 注意：真实训练中 input/depth 不需要梯度（只有模型参数需要），
# 若给 input 设 requires_grad=True 会额外保留整张计算图，虚增显存。
print("\n--- 训练态（前向+反向）显存随 batch 变化, crop=512 ---")
model.train()
for bs in (2, 4, 6, 8):
    img = torch.randn(bs, 3, 512, 512, device=device)
    dep = torch.randint(1, 65535, (bs, 512, 512), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        out = model(img, dep)
        out.sum().backward()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        print(f"  batch={bs}  峰值 allocated={peak:.2f} GiB  reserved={reserved:.2f} GiB")
    except RuntimeError:
        print(f"  batch={bs}  OOM")
    del img, dep
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

# 热身后的 256 crop 计时（排除一次性 kernel autotune）
print("\n--- 256x256 推理计时（排除首次 autotune）---")
model.eval()
img = torch.randn(1, 3, 256, 256, device=device)
dep = torch.randint(1, 65535, (1, 256, 256), device=device)
with torch.no_grad():
    for _ in range(3):
        model(img, dep)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        model(img, dep)
    torch.cuda.synchronize()
    print(f"  256x256: {(time.perf_counter() - t0) / 20 * 1000:.1f} ms/次")
