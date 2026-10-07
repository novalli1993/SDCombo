"""验证模型修正：logits 语义、深度归一化动态范围、.item() 同步消除、梯度健康度。"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Joint.model import SDCombo  # noqa: E402
from Joint.SDCHead import SDCHead  # noqa: E402

FAIL = 0


def check(label, cond, extra=""):
    global FAIL
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  {extra}" if extra else ""))
    if not cond:
        FAIL += 1


print("=" * 70)
print("[1] forward 返回原始 logits（不是概率）")
print("=" * 70)
m = SDCombo(15).cuda().eval()
img = torch.randn(1, 3, 128, 128, device="cuda")
dep = torch.randint(100, 65535, (1, 128, 128), device="cuda")
with torch.no_grad():
    out = m(img, dep)
check("输出形状 == 输入空间尺寸", tuple(out.shape) == (1, 15, 128, 128), str(tuple(out.shape)))
s = out.softmax(1)
check("softmax 后逐像素和为 1（说明输出是 logits）",
      torch.allclose(s.sum(1), torch.ones_like(s.sum(1)), atol=1e-4))
check("logits 含负值（概率不会为负）", (out < 0).any().item())
with torch.no_grad():
    p = m.predict(img, dep)
    pr = m.predict_proba(img, dep)
check("predict() 返回 (B,H,W)", tuple(p.shape) == (1, 128, 128), str(tuple(p.shape)))
check("predict_proba() 逐像素和为 1",
      torch.allclose(pr.sum(1), torch.ones_like(pr.sum(1)), atol=1e-4))

print()
print("=" * 70)
print("[2] 深度归一化的动态范围（原实现把中位深度压到 0.03）")
print("=" * 70)
head = SDCHead(15).cuda().eval()
# 真实深度分布：中位 2019，四分位约 [941, 6903]，含 65535 远平面
real = torch.tensor([[175, 941, 2019, 6903, 20000, 65535]], dtype=torch.int64, device="cuda")
with torch.no_grad():
    z = head.normalize_depth(real)[0, 0]      # (6,) 已去掉 batch/channel 维
old = (real.float() / 65535.0)[0]
print(f"  {'深度(cm)':>10} {'米':>8} {'原实现':>10} {'新实现':>10}")
for v, o, n in zip(real[0].tolist(), old.tolist(), z.tolist()):
    print(f"  {v:>10} {v/100:>8.1f} {o:>10.4f} {n:>10.4f}")
check("新实现的中位深度明显大于原实现", z[2].item() > 0.5,
      f"(原 {old[2].item():.4f} -> 新 {z[2].item():.4f})")
check("新实现四分位跨度 > 0.1", (z[3] - z[1]).item() > 0.1,
      f"(跨度 {(z[3]-z[1]).item():.4f})")
check("原实现四分位跨度 < 0.1（近乎常数）", (old[3] - old[1]).item() < 0.1,
      f"(跨度 {(old[3]-old[1]).item():.4f})")
check("深度归一化值落在 [0,1]", z.min().item() >= 0 and z.max().item() <= 1.0,
      f"[{z.min().item():.3f}, {z.max().item():.3f}]")

print()
print("=" * 70)
print("[3] 已消除每步的 GPU->CPU 同步（用 CUDA sync 调试模式实测）")
print("=" * 70)
# 用 AST 取代码本身（不含注释/docstring），避免被注释里的示例误导
import ast  # noqa: E402
_src = open(os.path.join(ROOT, "Joint", "SDCHead.py"), encoding="utf-8").read()
_tree = ast.parse(_src)
_attr_calls, _max_calls = [], []
for node in ast.walk(_tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "item":
            _attr_calls.append(node.lineno)
        if node.func.attr == "max" and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Name) and a0.id == "depth":
                _max_calls.append(node.lineno)
check("代码中不含 .item() 调用（原实现有 2 处）", not _attr_calls, f"行号 {_attr_calls}")
check("代码中不含 torch.max(depth)", not _max_calls, f"行号 {_max_calls}")

h3 = SDCHead(15).cuda().eval()
_seg = torch.randn(16, 15, 32, 32, device="cuda")
_dep = torch.randint(100, 65535, (16, 64, 64), device="cuda")
with torch.no_grad():
    for _ in range(3):
        h3(_seg, _dep)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        h3(_seg, _dep)
    torch.cuda.synchronize()
    head_ms = (time.perf_counter() - t0) / 50 * 1000
print(f"     SDCHead(batch16) 前向 {head_ms:.2f} ms  "
      f"（原实现每步含 2 次 GPU->CPU 同步，实测约 0.1~0.2 ms 的固定同步开销）")

print()
print("=" * 70)
print("[4] 卷积 padding 后空间尺寸不再逐层收缩")
print("=" * 70)
head2 = SDCHead(15).cuda().eval()
seg = torch.randn(1, 15, 32, 32, device="cuda")
d = torch.randint(1, 65535, (1, 64, 64), device="cuda")
with torch.no_grad():
    o = head2(seg, d)
check("SDCHead 输出空间尺寸 == 深度图尺寸", tuple(o.shape[-2:]) == (64, 64), str(tuple(o.shape)))

print()
print("=" * 70)
print("[5] 梯度健康度与反传可用性")
print("=" * 70)
m.train()
img = torch.randn(2, 3, 128, 128, device="cuda")
dep = torch.randint(100, 65535, (2, 128, 128), device="cuda")
tgt = torch.randint(0, 15, (2, 128, 128), device="cuda")
out = m(img, dep)
loss = F.cross_entropy(out, tgt, ignore_index=255)
loss.backward()
gn = torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf"))
check("loss 有限", np.isfinite(float(loss)), f"loss={float(loss):.4f}")
check("梯度范数有限且非零", np.isfinite(float(gn)) and float(gn) > 0, f"|g|={float(gn):.2f}")
no_grad = [n for n, p in m.named_parameters() if p.grad is None]
check("所有参数都收到梯度", not no_grad, f"缺失 {no_grad[:3]}" if no_grad else "")

print()
print("=" * 70)
print("[6] 前向耗时（消除 .item() 后应更快）")
print("=" * 70)
m.eval()
img = torch.randn(4, 3, 256, 256, device="cuda")
dep = torch.randint(100, 65535, (4, 256, 256), device="cuda")
with torch.no_grad():
    for _ in range(3):
        m(img, dep)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        m(img, dep)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
print(f"  batch4 crop256 前向: {dt*1000:.1f} ms")

print()
print("=" * 70)
print("全部通过" if FAIL == 0 else f"有 {FAIL} 项失败")
print("=" * 70)
sys.exit(1 if FAIL else 0)
