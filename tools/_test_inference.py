"""检查推理路径：整图与中心裁剪两种口径的输出形状、显存、覆盖比例、确定性。"""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Dataset.dataset_VKITTI import VKITTI  # noqa: E402
from Joint.model import SDCombo  # noqa: E402
from train import PipelineEval, PipelineEvalNoCrop  # noqa: E402

FAIL = 0


def check(label, cond, extra=""):
    global FAIL
    print(f"  {'OK  ' if cond else 'FAIL'} {label}" + (f"  {extra}" if extra else ""))
    if not cond:
        FAIL += 1


ROOT_DS = os.path.join(ROOT, "datasets", "VKITTI_II")
model = SDCombo(15).cuda().eval()

print("=" * 72)
print("[1] 整图推理（PipelineEvalNoCrop）：形状、显存、耗时")
print("=" * 72)
ds_full = VKITTI(ROOT_DS, "validation", transforms=PipelineEvalNoCrop())
img, ann, dep = ds_full[0]
check("整图张量形状为 3x375x1242", tuple(img.shape) == (3, 375, 1242), str(tuple(img.shape)))
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    t0 = time.perf_counter()
    out = model(img.unsqueeze(0).cuda(), dep.unsqueeze(0).cuda())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
peak = torch.cuda.max_memory_allocated() / 1024 ** 3
check("整图输出形状 == 输入分辨率", tuple(out.shape[-2:]) == (375, 1242), str(tuple(out.shape)))
print(f"     整图前向 {dt*1000:.0f} ms, 峰值显存 {peak:.2f} GiB")

print()
print("=" * 72)
print("[2] 原评估口径（RandomCrop 256）实际只看多少像素")
print("=" * 72)
H, W = 375, 1242
for name, h, w in (("RandomCrop(256) / CenterCrop(256)", 256, 256),
                   ("CenterCrop(375)", 375, 375),
                   ("整图", H, W)):
    print(f"     {name:<34} 覆盖 {h*w/(H*W)*100:>5.1f}%  ({h}x{w} / {H}x{W})")

print()
print("=" * 72)
print("[3] 评估确定性：同一张图重复取两次应完全一致")
print("=" * 72)
ds_c = VKITTI(ROOT_DS, "validation", transforms=PipelineEval(375))
a1 = ds_c[0]
a2 = ds_c[0]
check("CenterCrop 管线两次读取一致（可复现）",
      torch.equal(a1[1], a2[1]) and torch.equal(a1[2], a2[2]))

# 原实现用 RandomCrop，重复读取会不同 —— 这里直接验证 RandomCrop 的随机性
import Dataset.transforms as T  # noqa: E402
tf_rand = T.Compose([T.RandomCrop(256), T.ToTensor()])
_raw = Image.open(os.path.join(ROOT_DS, "images", "validation",
                               sorted(os.listdir(os.path.join(ROOT_DS, "images", "validation")))[0]))
_ann = Image.open(os.path.join(ROOT_DS, "annotations", "validation",
                               sorted(os.listdir(os.path.join(ROOT_DS, "annotations", "validation")))[0]))
_dep = Image.open(os.path.join(ROOT_DS, "depth", "validation",
                               sorted(os.listdir(os.path.join(ROOT_DS, "depth", "validation")))[0]))
r1 = tf_rand(_raw.copy(), _ann.copy(), _dep.copy())[1]
r2 = tf_rand(_raw.copy(), _ann.copy(), _dep.copy())[1]
check("原 RandomCrop 口径两次结果不同（说明原评估不可复现）", not torch.equal(r1, r2))

print()
print("=" * 72)
print("[4] 旧 checkpoint 与修正后模型结构不兼容（需重新训练）")
print("=" * 72)
old = os.path.join(ROOT, "work_dir", "_lr6e-5_ignore0_20261007_114908")
ckpts = []
if os.path.isdir(old):
    ckpts = [os.path.join(old, f) for f in os.listdir(old) if f.endswith(".pth")]
if ckpts:
    sd = torch.load(ckpts[0], map_location="cpu", weights_only=False)["model"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    norm_keys = [k for k in sd if "GroupNorm" in k or "running_mean" in k]
    check("旧 checkpoint 含 BatchNorm 统计量（新模型为 GroupNorm）",
          any("running_mean" in k for k in sd))
    check("加载旧 checkpoint 会大量不匹配（符合预期，需重训）",
          len(miss) > 0 or len(unexp) > 0,
          f"missing={len(miss)} unexpected={len(unexp)}")
else:
    print("     未找到旧 checkpoint，跳过")

print()
print("=" * 72)
print("全部通过" if FAIL == 0 else f"有 {FAIL} 项失败")
print("=" * 72)
sys.exit(1 if FAIL else 0)
