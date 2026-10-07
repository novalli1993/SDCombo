"""诊断预测坍缩：统计模型在验证集上的预测类别分布 vs 真实分布。"""
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import Dataset.transforms as T  # noqa: E402
from Dataset.dataset_VKITTI import VKITTI  # noqa: E402
from Joint.model import SDCombo  # noqa: E402

NAMES = ["Terrain", "Tree", "Vegetation", "Building", "Road", "GuardRail",
         "TrafficSign", "TrafficLight", "Pole", "Misc", "Truck", "Car",
         "Van", "Undefined", "未标注"]

crop = 256
mean = (33.6045, 33.9644, 27.2941)
std = (19.3824, 19.3147, 20.1879)
tf = T.Compose([T.RandomCrop(crop), T.ToTensor(), T.Normalize(mean=mean, std=std)])

ds = VKITTI(os.path.join(ROOT, "datasets", "VKITTI_II"), "validation", transforms=tf)
ckpt = sys.argv[1] if len(sys.argv) > 1 else None
model = SDCombo(15).cuda()
if ckpt:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"checkpoint: {os.path.basename(ckpt)}  missing={len(miss)} unexpected={len(unexp)}")
model.eval()

N = 40
idx = np.linspace(0, len(ds) - 1, N).astype(int)
gt_hist = np.zeros(15, dtype=np.int64)
pr_hist = np.zeros(15, dtype=np.int64)
conf = np.zeros((15, 15), dtype=np.int64)

with torch.no_grad():
    for i in idx:
        image, ann, dep = ds[int(i)]
        image = image.unsqueeze(0).cuda()
        dep = dep.unsqueeze(0).cuda()
        out = model(image, dep)
        pred = out.argmax(1)[0].cpu().numpy().astype(np.int64)
        g = ann.numpy().astype(np.int64)
        gt_hist += np.bincount(g.ravel(), minlength=15)[:15]
        pr_hist += np.bincount(pred.ravel(), minlength=15)[:15]
        k = (g >= 0) & (g < 15)
        conf += np.bincount(15 * g[k] + pred[k], minlength=225).reshape(15, 15)

gt_tot, pr_tot = gt_hist.sum(), pr_hist.sum()
print(f"\n抽样 {N} 个 256x256 裁块, 共 {gt_tot:,} 像素\n")
print(f"{'类':>3} {'名称':<14} {'GT占比':>9} {'预测占比':>10} {'该类的预测主要落到哪':<22}")
print("-" * 78)
for i in range(15):
    if gt_hist[i] == 0 and pr_hist[i] == 0:
        continue
    col = conf[:, i]
    top = col.argsort()[::-1][:3]
    tops = ", ".join(f"{NAMES[j]}:{col[j]/max(col.sum(),1):.0%}" for j in top if col[j] > 0)
    print(f"{i:>3} {NAMES[i]:<14} {gt_hist[i]/gt_tot:>8.2%} {pr_hist[i]/pr_tot:>9.2%}   {tops:<22}")

print(f"\n预测到的类别集合: {[NAMES[i] for i in range(15) if pr_hist[i] > 0]}")
print(f"从未被预测的类别: {[NAMES[i] for i in range(15) if pr_hist[i] == 0]}")

# 模型输出 logits 的熵，判断是否饱和
out_logits = []
with torch.no_grad():
    for i in idx[:10]:
        image, ann, dep = ds[int(i)]
        o = model(image.unsqueeze(0).cuda(), dep.unsqueeze(0).cuda())
        out_logits.append(o.float().cpu())
o = torch.cat(out_logits, 0)
p = o.softmax(1)
print(f"\nlogits 统计: min={o.min():.2f} max={o.max():.2f} mean={o.mean():.3f}")
print(f"softmax 最大概率均值={p.max(1)[0].mean():.4f}"
      f"  （越接近 1 说明越饱和/越自信）")
ent = -(p * p.clamp_min(1e-12).log()).sum(1)
print(f"预测熵均值={ent.mean():.4f} nats  (ln15={np.log(15):.3f} 为均匀分布上限)")
