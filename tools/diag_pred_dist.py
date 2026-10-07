"""对同一 checkpoint 在验证集上统计预测分布，用于判断振荡是否来自"类别偏好翻转"。"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Dataset.dataset_VKITTI import VKITTI  # noqa: E402
from Joint.model import SDCombo  # noqa: E402
from train import PipelineEval  # noqa: E402

NAMES = ["Terrain", "Tree", "Vegetation", "Building", "Road", "GuardRail",
         "TrafficSign", "TrafficLight", "Pole", "Misc", "Truck", "Car",
         "Van", "Undefined", "Unknown"]

ckpts = sys.argv[1:]
if not ckpts:
    raise SystemExit("用法: python tools/diag_pred_dist.py ckpt1.pth [ckpt2.pth ...]")

ds = VKITTI(os.path.join(ROOT, "datasets", "VKITTI_II"), "validation",
            transforms=PipelineEval(375))
N = 60
idx = np.linspace(0, len(ds) - 1, N).astype(int)

gt = np.zeros(15, dtype=np.int64)
print(f"{'checkpoint':<28} " + " ".join(f"{n[:4]:>6}" for n in NAMES[:14]))
for ck in ckpts:
    model = SDCombo(15).cuda()
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    pr = np.zeros(15, dtype=np.int64)
    for i in idx:
        image, ann, dep = ds[int(i)]
        if i == idx[0]:
            gt += np.bincount(ann.numpy().ravel(), minlength=15)[:15]
        with torch.no_grad():
            p = model(image.unsqueeze(0).cuda(), dep.unsqueeze(0).cuda()).argmax(1)
        pr += np.bincount(p[0].cpu().numpy().ravel(), minlength=15)[:15]
    tot = pr.sum()
    name = os.path.basename(ck)[:26]
    print(f"{name:<28} " + " ".join(f"{pr[i]/tot*100:5.1f}%" for i in range(14)))
    print(f"{'  -> 预测到的类别数':<28} {(pr > 0).sum()}")
    del model
    torch.cuda.empty_cache()

# 真实分布参照
tot = gt.sum()
print(f"{'GT(参照)':<28} " + " ".join(f"{gt[i]/tot*100:5.1f}%" for i in range(14)))
