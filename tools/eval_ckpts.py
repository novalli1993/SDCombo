"""在**完全相同**的评估协议下对比多个 checkpoint 的 mIoU。

用途：消除"评估口径不同导致的假差异"。例如旧实现用 RandomCrop(256) 只覆盖
14% 像素，而修正后用 CenterCrop(375) 覆盖 30%，两者数值不可直接比较。
本脚本对每个 ckpt 用同一 crop / 同一批图（按固定间隔抽样）计算混淆矩阵。

用法:
    python tools/eval_ckpts.py --ckpt a.pth b.pth --crop 256 --samples 400
    python tools/eval_ckpts.py --ckpt a.pth --full-res
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Dataset.dataset_VKITTI import VKITTI  # noqa: E402
from Joint.model import SDCombo  # noqa: E402
from train import PipelineEval, PipelineEvalNoCrop  # noqa: E402

NAMES = ["Terrain", "Tree", "Vegetation", "Building", "Road", "GuardRail",
         "TrafficSign", "TrafficLight", "Pole", "Misc", "Truck", "Car",
         "Van", "Undefined", "Unknown"]


def build_model(ckpt):
    model = SDCombo(15).cuda()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, len(missing), len(unexpected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--full-res", action="store_true")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--out", default=None, help="把结果写成 json")
    args = ap.parse_args()

    tf = PipelineEvalNoCrop() if args.full_res else PipelineEval(args.crop)
    ds = VKITTI(os.path.join(ROOT, "datasets", "VKITTI_II"), "validation", transforms=tf)
    n = min(args.samples, len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    proto = "整图(不裁剪)" if args.full_res else f"CenterCrop({args.crop})"
    print(f"评估协议: {proto}  样本数: {n}  验证集总量: {len(ds)}")
    print()

    hdr = f"{'checkpoint':<40} {'mIoU':>6} {'acc':>7} {'类别数':>6} {'缺/多':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for ck in args.ckpt:
        model, nmiss, nunexp = build_model(ck)
        cm = torch.zeros(15, 15, dtype=torch.int64, device="cuda")
        with torch.no_grad():
            for i in idx:
                image, ann, dep = ds[int(i)]
                out = model(image.unsqueeze(0).cuda(), dep.unsqueeze(0).cuda())
                p = out.argmax(1).flatten()
                a = ann.cuda().flatten()
                k = (a >= 0) & (a < 15)
                inds = 15 * a[k].to(torch.int64) + p[k]
                cm += torch.bincount(inds, minlength=225).reshape(15, 15)
        h = cm.float()
        row = h.sum(1)
        col = h.sum(0)
        diag = torch.diag(h)
        iu = diag / (row + col - diag).clamp(min=1e-9)
        present = row > 0
        iu_v = iu.clone()
        iu_v[~present] = float("nan")
        miou = iu_v[~torch.isnan(iu_v)].mean().item() * 100
        acc = (diag.sum() / h.sum()).item() * 100
        ncls = int((col > 0).sum())
        name = os.path.basename(ck)[:38]
        print(f"{name:<40} {miou:>6.1f} {acc:>6.1f}% {ncls:>6} {nmiss}/{nunexp:>5}")
        rows.append(dict(ckpt=name, miou=miou, acc=acc, ncls=ncls,
                         per_class=[round(v * 100, 1) for v in iu.tolist()]))
        del model
        torch.cuda.empty_cache()

    print("\n逐类 IoU:")
    for r in rows:
        print(f"  {r['ckpt']:<40} " + " ".join(f"{v:>5.1f}" for v in r["per_class"]))
    print(f"  {'类别名':<40} " + " ".join(f"{n[:5]:>5}" for n in NAMES))

    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
