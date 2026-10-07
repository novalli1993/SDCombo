"""受控实验：在相同数据/相同随机种子下比较不同学习率的稳定性。

目的：确认训练发散（loss -> nan）是否由 lr=1e-2 + AdamW 引起。
每个配置跑固定步数，报告 loss 轨迹与是否出现 nan/inf。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import Dataset.transforms as T  # noqa: E402
from Dataset.dataset_VKITTI import VKITTI  # noqa: E402
from Joint.model import SDCombo  # noqa: E402
from Utils.train_val import criterion  # noqa: E402

MEAN = (33.6045, 33.9644, 27.2941)
STD = (19.3824, 19.3147, 20.1879)


def build_ds():
    tf = T.Compose([T.RandomCrop(256), T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD)])
    return VKITTI(os.path.join(ROOT, "datasets", "VKITTI_II"), "training", transforms=tf)


def run(lr, steps, clip=None, seed=0, tag=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = build_ds()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=ds.collate_fn)

    model = SDCombo(15).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.05)
    model.train()

    losses, nan_at, max_gn = [], None, 0.0
    t0 = time.perf_counter()
    it = iter(loader)
    for step in range(steps):
        try:
            image, ann, dep = next(it)
        except StopIteration:
            it = iter(loader)
            image, ann, dep = next(it)
        image, ann, dep = image.cuda(), ann.cuda(), dep.cuda()
        out = model(image, dep)
        loss = criterion(out, ann)
        opt.zero_grad()
        loss.backward()
        # 记录梯度范数（裁剪前）
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        max_gn = max(max_gn, float(gn))
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        v = float(loss)
        losses.append(v)
        if (not np.isfinite(v)) and nan_at is None:
            nan_at = step
            break
    dt = time.perf_counter() - t0
    finite = [x for x in losses if np.isfinite(x)]
    print(f"  lr={lr:<8g} clip={str(clip):<6} steps={len(losses):<5} "
          f"loss[0]={losses[0]:.4f} " +
          (f"loss[-1]={finite[-1]:.4f} " if finite else "loss[-1]=nan ") +
          f"max|g|={max_gn:.1f} " +
          (f"★NaN@{nan_at} " if nan_at is not None else "稳定 ") +
          f"({dt:.0f}s)")
    return nan_at is None, max_gn, losses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()

    print(f"数据: {os.path.join('datasets', 'VKITTI_II')}  batch=4  crop=256")
    print(f"每个配置固定跑 {args.steps} 步\n")
    print(f"{'配置':<10} {'':<22} {'结果'}")
    print("-" * 78)
    results = {}
    for lr, clip in [(1e-2, None), (1e-3, None), (1e-4, None),
                     (1e-2, 1.0), (1e-4, 1.0), (3e-5, None)]:
        ok, gn, ls = run(lr, args.steps, clip=clip)
        results[(lr, clip)] = (ok, gn)
        print()
