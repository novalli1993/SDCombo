"""临时探针：端到端训练步吞吐（真实数据管线 + 真实 Focal Loss）。

纯 GPU 基准（tools/_probe_hw.py）会给 batch 128 -> ~490 img/s，
但真实训练还受 DataLoader 制约。本探针把「取数」与「算一步」分开计时，
判断当前配置到底卡在 GPU 还是 CPU/磁盘。

同时报告 raw 与 hha 两种深度，以及不同 num_workers 的影响。
"""
import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import train_VKITTI as TV                                    # noqa: E402
from Dataset.dataset_VKITTI import VKITTI                     # noqa: E402
from Joint.model_DL4sDL import _SDCombo as SDCombo            # noqa: E402
from Utils.train_val_vkitti import criterion                  # noqa: E402

NUM_CLASSES = 14


def bench_loader(data_path, mode, crop, batch, workers, iters):
    ds = VKITTI(data_path, "training", depth_mode=mode,
                transforms=TV.get_transform(True, 375, crop, depth_mode=mode))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     num_workers=workers, pin_memory=True,
                                     collate_fn=VKITTI.collate_fn, drop_last=True)
    it = iter(dl)
    # 预热
    for _ in range(2):
        next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        next(it)
        n += 1
    dt = time.perf_counter() - t0
    img = n * batch
    return img / dt, dt / n


def bench_step(model, optimizer, scaler, data, steps, amp=True):
    def step():
        rgb, ann, dep = data
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(rgb, dep)
            loss = criterion(out, ann, ignore_index=255, alpha=0.5, gamma=2.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.item()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="datasets/VKITTI_II")
    ap.add_argument("--modes", default="raw,hha")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", default="4,8")
    ap.add_argument("--loader-iters", type=int, default=20)
    ap.add_argument("--step-iters", type=int, default=8)
    args = ap.parse_args()

    hdr = "{:>5} {:>8} {:>6} {:>7} {:>12} {:>13} {:>10}".format(
        "mode", "workers", "batch", "crop", "loader img/s", "GPU step img/s", "瓶颈")
    print(hdr)
    print("-" * len(hdr))

    model = SDCombo(aux=True, num_classes=NUM_CLASSES).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    model.train()

    for mode in args.modes.split(","):
        for w in [int(x) for x in args.workers.split(",")]:
            try:
                l_img_s, l_iter = bench_loader(args.data_path, mode, args.crop,
                                               args.batch, w, args.loader_iters)
            except Exception as exc:                       # noqa: BLE001
                print("{:>5} {:>8} {:>6} {:>7} {:>12} {:>13}".format(
                    mode, w, args.batch, args.crop, "ERR: {}".format(exc)[:12], "-", "-"))
                continue
            ds = VKITTI(args.data_path, "training", depth_mode=mode,
                        transforms=TV.get_transform(True, 375, args.crop, depth_mode=mode))
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                             num_workers=w, pin_memory=True,
                                             collate_fn=VKITTI.collate_fn, drop_last=True)
            data = next(iter(dl))
            data = [t.cuda(non_blocking=True) for t in data]
            try:
                s_iter = bench_step(model, opt, scaler, data, args.step_iters)
            except torch.cuda.OutOfMemoryError:
                print("{:>5} {:>8} {:>6} {:>7} {:>12} {:>13}".format(
                    mode, w, args.batch, args.crop, "{:.0f}".format(l_img_s), "OOM", "-"))
                torch.cuda.empty_cache()
                continue
            g_img_s = args.batch / s_iter
            verdict = "GPU" if g_img_s < l_img_s else "数据管线"
            print("{:>5} {:>8} {:>6} {:>7} {:>12.1f} {:>13.1f} {:>10}".format(
                mode, w, args.batch, args.crop, l_img_s, g_img_s, verdict))
            del data, dl, ds
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
