"""临时探针：在**完整训练状态**下找 batch 上限（含 optimizer state 的持续占用）。

tools/_probe_hw.py 只用 `max_memory_reserved()` 看单步峰值，且每步都重建优化器，
会**低估**真实训练的常驻占用（AdamW 的 exp_avg/exp_avg_sq 是随训练累积的）。
本探针连续跑若干步不做清理，更接近真实训练，用来确定不会 OOM 的 batch 上限。
"""
import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Joint.model_DL4sDL import _SDCombo as SDCombo     # noqa: E402
from Utils.train_val_vkitti import criterion            # noqa: E402

NUM_CLASSES = 14


def try_batch(batch, crop, depth_mode, steps, amp=True):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = SDCombo(aux=True, num_classes=NUM_CLASSES).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda") if amp else None
        rgb = torch.randn(batch, 3, crop, crop, device="cuda")
        depth = (torch.rand(batch, 3, crop, crop, device="cuda") if depth_mode == "hha"
                 else torch.randint(0, 65535, (batch, crop, crop), device="cuda").float())
        target = torch.randint(0, NUM_CLASSES, (batch, crop, crop), device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(rgb, depth)
                loss = criterion(out, target, ignore_index=255, alpha=0.5, gamma=2.0)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps
        alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserv = torch.cuda.max_memory_reserved() / 1024 ** 3
        return True, dt, alloc, reserv, None
    except torch.cuda.OutOfMemoryError as exc:
        return False, 0.0, 0.0, 0.0, str(exc).split("\n")[0][:110]
    finally:
        for v in ("model", "opt", "rgb", "depth", "target", "out", "loss"):
            if v in dir():
                pass
        try:
            del model, opt, rgb, depth, target, out, loss
        except Exception:                                   # noqa: BLE001
            pass
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batches", default="64,96,112,128,144,160")
    ap.add_argument("--depth-mode", default="raw")
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("显存总量 {:.2f} GiB   crop={}  depth={}  steps={}".format(
        total, args.crop, args.depth_mode, args.steps))
    print("{:>6} {:>8} {:>10} {:>12} {:>12} {:>10}".format(
        "batch", "结果", "s/iter", "peak alloc", "reserved", "占用%"))
    print("-" * 62)
    for b in [int(x) for x in args.batches.split(",")]:
        ok, dt, alloc, reserv, err = try_batch(b, args.crop, args.depth_mode, args.steps)
        if ok:
            print("{:>6} {:>8} {:>10.3f} {:>10.2f}G {:>10.2f}G {:>9.1f}%".format(
                b, "OK", dt, alloc, reserv, 100 * reserv / total))
        else:
            print("{:>6} {:>8}  {}".format(b, "OOM", err))


if __name__ == "__main__":
    main()
