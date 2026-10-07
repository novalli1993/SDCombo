"""临时探针：在本机（RTX 5090 D v2 / sm_120）上寻找最优训练设置。

与 tools/bench_train.py 的区别：
  * 用**真实**的 Focal Loss 与真实的深度分支输入（单通道 raw 或 3 通道 HHA）；
  * 除纯 GPU 吞吐外，还端到端测「真实数据管线 + 训练一步」，暴露数据加载瓶颈；
  * 对比 torch.compile / channels_last / TF32（matmul precision）三个开关；
  * 跑到显存上限，找出 batch 的拐点，而不是只测到 64。
"""
import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Joint.model_DL4sDL import _SDCombo as SDCombo          # noqa: E402
from Utils.train_val_vkitti import criterion                 # noqa: E402

NUM_CLASSES = 14


def make_depth(batch, crop, depth_mode):
    if depth_mode == "hha":
        return torch.rand(batch, 3, crop, crop, device="cuda")
    return torch.randint(0, 65535, (batch, crop, crop), device="cuda").float()


def one_step(model, optimizer, scaler, rgb, depth, target, amp=True, clip=1.0):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=amp):
        out = model(rgb, depth)
        loss = criterion(out, target, ignore_index=255, alpha=0.5, gamma=2.0)
    if scaler is not None:
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return loss


def timed(model, optimizer, scaler, tensors, steps, amp, warmup=3):
    for _ in range(warmup):
        one_step(model, optimizer, scaler, *tensors, amp=amp)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step(model, optimizer, scaler, *tensors, amp=amp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main():
    ap = argparse.ArgumentParser(description="SDCombo 硬件调优扫描")
    ap.add_argument("--steps", default=8, type=int)
    ap.add_argument("--depth-mode", default="raw", choices=["raw", "hha"])
    ap.add_argument("--crops", default="256,384")
    ap.add_argument("--batches", default="16,32,64,96,128,192")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    args = ap.parse_args()

    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1024 ** 3
    print("GPU: {}  显存 {:.1f} GiB  torch {}  arch {}".format(
        torch.cuda.get_device_name(0), total, torch.__version__,
        torch.cuda.get_arch_list()[-3:]))
    print("capability sm_{}{}  AMP={}  depth_mode={}\n".format(
        *torch.cuda.get_device_capability(0), args.amp, args.depth_mode))

    crops = [int(x) for x in args.crops.split(",")]
    batches = [int(x) for x in args.batches.split(",")]

    hdr = "{:>16} {:>6} {:>6} {:>9} {:>10} {:>11} {:>7}".format(
        "配置", "batch", "crop", "s/iter", "img/s", "reserved", "占用%")
    print(hdr)
    print("-" * len(hdr))

    best = None
    for tag, use_compile, cl, precision in [
        ("baseline", False, False, "high"),
        ("tf32-high", False, False, "high"),
    ]:
        if tag == "tf32-high":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
        if tag == "baseline":
            torch.set_float32_matmul_precision("highest")
        for crop in crops:
            for batch in batches:
                model = SDCombo(aux=True, num_classes=NUM_CLASSES).to(dev)
                if use_compile:
                    try:
                        model = torch.compile(model)
                    except Exception as exc:                      # noqa: BLE001
                        print("  torch.compile 不可用: {}".format(exc))
                        del model
                        torch.cuda.empty_cache()
                        break
                if cl:
                    model = model.to(memory_format=torch.channels_last)
                opt = torch.optim.AdamW(model.parameters(), lr=5e-5,
                                        betas=(0.9, 0.999), weight_decay=0.01)
                scaler = torch.amp.GradScaler("cuda") if args.amp else None
                rgb = torch.randn(batch, 3, crop, crop, device=dev)
                if cl:
                    rgb = rgb.to(memory_format=torch.channels_last)
                depth = make_depth(batch, crop, args.depth_mode)
                target = torch.randint(0, NUM_CLASSES, (batch, crop, crop), device=dev)
                try:
                    s_iter = timed(model, opt, scaler, (rgb, depth, target),
                                   args.steps, args.amp)
                except torch.cuda.OutOfMemoryError:
                    print("{:>16} {:>6} {:>6} {:>9} {:>10} {:>11} {:>7}".format(
                        tag, batch, crop, "-", "OOM", "-", "-"))
                    del model, opt, rgb, depth, target
                    torch.cuda.empty_cache()
                    continue
                reserv = torch.cuda.max_memory_reserved() / 1024 ** 3
                img_s = batch / s_iter
                pct = 100.0 * reserv / total
                print("{:>16} {:>6} {:>6} {:>9.3f} {:>10.1f} {:>10.2f}G {:>6.1f}".format(
                    tag, batch, crop, s_iter, img_s, reserv, pct))
                if best is None or img_s > best[3]:
                    best = (tag, batch, crop, img_s, reserv, pct)
                del model, opt, rgb, depth, target
                torch.cuda.empty_cache()

    if best:
        print("\n最快可用配置: {}  batch={} crop={} -> {:.1f} img/s "
              "(reserved {:.2f} GiB, {:.1f}%)".format(*best))


if __name__ == "__main__":
    main()
