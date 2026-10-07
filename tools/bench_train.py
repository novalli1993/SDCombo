"""吞吐 / 显存基准 —— 论文主线模型 SDCombo 在 VKITTI 分辨率下的 batch x crop 扫描。

用合成张量直接驱动训练一步（不读磁盘），测的是纯 GPU 吞吐，用于选 batch size
并判断瓶颈在哪里。

用法（仓库根目录，先 conda activate SDCombo）:
    python tools\\bench_train.py                     # 默认扫描一组常用配置
    python tools\\bench_train.py --steps 10          # 每配置多跑几步，结果更稳
    python tools\\bench_train.py --batch 16 --crop 256
    python tools\\bench_train.py --channels-last     # 对比 channels_last（本模型上通常更慢）
"""
import argparse
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Joint.model_DL4sDL import _SDCombo as SDCombo  # noqa: E402

NUM_CLASSES = 14


def build(batch, crop, channels_last=False):
    rgb = torch.randn(batch, 3, crop, crop, device="cuda")
    # 原始深度：真实 VKITTI 是 16bit（最大 65535），这里保持同一量纲
    depth = torch.randint(0, 65535, (batch, crop, crop), device="cuda").float()
    target = torch.randint(0, NUM_CLASSES, (batch, crop, crop), device="cuda")
    if channels_last:
        rgb = rgb.to(memory_format=torch.channels_last)
    return rgb, depth, target


def run_one(model, optimizer, scaler, batch, crop, steps, amp, channels_last=False):
    rgb, depth, target = build(batch, crop, channels_last)
    model.train()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    for phase in ("warmup", "timed"):
        n = 1 if phase == "warmup" else steps     # 预热一次（cuDNN autotune），不计入统计
        if phase == "timed":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        for _ in range(n):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(rgb, depth)
                loss = criterion(out["out"], target) + 0.5 * criterion(out["aux"], target)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if phase == "timed":
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
    img_s = batch * steps / dt
    alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
    reserv = torch.cuda.max_memory_reserved() / 1024 ** 3
    del rgb, depth, target, out, loss
    torch.cuda.empty_cache()
    return img_s, dt / steps, alloc, reserv


def main():
    ap = argparse.ArgumentParser(description="SDCombo batch x crop 基准")
    ap.add_argument("--steps", default=6, type=int, help="每个配置的计时步数")
    ap.add_argument("--batch", type=int, default=0, help="只测单个 batch（0=扫描）")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", action="store_false", dest="amp")
    ap.add_argument("--channels-last", action="store_true", help="对比 channels_last 的影响")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[FAIL] 没有可用的 CUDA 设备")
        return 1
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print("GPU: {}  显存: {:.1f} GiB  torch {}".format(
        torch.cuda.get_device_name(0), total_gib, torch.__version__))

    model = SDCombo(aux=True, num_classes=NUM_CLASSES).to(args.device)
    n_param = sum(p.numel() for p in model.parameters())
    print("SDCombo 参数量: {:.2f} M   AMP={}  channels_last={}\n".format(
        n_param / 1e6, args.amp, args.channels_last))

    configs = [(args.batch, args.crop)] if args.batch else [
        (8, 256), (16, 256), (24, 256), (32, 256), (48, 256), (64, 256),
        (16, 384), (8, 384), (8, 512), (4, 512),
    ]
    header = "{:>6} {:>6} {:>10} {:>12} {:>14} {:>8}".format(
        "batch", "crop", "s/iter", "img/s", "reserved/GB", "占用%")
    print(header)
    print("-" * len(header))
    results = []
    for batch, crop in configs:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda") if args.amp else None
        try:
            img_s, s_iter, alloc, reserv = run_one(model, optimizer, scaler, batch, crop,
                                                   args.steps, args.amp, args.channels_last)
        except torch.cuda.OutOfMemoryError:
            print("{:>6} {:>6} {:>10} {:>12} {:>14} {:>8}".format(batch, crop, "-", "OOM", "-", "-"))
            torch.cuda.empty_cache()
            continue
        pct = 100.0 * reserv / total_gib
        print("{:>6} {:>6} {:>10.3f} {:>12.1f} {:>14.2f} {:>8.1f}".format(
            batch, crop, s_iter, img_s, reserv, pct))
        results.append((batch, crop, img_s, reserv, pct))

    if results:
        best = max(results, key=lambda r: r[2])
        print("\n最快: batch={} crop={} -> {:.1f} img/s（reserved {:.2f} GB, {:.1f}%）".format(
            best[0], best[1], best[2], best[3], best[4]))
        safe = [r for r in results if r[4] < 90]
        if safe:
            b = max(safe, key=lambda r: r[2])
            print("推荐: --batch-size {} --crop_size {} -> {:.1f} img/s，显存留有余量".format(
                b[0], b[1], b[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
