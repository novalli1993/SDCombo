"""训练吞吐/显存基准：扫描 batch x crop，并对比 channels_last 与梯度检查点。

用合成数据固定"每配置处理 N 个 batch"，报告 img/s 与峰值显存，用于在 24GB 卡上
为 SDCombo 选一组既充分利用显存、又不 OOM 的训练配置。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Joint.model import SDCombo  # noqa: E402

TOTAL_MEM = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


def make_batch(bs, crop, device="cuda"):
    img = torch.randn(bs, 3, crop, crop, device=device)
    dep = torch.randint(100, 3000, (bs, crop, crop), device=device, dtype=torch.int64)
    tgt = torch.randint(0, 15, (bs, crop, crop), device=device, dtype=torch.int64)
    return img, dep, tgt


def bench(bs, crop, steps, amp=True, channels_last=False, with_cp=False, warmup=3):
    model = SDCombo(15, with_cp=with_cp).cuda()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    model.train()

    img, dep, tgt = make_batch(bs, crop)
    if channels_last:
        img = img.to(memory_format=torch.channels_last)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    done = 0
    try:
        for step in range(warmup + steps):
            if step == warmup:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(img, dep)
                loss = F.cross_entropy(out, tgt, ignore_index=255)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            done += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        del model, opt
        torch.cuda.empty_cache()
        return None

    peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
    peak_resv = torch.cuda.max_memory_reserved() / 1024 ** 3
    ips = done * bs / dt
    del model, opt, img, dep, tgt
    torch.cuda.empty_cache()
    return dict(bs=bs, crop=crop, amp=amp, chlast=channels_last, cp=with_cp,
                ips=ips, ms_it=dt / done * 1000, peak_alloc=peak_alloc,
                peak_resv=peak_resv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  总显存 {TOTAL_MEM:.1f} GiB")
    print(f"每配置 {args.steps} 个 batch（另有 3 个 warmup）\n")
    hdr = (f"{'batch':>6} {'crop':>6} {'ch_last':>8} {'cp':>5} "
           f"{'img/s':>9} {'ms/iter':>9} {'峰值alloc':>10} {'峰值resv':>10} {'显存占比':>9}")
    print(hdr)
    print("-" * len(hdr))

    results = []

    def run(bs, crop, **kw):
        r = bench(bs, crop, args.steps, **kw)
        tag = f"{bs:>6} {crop:>6} {str(kw.get('channels_last', False)):>8} " \
              f"{str(kw.get('with_cp', False)):>5}"
        if r is None:
            print(f"{tag} {'OOM':>9}")
            return None
        print(f"{tag} {r['ips']:>9.1f} {r['ms_it']:>9.1f} "
              f"{r['peak_alloc']:>8.2f}Gi {r['peak_resv']:>8.2f}Gi "
              f"{r['peak_resv']/TOTAL_MEM*100:>8.1f}%")
        results.append(r)
        return r

    # 基准：当前配置
    run(4, 256)
    # batch 扫描（crop 256）
    for bs in ([8, 16, 32] if args.quick else [8, 16, 24, 32, 48]):
        run(bs, 256)
    # crop 扫描（batch 8）
    for crop in [384, 512]:
        run(8, crop)
        run(16, crop)
    # channels_last 对比
    run(16, 256, channels_last=True)
    run(8, 512, channels_last=True)
    # 梯度检查点（大 crop 时换显存）
    run(8, 512, with_cp=True)
    run(16, 512, with_cp=True)

    if results:
        best = max(results, key=lambda r: r["ips"])
        print("\n" + "=" * 70)
        print(f"吞吐最高: batch={best['bs']} crop={best['crop']} "
              f"ch_last={best['chlast']} cp={best['cp']}  -> {best['ips']:.1f} img/s")
        # 在显存占用 <80% 的配置里选最快
        safe = [r for r in results if r["peak_resv"] / TOTAL_MEM < 0.80]
        if safe:
            b2 = max(safe, key=lambda r: r["ips"])
            print(f"显存<80% 中最高: batch={b2['bs']} crop={b2['crop']} "
                  f"ch_last={b2['chlast']} cp={b2['cp']} -> {b2['ips']:.1f} img/s "
                  f"(resv {b2['peak_resv']:.2f} GiB, {b2['peak_resv']/TOTAL_MEM*100:.0f}%)")
        print("=" * 70)
        out = os.path.join(ROOT, "work_dir", "bench_train.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
