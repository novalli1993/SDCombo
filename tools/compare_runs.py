"""汇总对比多次训练运行的逐 epoch 指标（从 work_dir 的归档目录读取）。

用法:
    python tools/compare_runs.py
    python tools/compare_runs.py --runs _baseline10ep_lr2.4e-4_* --runs _lr6e-5*

每个运行的指标来自 work_dir/<run>/evaluation*.txt，由 train.py 逐 epoch 写入。
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_eval(path):
    """从 evaluation*.txt 解析逐 epoch 的 train_loss / acc / mIoU。"""
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for block in txt.split("[epoch:")[1:]:
        ep = int(re.match(r"\s*(\d+)", block).group(1))
        loss = re.search(r"train_loss:\s*([\d.eE+-]+)", block)
        acc = re.search(r"global correct:\s*([\d.]+)", block)
        mi = re.search(r"mean IoU:\s*([\d.]+|nan)", block)
        iou = re.search(r"IoU:\s*\[([^\]]+)\]", block)
        out.append(dict(
            epoch=ep,
            loss=float(loss.group(1)) if loss else float("nan"),
            acc=float(acc.group(1)) if acc else float("nan"),
            miou=float(mi.group(1)) if mi and mi.group(1) != "nan" else float("nan"),
            iou=[float(x.strip().strip("'")) for x in iou.group(1).split(",")] if iou else [],
        ))
    return out


def n_nonzero(iou):
    return sum(1 for v in iou if v and v > 0)


def summarize(name, rows):
    if not rows:
        return None
    mious = [r["miou"] for r in rows if r["miou"] == r["miou"]]
    accs = [r["acc"] for r in rows if r["acc"] == r["acc"]]
    losses = [r["loss"] for r in rows if r["loss"] == r["loss"]]
    best = max(mious) if mious else float("nan")
    last = mious[-1] if mious else float("nan")
    # 稳定性：后一半 epoch 的 mIoU 标准差（越小越稳）
    half = mious[len(mious) // 2:] if mious else []
    import statistics
    sd = statistics.pstdev(half) if len(half) > 1 else 0.0
    has_nan = any(r["loss"] != r["loss"] or r["miou"] != r["miou"] for r in rows)
    return dict(name=name, n=len(rows), best=best, last=last, sd=sd,
                acc=accs[-1] if accs else float("nan"),
                loss=losses[-1] if losses else float("nan"),
                cls=n_nonzero(rows[-1]["iou"]), has_nan=has_nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", default=None,
                    help="work_dir 下的运行目录名（支持通配），可多次指定")
    args = ap.parse_args()

    wd = os.path.join(ROOT, "work_dir")
    if args.runs:
        names = []
        for pat in args.runs:
            names += [os.path.basename(p) for p in glob.glob(os.path.join(wd, pat))]
    else:
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(wd, "*"))
                       if os.path.isdir(p))

    results, details = [], {}
    for n in names:
        d = os.path.join(wd, n)
        evals = glob.glob(os.path.join(d, "evaluation*.txt"))
        if not evals:
            continue
        # 一个运行目录里可能有多个 evaluation*.txt（例如中途重启产生的新 mark）。
        # 解析全部候选，取 epoch 数最多的那个，避免读到不完整的记录。
        best_rows, best_src = [], None
        for e in evals:
            rows = parse_eval(e)
            if len(rows) > len(best_rows):
                best_rows, best_src = rows, e
        s = summarize(n, best_rows)
        if s:
            s["src"] = os.path.basename(best_src)
            results.append(s)
            details[n] = best_rows

    if not results:
        print("未找到任何运行记录（work_dir 下没有 evaluation*.txt）")
        return 1

    hdr = (f"{'运行':<44} {'ep':>3} {'末loss':>8} {'末acc':>7} {'末mIoU':>7} "
           f"{'最优mIoU':>8} {'后半std':>8} {'类别数':>6} {'发散':>5}")
    print(hdr)
    print("-" * len(hdr))
    for s in results:
        print(f"{s['name']:<44} {s['n']:>3} {s['loss']:>8.4f} {s['acc']:>6.1f}% "
              f"{s['last']:>7.1f} {s['best']:>8.1f} {s['sd']:>8.2f} "
              f"{s['cls']:>6} {'是' if s['has_nan'] else '否':>5}")

    print("\n逐 epoch mIoU 曲线:")
    for s in results:
        rows = details[s["name"]]
        curve = " ".join(f"{r['miou']:.1f}" if r["miou"] == r["miou"] else "nan"
                         for r in rows)
        print(f"  {s['name']:<44} {curve}")

    print("\n逐 epoch train_loss 曲线:")
    for s in results:
        rows = details[s["name"]]
        curve = " ".join(f"{r['loss']:.3f}" if r["loss"] == r["loss"] else "nan"
                         for r in rows)
        print(f"  {s['name']:<44} {curve}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
