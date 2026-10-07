"""论文 §6.3 的对照实验：**HHA vs 原始深度**（VKITTI 2 版）。

复现论文 Table 6 / §6.5 的核心结论：
    SDCombo with HHA  mIoU 36.5 / mAcc 65.4   （论文，Stanford2D3D）
    SDCombo with Depth mIoU 17.6 / mAcc 45.5
    -> "depth information encoding is an important factor"

本脚本把两组训练 + 整图评估串起来，并强制**唯一变量原则**：
除 `--depth-mode` 外，模型、优化器、lr、调度、epoch、batch、crop、增广、损失逐项相同。

用法：
    python tools/compare_depth_encoding.py                 # 全量 12 epoch（默认）
    python tools/compare_depth_encoding.py --limit 8000 --epochs 4    # 快速版
    python tools/compare_depth_encoding.py --only hha      # 只跑一组
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def eval_marks():
    """已存在的**训练** mark 集合。

    注意要排除 `evaluationFull_*.txt` —— 那是 evaluation_VKITTI.py 写的独立评估记录，
    它的文件名同样是 `evaluation<mark>.txt` 形式，若不排除会被当成训练记录去解析，
    导致解析不到 `[epoch: N]` 而崩溃（本脚本曾经踩过这个坑）。
    """
    marks = set()
    for p in glob.glob(os.path.join(ROOT, "work_dir", "evaluation", "evaluation*.txt")):
        stem = os.path.basename(p)[len("evaluation"):-len(".txt")]
        if stem.startswith("Full_") or not stem:
            continue
        marks.add(stem)
    return marks


def parse(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    rows = []
    for m in re.finditer(r"\[epoch: (\d+)\](.*?)(?=\[epoch: |\Z)", t, re.S):
        body = m.group(2)
        g = lambda pat: (re.search(pat, body).group(1) if re.search(pat, body) else None)  # noqa: E731
        rows.append(dict(
            epoch=int(m.group(1)), loss=g(r"train_loss: ([\d.]+)"),
            acc=g(r"global correct: ([\d.]+)"),
            miou=g(r"mean IoU: ([\d.]+)"),
            miou_present=g(r"mean IoU \(present \d+ classes only\): ([\d.]+)"),
            per_class=([float(x) for x in re.findall(r"[\d.]+",
                        re.search(r"IoU: \[([^\]]*)\]", body).group(1))]
                       if re.search(r"IoU: \[([^\]]*)\]", body) else [])))
    return rows


def run_train(tag, depth_mode, args):
    before = eval_marks()
    cmd = [sys.executable, "train_VKITTI.py",
           "--data-path", args.data_path,
           "--depth-mode", depth_mode,
           "--batch-size", str(args.batch_size),
           "--crop_size", str(args.crop_size),
           "--num-workers", str(args.num_workers),
           "--epochs", str(args.epochs),
           "--lr", str(args.lr), "--lr-min", str(args.lr_min),
           "--warmup-epochs", str(args.warmup_epochs),
           "--max-grad-norm", str(args.max_grad_norm),
           "--class-weight", args.class_weight,
           "--eval-batch-size", str(args.eval_batch_size),
           "--print-freq", "400"]
    if args.limit:
        cmd += ["--limit", str(args.limit)]

    print("\n" + "=" * 78)
    print("[{}] depth_mode={}   {}".format(tag, depth_mode, " ".join(cmd[2:])))
    print("=" * 78, flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", errors="replace")
    wall = time.time() - t0
    if proc.returncode != 0:
        print("[{}] 训练失败 rc={}".format(tag, proc.returncode))
        return None

    new = sorted(eval_marks() - before)
    if not new:
        print("[{}] 未找到新的 mark".format(tag))
        return None
    mark = new[-1]
    rows = parse(os.path.join(ROOT, "work_dir", "evaluation", "evaluation{}.txt".format(mark)))
    return dict(tag=tag, mode=depth_mode, mark=mark, wall=wall, rows=rows)


def run_eval(res, args):
    """整图评估最后一个 epoch 的权重（论文 §6.4.2 口径）。"""
    last_epoch = max(r["epoch"] for r in res["rows"])
    weight = os.path.join(ROOT, "work_dir", "model",
                          "model_{}_{}.pth".format(res["mark"], last_epoch))
    if not os.path.isfile(weight):
        print("[{}] 找不到权重 {}".format(res["tag"], weight))
        return None
    cmd = [sys.executable, "evaluation_VKITTI.py", "--weight", weight,
           "--data-path", args.data_path, "--crop_size", "0",
           "-b", "4", "--num-workers", str(args.num_workers)]
    print("\n[{}] 整图评估: {}".format(res["tag"], os.path.basename(weight)), flush=True)
    subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8", errors="replace")
    # 评估结果写进新的 evaluationFull_*.txt，取最新一个
    cands = sorted(glob.glob(os.path.join(ROOT, "work_dir", "evaluation", "evaluationFull_*.txt")),
                   key=os.path.getmtime)
    if not cands:
        return None
    t = open(cands[-1], encoding="utf-8", errors="replace").read()
    full = re.search(r"mean IoU: ([\d.]+)", t)
    fp = re.search(r"mean IoU \(present \d+ classes only\): ([\d.]+)", t)
    acc = re.search(r"global correct: ([\d.]+)", t)
    iou = re.search(r"IoU: \[([^\]]*)\]", t)
    return dict(miou=float(full.group(1)) if full else None,
                miou_present=float(fp.group(1)) if fp else None,
                acc=float(acc.group(1)) if acc else None,
                per_class=[float(x) for x in re.findall(r"[\d.]+", iou.group(1))] if iou else [])


def main():
    ap = argparse.ArgumentParser(description="HHA vs 原始深度 对照（论文 §6.3）")
    ap.add_argument("--data-path", default="datasets/VKITTI_II")
    ap.add_argument("--only", default="", choices=["", "hha", "raw"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--class-weight", default="none")
    args = ap.parse_args()

    modes = ["hha", "raw"] if not args.only else [args.only]
    for m in modes:
        if m == "hha":
            d = os.path.join(ROOT, args.data_path, "HHA", "training")
            if not os.path.isdir(d):
                print("[FAIL] 找不到 {}，请先运行 tools/prepare_hha_vkitti.py".format(d))
                return 1

    print("唯一变量原则：两组只有 --depth-mode 不同")
    print("共同设置: batch={} crop={} epochs={} lr={} lr_min={} warmup={} "
          "grad_norm={} class_weight={} workers={}".format(
              args.batch_size, args.crop_size, args.epochs, args.lr, args.lr_min,
              args.warmup_epochs, args.max_grad_norm, args.class_weight, args.num_workers))
    if args.limit:
        print("limit={}（快速版，数值不可与论文比较）".format(args.limit))

    results = []
    for m in modes:
        r = run_train(m, m, args)
        if r:
            r["full"] = run_eval(r, args)
            results.append(r)

    print("\n\n" + "#" * 78)
    print("# 论文 §6.3 对照结果（VKITTI 2）")
    print("#" * 78)
    if not results:
        print("没有完成的实验")
        return 1

    print("\n-- 训练中的逐 epoch 快速评估（末值） --")
    print("{:>6} {:>10} {:>10} {:>14} {:>12}".format(
        "mode", "末 mIoU", "末 mAcc", "末 mIoU(出现类)", "训练墙钟"))
    for r in results:
        last = r["rows"][-1]
        print("{:>6} {:>10} {:>10} {:>14} {:>12}".format(
            r["mode"], last["miou"], last["acc"], last["miou_present"] or "-",
            "{:.0f}s".format(r["wall"])))

    print("\n-- 整图评估（论文 §6.4.2 口径，最终数字）--")
    have_full = [r for r in results if r.get("full")]
    if have_full:
        print("{:>6} {:>12} {:>12} {:>18} {:>10}".format(
            "mode", "mIoU(14类)", "mAcc", "mIoU(出现类)", "权重"))
        for r in have_full:
            f = r["full"]
            print("{:>6} {:>12} {:>12} {:>18} {:>10}".format(
                r["mode"], f["miou"], f["acc"], f["miou_present"], r["mark"]))
        if len(have_full) == 2:
            a, b = have_full[0], have_full[1]
            d = (a["full"]["miou"] or 0) - (b["full"]["miou"] or 0)
            print("\n>>> {} - {} = {:+.2f} mIoU 点".format(
                a["mode"], b["mode"], d))
            print(">>> 论文对应结论（Stanford2D3D）: HHA - Depth = 36.5 - 17.6 = +18.9 mIoU 点")
    else:
        print("（无整图评估结果）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
