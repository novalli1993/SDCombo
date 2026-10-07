"""临时实验电池：在真实 VKITTI 上比较「类别权重 / 训练预算」对收敛的影响。

不修改仓库任何文件，只按不同参数反复调用 train_VKITTI.py，
并从 work_dir/evaluation/evaluation<mark>.txt 里抽取逐 epoch 的 mIoU。
"""
import os
import re
import subprocess
import sys
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EVAL_DIR = os.path.join(ROOT, "work_dir", "evaluation")


def existing_marks():
    return {f[10:-4] for f in os.listdir(EVAL_DIR)
            if f.startswith("evaluation") and f.endswith(".txt")} if os.path.isdir(EVAL_DIR) else set()


def parse_run(mark):
    path = os.path.join(EVAL_DIR, "evaluation{}.txt".format(mark))
    if not os.path.isfile(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    epochs = []
    for m in re.finditer(r"\[epoch: (\d+)\](.*?)(?=\[epoch: |\Z)", text, re.S):
        ep, body = int(m.group(1)), m.group(2)
        loss = re.search(r"train_loss: ([\d.]+)", body)
        miou = re.search(r"mean IoU[^:]*: ([\d.]+)", body)
        gacc = re.search(r"global correct: ([\d.]+)", body)
        iou = re.search(r"IoU: \[([^\]]*)\]", body)
        per = [float(x) for x in re.findall(r"[\d.]+", iou.group(1))] if iou else []
        epochs.append(dict(epoch=ep,
                           loss=float(loss.group(1)) if loss else None,
                           miou=float(miou.group(1)) if miou else None,
                           gacc=float(gacc.group(1)) if gacc else None,
                           per_class=per))
    last = re.search(r"training time (.*)", text)
    return dict(epochs=epochs, time=last.group(1).strip() if last else "?")


def run(tag, extra, timeout=3600):
    before = existing_marks()
    cmd = [PY, "train_VKITTI.py", "--num-workers", "4", "--print-freq", "200"] + extra
    print("\n" + "=" * 78)
    print("[{}] {}".format(tag, " ".join(extra)))
    print("=" * 78, flush=True)
    t = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    dt = time.time() - t
    if proc.returncode != 0:
        print("[{}] FAILED rc={}".format(tag, proc.returncode))
        print((proc.stderr or "")[-2000:])
        return None
    after = existing_marks()
    new = sorted(after - before)
    if not new:
        print("[{}] no new mark found".format(tag))
        return None
    mark = new[-1]
    res = parse_run(mark)
    if res:
        print("[{}] mark={} wall={:.0f}s".format(tag, mark, dt))
        for e in res["epochs"]:
            print("   ep{:>2}  loss={:>7}  gacc={:>6}  mIoU={:>6}".format(
                e["epoch"], e["loss"], e["gacc"], e["miou"]))
        res["tag"] = tag
        res["mark"] = mark
        res["wall"] = dt
    return res


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = []

    if which in ("all", "w"):
        # 权重方案 x 相同预算
        for tag, cw in [("none", "none"), ("median", "median"), ("inverse", "inverse")]:
            r = run(tag, ["--limit", "8000", "--epochs", "3", "--batch-size", "32",
                          "--class-weight", cw])
            if r:
                out.append(r)

    if which in ("all", "b"):
        # 训练预算（不加权）
        for tag, ep, bs, lim in [("short", "2", "32", "8000"),
                                 ("mid", "6", "32", "8000"),
                                 ("mid16k", "6", "32", "16000")]:
            r = run(tag, ["--limit", lim, "--epochs", ep, "--batch-size", bs,
                          "--class-weight", "none"])
            if r:
                out.append(r)

    if which in ("all", "bs"):
        # batch size / lr 对照（同 16k 子集 / 12 epoch，raw 深度）
        # 注：batch 128 实测会在真实训练中 OOM（峰值约 17 GiB > 24 GiB 可用），
        #     故稳定上限取 96。
        for tag, bs, lr in [("bs32", "32", "5e-5"),
                            ("bs96", "96", "5e-5"),
                            ("bs96lr1.5e4", "96", "1.5e-4")]:
            r = run(tag, ["--limit", "16000", "--epochs", "12", "--batch-size", bs,
                          "--lr", lr, "--num-workers", "8", "--class-weight", "none"])
            if r:
                out.append(r)

    print("\n\n" + "#" * 78)
    print("汇总")
    print("#" * 78)
    for r in out:
        best = max((e["miou"] for e in r["epochs"] if e["miou"] is not None), default=None)
        last = r["epochs"][-1] if r["epochs"] else {}
        print("{:>8}  mark={}  wall={:>6.0f}s  last_mIoU={}  best_mIoU={}".format(
            r["tag"], r["mark"], r["wall"], last.get("miou"), best))
        for e in r["epochs"]:
            print("     ep{} mIoU={} gacc={} loss={}".format(
                e["epoch"], e["miou"], e["gacc"], e["loss"]))
