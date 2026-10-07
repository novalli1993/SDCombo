"""验证 ConfusionMatrix 的修复：nan 消除、ignore 类排除、与手算一致。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from Utils.distributed_utils import ConfusionMatrix  # noqa: E402

FAIL = 0


def check(label, cond):
    global FAIL
    print(f"  {'OK  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAIL += 1


print("=" * 66)
print("[1] GT 中从未出现的类别不应产生 nan")
print("=" * 66)
n = 5
cm = ConfusionMatrix(n)
# 只在类别 0..2 上构造数据，类别 3/4 从未出现
gt = torch.tensor([0, 0, 1, 1, 2, 0, 1, 2])
pr = torch.tensor([0, 1, 1, 2, 2, 0, 1, 0])
cm.update(gt, pr)
acc_global, acc, iu = cm.compute()
check("compute() 无 nan", not torch.isnan(iu).any())
check("未出现类别 IoU = 0（而非 nan）", iu[3].item() == 0 and iu[4].item() == 0)
check("acc 无 nan", not torch.isnan(acc).any())

mi = cm.mean_iou(ignore_index=0)
check("mean_iou() 无 nan", not torch.isnan(mi))
# 手算（gt/pr 见上，共 8 个像素；只在类 0/1/2 上统计，类 3/4 从未出现）：
#   类1: TP=2 (idx2,6)  FN=1 (idx3)  FP=1 (idx7) -> IoU = 2/(2+1+1) = 0.5
#   类2: TP=1 (idx4)    FN=1 (idx5)  FP=1 (idx3) -> IoU = 1/(1+1+1) = 1/3
#   mean(排除类0) = (0.5 + 1/3) / 2
expect = (0.5 + 1 / 3) / 2
check(f"mean_iou 等于手算 {expect:.4f}（实得 {mi.item():.4f}）", abs(mi.item() - expect) < 1e-6)
check("类1 IoU = 0.5", abs(iu[1].item() - 0.5) < 1e-6)
check("类2 IoU = 1/3", abs(iu[2].item() - 1 / 3) < 1e-6)
# 交叉验证：混淆矩阵的列和应等于各类被预测的次数
pred_cnt = torch.bincount(pr, minlength=n).float()
check("混淆矩阵列和 == 各类被预测次数", torch.equal(cm.mat.sum(0).float(), pred_cnt))

print()
print("=" * 66)
print("[2] 忽略类 0 应从 mean IoU 中排除")
print("=" * 66)
check("mean_iou(ignore_index=0) != mean_iou(ignore_index=-1)",
      abs(cm.mean_iou(0).item() - cm.mean_iou(-1).item()) > 1e-6)

print()
print("=" * 66)
print("[3] 全部类别都出现时，与朴素实现一致")
print("=" * 66)
n2 = 4
cm2 = ConfusionMatrix(n2)
g = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 1, 1, 2, 3])
p = torch.tensor([0, 1, 2, 3, 1, 1, 2, 2, 1, 0, 2, 3])
cm2.update(g, p)
h = cm2.mat.float()
ref_iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
_, _, iu2 = cm2.compute()
check("iu 与朴素公式逐元素一致", torch.allclose(iu2, ref_iu, equal_nan=False))
# 所有类别都出现过，此时排除 0 号类后仍应可算
mi2 = cm2.mean_iou(ignore_index=0)
expect2 = ref_iu[1:].mean()
check(f"mean_iou(排除类0) = {expect2:.4f}", abs(mi2.item() - expect2.item()) < 1e-6)

print()
print("=" * 66)
print("[4] 空矩阵（极端情况）不应崩且不应出 nan")
print("=" * 66)
cm3 = ConfusionMatrix(3)
cm3.mat = torch.zeros((3, 3), dtype=torch.int64)
try:
    a, c, i = cm3.compute()
    check("空矩阵 compute() 不崩", True)
    check("空矩阵 acc_global 不是 nan", not torch.isnan(a))
    check("空矩阵 mean_iou 为 nan 且被显式返回", torch.isnan(cm3.mean_iou()))
except Exception as e:
    check(f"空矩阵 compute() 不崩（异常 {e}）", False)

print()
print("=" * 66)
print("全部通过" if FAIL == 0 else f"有 {FAIL} 项失败")
print("=" * 66)
sys.exit(1 if FAIL else 0)
