"""VKITTI 2 数据集专用的训练/评估工具（配合论文主线模型 Joint/model_DL4sDL.py）。

与 `Utils/train_val.py`（Stanford2D3D 用）的差异，都是为了 VKITTI 这个数据集：

1. **`ignore_index = 255`**。VKITTI 的 14 个类别 0..13 全部是真实类别（0 = Terrain，
   训练集占约 16%），没有 Stanford2D3D 那种 `<UNK>`；而 `Dataset/dataset_VKITTI.py`
   的 `collate_fn` 用 255 填充，所以 255 才是正确的忽略值。
   若沿用 `ignore_index=0`，相当于丢弃 16% 的监督信号且让 Terrain 永远学不会。
2. **可选的类别权重**（median / inverse frequency）。VKITTI 类别极不平衡
   （最稀有类 0.26% vs 最多类 26.6%，相差约 100 倍），权重来自数据准备阶段
   生成的 `datasets/vkitti_report.json` 的 `class_histogram`。
3. **Focal Loss 向量化**：与原实现的逐样本 Python 循环数学等价
   （`FL = -alpha * (1 - p_t)^gamma * log(p_t)`），但快很多。
4. **warmup + 余弦退火、梯度裁剪、现代 AMP API**（`torch.amp.*`，无弃用告警）。

数值口径与仓库其余脚本保持一致：仍然用 `Utils/distributed_utils.ConfusionMatrix`
统计（其 `a >= 0 & a < n` 天然把 255 排除在外），mIoU 由 `ConfusionMatrix.__str__`
给出（VKITTI 没有 `<UNK>`，因此宏里那句 "Ignore '<UNK>'" 对 VKITTI 是无害的误标）。
"""
import math
import os
import datetime

import torch
from torch import nn

from Utils.distributed_utils import ConfusionMatrix, MetricLogger, SmoothedValue


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
class FocalLoss:
    """Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)。

    Args:
        alpha: 平衡因子（论文 §6.2 用 0.5）
        gamma: 聚焦参数（论文 §6.2 用 2）
        weight: 每个类别的权重 tensor，或 None
        ignore_index: 忽略的标签值（VKITTI 用 255）
    """

    def __init__(self, alpha=0.5, gamma=2.0, weight=None, ignore_index=255):
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                         reduction="none")

    def _single(self, logits, target):
        logpt = -self.ce_fn(logits, target)          # = log(p_t)，被忽略的像素为 0
        pt = torch.exp(logpt)
        loss = -self.alpha * (1.0 - pt) ** self.gamma * logpt
        return loss.mean()

    def __call__(self, inputs, target):
        if torch.is_tensor(inputs):
            return self._single(inputs, target)
        losses = {name: self._single(x, target) for name, x in inputs.items()}
        if "aux" in losses:
            return losses["out"] + 0.5 * losses["aux"]
        return losses["out"]


def criterion(inputs, target, weight=None, ignore_index=255, alpha=0.5, gamma=2.0):
    """统一的损失入口，签名与 Utils/train_val.py 的 criterion 兼容（多两个可选项）。"""
    return FocalLoss(alpha=alpha, gamma=gamma, weight=weight, ignore_index=ignore_index)(inputs, target)


# ---------------------------------------------------------------------------
# 类别权重
# ---------------------------------------------------------------------------
def load_class_histogram(stats_path, data_path=None, num_classes=14, max_files=None):
    """取类别像素直方图。

    优先读数据准备脚本产出的 `vkitti_report.json` 的 `class_histogram`（全量精确值）；
    没有该文件时，退化为扫描 `annotations/training` 下的标签图（默认全量，`max_files`
    可限制张数），并在日志里标注来源。若扫描结果里有类别一个像素都没有，说明样本不完整，
    会显式告警——**不要**拿不完整的直方图去做类别加权（会把损失压成 0）。
    """
    if stats_path and os.path.isfile(stats_path):
        import json
        with open(stats_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        hist = report.get("class_histogram")
        if hist:
            counts = [int(hist.get(str(i), hist.get(i, 0))) for i in range(num_classes)]
            if sum(counts) > 0 and min(counts) > 0:
                return counts, "report:{}".format(os.path.basename(stats_path))
            print("[WARN] {} 的 class_histogram 不完整（有类别为 0 像素），忽略该文件".format(stats_path))

    if data_path:
        import numpy as np
        from PIL import Image
        ann_dir = os.path.join(data_path, "annotations", "training")
        if os.path.isdir(ann_dir):
            names = sorted(os.listdir(ann_dir))
            if max_files:
                names = names[:max_files]
            counts = [0] * num_classes
            for i, name in enumerate(names):
                arr = np.array(Image.open(os.path.join(ann_dir, name)))
                for c in range(num_classes):
                    counts[c] += int((arr == c).sum())
                if (i + 1) % 5000 == 0:
                    print("    已扫描 {}/{} 张标签 ...".format(i + 1, len(names)))
            missing = [c for c, v in enumerate(counts) if v == 0]
            if missing:
                print("[WARN] 扫描 {} 张标签后，这些类别像素数为 0：{}；"
                      "直方图不完整，建议改用全量扫描或提供 prepare_vkitti.py 的报告".format(
                          len(names), missing))
            return counts, "scan:{}x{}".format(len(names), os.path.basename(ann_dir))

    raise FileNotFoundError(
        "找不到类别频率来源：既没有 --class-stats 指定的 json，也无法扫描 {}/annotations/training"
        .format(data_path))


def build_class_weights(counts, mode="median", num_classes=14):
    """按类别像素频率构造损失权重。

    - median : w_c = median(count) / count_c      （中位数频率法）
    - inverse: w_c = (1/count_c) / mean(1/count)  （逆频率法）
    两种都会归一化到 mean(w) = 1，便于沿用同一套 lr。

    要求每个类别都有像素（`counts` 里不能有 0），否则权重会退化成 0/极大值，
    交叉熵被压成 0、训练等于空转——这种情况直接报错，由调用方决定是否放弃加权。
    """
    counts_t = torch.tensor([int(c) for c in counts], dtype=torch.float64)
    if counts_t.numel() != num_classes:
        raise ValueError("类别数不一致: {} vs {}".format(counts_t.numel(), num_classes))
    zero = (counts_t <= 0).nonzero().flatten().tolist()
    if zero:
        raise ValueError("这些类别在直方图中为 0 像素：{}，无法构造可靠的类别权重".format(zero))
    if mode == "median":
        w = counts_t.median() / counts_t
    elif mode == "inverse":
        w = 1.0 / counts_t
    else:
        raise ValueError("未知的 class-weight 模式: {}".format(mode))
    w = w / w.mean()
    return w.to(torch.float32)


# ---------------------------------------------------------------------------
# 学习率
# ---------------------------------------------------------------------------
def create_lr_scheduler(optimizer, epochs, warmup_epochs=1, lr_min=1e-6, warmup_start_factor=1e-3):
    """warmup（逐 epoch 线性升温）+ 余弦退火到 lr_min。

    论文 §6.2 的调度是「余弦退火、最小 lr 1e-7、T_max = 训练 epoch 数」；
    这里额外加了 1 个 epoch 的 warmup —— 这是在大 batch + AdamW 下更稳的做法
    （论文原文用的是 batch 30 且没有 warmup，若要对齐可把 warmup_epochs 设为 0）。
    """
    warmup_epochs = max(int(warmup_epochs), 0)
    cosine_epochs = max(int(epochs) - warmup_epochs, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr_min)
    if warmup_epochs <= 0:
        return cosine, warmup_epochs, cosine_epochs
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]), warmup_epochs, cosine_epochs


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, lr_scheduler, record_mark,
                    print_freq=100, scaler=None, class_weights=None, max_grad_norm=1.0,
                    ignore_index=255, alpha=0.5, gamma=2.0):
    """跑一个 epoch，返回 (平均损失, 当前 lr)。

    `max_grad_norm > 0` 时做梯度裁剪（默认 1.0）。本机实测未裁剪时梯度范数可达数十，
    训练数千步后出现 nan，因此默认开启。
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4e}"))
    metric_logger.add_meter("gnorm", SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Epoch: [{}]".format(epoch)

    step = 0
    for image, annotation, depth in metric_logger.log_every(data_loader, print_freq, record_mark, header):
        image, annotation, depth = image.to(device), annotation.to(device), depth.to(device)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            output = model(image, depth)
            loss = criterion(output, annotation, weight=class_weights, ignore_index=ignore_index,
                             alpha=alpha, gamma=gamma)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            else:
                gnorm = torch.tensor(0.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            else:
                gnorm = torch.tensor(0.0)
            optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr, gnorm=float(gnorm))
        if writer is not None:
            writer.add_scalar("loss/batch", loss.item(), step + len(data_loader) * epoch)
            writer.add_scalar("lr/batch", lr, step + len(data_loader) * epoch)
        step += 1

    lr_scheduler.step()
    return metric_logger.meters["loss"].global_avg, lr


def evaluate(model, data_loader, device, num_classes, record_mark, ignore_index=255):
    """验证集评估；返回值是 ConfusionMatrix（其 __str__ 打印 acc / 逐类 IoU / mIoU）。

    VKITTI 用 `class_names=VKITTI_CLASSES`、`mean_iou_skip=0`（14 类全部计入 mIoU）；
    256/255 之类越界标签由 ConfusionMatrix 的 `a >= 0 & a < n` 天然过滤。
    """
    from Dataset.dataset_VKITTI import CLASSES as VKITTI_CLASSES

    model.eval()
    confmat = ConfusionMatrix(num_classes, class_names=VKITTI_CLASSES, mean_iou_skip=0)
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    with torch.no_grad():
        for image, annotation, depth in metric_logger.log_every(data_loader, 100, record_mark, header):
            image, annotation, depth = image.to(device), annotation.to(device), depth.to(device)
            output = model(image, depth)["out"]
            confmat.update(annotation.flatten(), output.argmax(1).flatten())
        confmat.reduce_from_all_processes()
    return confmat


def epoch_time_str(start):
    return str(datetime.datetime.now() - start)
