# source: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
import torch
from torch import nn
from Utils.distributed_utils import ConfusionMatrix, MetricLogger, SmoothedValue


# [SDCombo-patch] 原实现硬编码 ignore_index=0，但 VKITTI 2 的类别 0 是
# Terrain（训练集占 17.7%、验证集占 3.8% 的真实类别），并非忽略标签。
# 原设定等于丢弃了 17.7% 的监督信号，导致该类别永远学不会（IoU 恒为 0）。
# 这里改为可配置，默认 255（dataset_*.py 的 collate_fn 正是用 255 做 padding 值，
# 且 VKITTI 标签只取 0..14，不存在 255），从而让全部 15 类都参与训练。
IGNORE_INDEX = 255


def criterion(inputs, target, weight=None, ignore_index=None):
    """交叉熵损失。

    [SDCombo-patch] 新增 class weight 支持：VKITTI 2 的类别分布极度不平衡
    （Pole 0.18% vs Vegetation 26.7%，相差 145 倍），无权重时模型会坍缩到少数
    大类上。传入 weight 后按类别频率重加权（建议用中位数频率法）。
    """
    return nn.functional.cross_entropy(
        inputs, target,
        weight=weight,
        ignore_index=IGNORE_INDEX if ignore_index is None else ignore_index)


def evaluate(model, data_loader, device, num_classes, record_mark, ignore_index=None):
    model.eval()
    # [SDCombo-patch] 混淆矩阵的 mean IoU 需要与损失用同一个 ignore_index，
    # 否则被忽略的类别仍会被计入指标，结论与训练目标不一致。
    confmat = ConfusionMatrix(num_classes,
                             ignore_index=IGNORE_INDEX if ignore_index is None else ignore_index)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, annotation, depth in metric_logger.log_every(data_loader, 100, record_mark, header):
            image = image.to(device, non_blocking=True)
            annotation = annotation.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)
            output = model(image, depth)

            confmat.update(annotation.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def build_class_weights(freqs, mode="median", device=None, max_ratio=50.0):
    """按训练集类别频率构造损失权重。

    Args:
        freqs: 长度为 num_classes 的序列（各类像素数或频率），类别顺序需与标签一致。
        mode: 'median' -> median(freq)/freq（对长尾更稳）；'inv' -> 1/freq 归一化。
        max_ratio: 权重上限相对于最小权重的倍数，避免极稀有类权重爆炸。
    """
    f = torch.as_tensor(freqs, dtype=torch.float64)
    f = f.clamp(min=0)
    if mode == "inv":
        w = torch.where(f > 0, 1.0 / f.clamp(min=1e-12), torch.zeros_like(f))
    else:
        med = f[f > 0].median() if (f > 0).any() else torch.tensor(1.0, dtype=torch.float64)
        w = torch.where(f > 0, med / f.clamp(min=1e-12), torch.zeros_like(f))
    if (w > 0).any():
        pos = w[w > 0]
        w = w.clamp(max=pos.min() * max_ratio)
        w = w / pos.mean()                       # 归一化，保持损失量级
    return w.to(dtype=torch.float32, device=device)


def create_lr_scheduler(optimizer, epochs, lr_min, warmup_epochs=0, last_epoch=-1):
    """学习率：可选线性 warmup + 余弦退火到 lr_min。

    [SDCombo-patch] 原实现为 `CosineAnnealingLR(optimizer, T_max=half_life, ...)`，
    其中 half_life 来自 `--cos` 的第一个元素，与 `--epochs` 完全解耦：
    轮数大于 half_life 时学习率会在中途降到 lr_min 并一直平移到底。
    官方 InternImage 配置是 ~160k iter 的 poly 衰减 + 1500 iter linear warmup，
    而本仓库既无 warmup，T_max 也不等于总轮数。这里改为 T_max=epochs（余弦在
    整个训练周期内走完半周期），并支持 warmup。
    """
    total = max(int(epochs), 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total, eta_min=lr_min, last_epoch=last_epoch)
    if warmup_epochs and warmup_epochs > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=max(int(warmup_epochs), 1))
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cosine], milestones=[max(int(warmup_epochs), 1)])
    return cosine


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler,
                    record_mark, print_freq=10, scaler=None, class_weights=None,
                    max_grad_norm=0.0, optimizer_step_interval=1):
    """
    Args:
        model: model to be trained
        optimizer: learning rate setting
        data_loader: dataset
        device: device training on
        epoch: epoch times
        lr_scheduler: learning rate scheduler
        print_freq: report time
        record_mark: record file mark
        scaler: Automatic Mixed Precision
        class_weights: 类别权重张量（缓解长尾类别坍缩），None 表示不加权
        max_grad_norm: >0 时做梯度裁剪。原实现完全没有裁剪，实测最大梯度范数
            约 59，配合过大学习率会在数千步后发散为 nan。
        optimizer_step_interval: 梯度累积步数，用于在有限显存下等效放大 batch。
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.4e}'))
    metric_logger.add_meter('gnorm', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr = 0.0
    gnorm = 0.0
    accum = max(int(optimizer_step_interval), 1)
    for step, (image, annotation, depth) in enumerate(
            metric_logger.log_every(data_loader, print_freq, record_mark, header)):
        image = image.to(device, non_blocking=True)
        annotation = annotation.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(image, depth)
            loss = criterion(output, annotation, weight=class_weights)
            loss = loss / accum

        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                if max_grad_norm and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    gnorm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (step + 1) % accum == 0:
                if max_grad_norm and max_grad_norm > 0:
                    gnorm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item() * accum, lr=lr,
                             gnorm=gnorm)

    lr_scheduler.step()

    return metric_logger.meters["loss"].global_avg, lr
