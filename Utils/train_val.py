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


def criterion(inputs, target):
    losses = nn.functional.cross_entropy(inputs, target, ignore_index=IGNORE_INDEX)

    return losses


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
            image, annotation, depth = image.to(device), annotation.to(device), depth.to(device)
            output = model(image, depth)

            confmat.update(annotation.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def create_lr_scheduler(optimizer, half_life, lr_min, last_epoch):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=half_life, eta_min=lr_min, last_epoch=last_epoch)


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, record_mark, print_freq=10, scaler=None):
    """
    Args:
        model: model to be trained
        optimizer: learning rate setting
        data_loader: dataset
        device: device training on
        epoch: epoch times
        lr_scheduler: warm up
        print_freq: report time
        record_mark: record file mark
        scaler: Automatic Mixed Precision
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.4e}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr = []
    for image, annotation, depth in metric_logger.log_every(data_loader, print_freq, record_mark, header):
        image, annotation, depth = image.to(device), annotation.to(device), depth.to(device)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(image, depth)
            loss = criterion(output, annotation)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    lr_scheduler.step()

    return metric_logger.meters["loss"].global_avg, lr
