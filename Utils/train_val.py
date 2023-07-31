# source: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
import torch
from torch import nn
from Utils.distributed_utils import ConfusionMatrix, MetricLogger, SmoothedValue


def criterion(inputs, target):
    losses = nn.functional.cross_entropy(inputs, target, ignore_index=0)

    return losses


def evaluate(model, data_loader, device, num_classes, record_mark):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
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

        with torch.cuda.amp.autocast(enabled=scaler is not None):
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
