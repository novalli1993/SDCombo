# source: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
import torch
from torch import nn
from Utils.distributed_utils import ConfusionMatrix, MetricLogger, SmoothedValue


class FocalLoss():
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='none')

    def __call__(self, inputs, target):
        # Focal Loss
        if len(inputs) == 1:
            logpt = -self.ce_fn(inputs, target)
            loss = torch.empty_like(logpt)
            for i in range(len(logpt)):
                loss[i]=-self.alpha * ((1 - torch.exp(logpt[i])) ** self.gamma) * logpt[i]
            return torch.mean(loss)
        else:
            losses = {}
            for name, x in inputs.items():
                logpt = -self.ce_fn(x, target)
                loss = torch.empty_like(logpt)
                for i in range(len(logpt)):
                    loss[i] = -self.alpha * ((1 - torch.exp(logpt[i])) ** self.gamma) * logpt[i]
                losses[name]=torch.mean(loss)
            loss = losses['out'] + 0.5 * losses['aux']
        return loss


def criterion(inputs, target):
    # Cross Entropy Loss
    # if len(inputs) == 1:
    #     return nn.functional.cross_entropy(inputs, target, ignore_index=0)
    # else:
    #     losses = {}
    #     for name, x in inputs.items():
    #         losses[name] = nn.functional.cross_entropy(x, target, ignore_index=0)
    #     return losses['out'] + 0.5 * losses['aux']

    # Focal Loss
    fl = FocalLoss(alpha=0.5, gamma=2, weight=None, ignore_index=0)
    losses = fl(inputs, target)
    return losses


def evaluate(model, data_loader, device, num_classes, record_mark):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, annotation, depth in metric_logger.log_every(data_loader, 100, record_mark, header):
            image, annotation, depth = image.to(device), annotation.to(device), depth.to(device)
            output = model(image, depth)['out']

            confmat.update(annotation.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def create_lr_scheduler(optimizer, half_life, lr_min):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=half_life, eta_min=lr_min)


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, lr_scheduler, record_mark, print_freq=10,
                    scaler=None):
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

    step = 0
    for image, annotation, depth in metric_logger.log_every(data_loader, print_freq, record_mark, header):
        image, annotation, depth = image.to(device), annotation.to(device), depth.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image, depth)
            loss = criterion(output, annotation)
            if record_mark is not None:
                writer.add_scalar("loss/batch", loss, step + len(data_loader) * epoch)

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
        if record_mark is not None:
            writer.add_scalar("lr/batch", lr, step + len(data_loader) * epoch)
        step += 1

    lr_scheduler.step()

    return metric_logger.meters["loss"].global_avg, lr
