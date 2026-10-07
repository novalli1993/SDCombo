import os
import datetime

import Dataset.transforms as T
from Dataset.dataset_VKITTI import VKITTI
from Joint.model import SDCombo
from Utils.train_val import *


def create_model(pretrained, num_classes, with_cp=False, channels=64,
                 depths=(4, 4, 18, 4), groups=(4, 8, 16, 32)):
    model = SDCombo(num_classes, channels=channels, depths=depths,
                    groups=groups, with_cp=with_cp)
    missing_keys = unexpected_keys = []
    if pretrained is not None:
        # 同 evaluation.py：checkpoint 内含 args(argparse.Namespace)，
        # PyTorch 2.6+ 默认 weights_only=True 无法反序列化，需显式关闭。
        weights_dict = torch.load(pretrained, map_location='cpu', weights_only=False)['model']
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0:
            print("missing keys: ", end='')
            for i in missing_keys:
                print(i, end=', ')
            print('\n')
        if len(unexpected_keys) != 0:
            print("unexpected_keys: ", end='')
            for i in unexpected_keys:
                print(i, end=', ')
            print('\n')
    return model, missing_keys, unexpected_keys


def get_transform(train, base_size=375, crop_size=256):
    """训练/验证的数据管线。

    [SDCombo-patch] base_size / crop_size 改为可配置（原来硬编码 375/256）。
    注意 base_size 决定 RandomResize 的缩放范围 [0.75*base, 2.0*base]：
      - 当 base_size 较小（如 375）时，缩放上界 750 远大于 crop，裁剪掉大量内容；
      - 想让 crop 真正"看得更多"，应同步放大 base_size（建议 base_size ≈ crop_size）。
    """
    mean = (33.6045, 33.9644, 27.2941)
    std = (19.3824, 19.3147, 20.1879)
    return PipelineTrain(base_size, crop_size, mean=mean, std=std) if train else \
        PipelineEval(crop_size, mean=mean, std=std)


class PipelineTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(33.6045, 33.9644, 27.2941),
                 std=(19.3824, 19.3147, 20.1879)):
        min_size = int(0.75 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)


class PipelineEval:
    """验证/推理管线。

    [SDCombo-patch] 原实现用 `T.RandomCrop(crop_size)`，即**评估时也做随机裁剪**：
      1) 每轮评估看到的 256x256 区域都不同，指标不可复现；
      2) VKITTI 原图 1242x375，256 裁切只覆盖约 14% 的像素（中心裁剪 375 也仅 30%）。
    改为确定性中心裁剪（`T.CenterCrop`），保证同一 checkpoint 每次评估结果一致。
    如需评估整幅图（不裁剪），用 `PipelineEvalNoCrop`；本机实测整图前向仅
    1.61 GiB / 0.70s，24GB 卡完全放得下。
    """

    def __init__(self, crop_size, mean=(33.6045, 33.9644, 27.2941), std=(19.3824, 19.3147, 20.1879)):
        self.transforms = T.Compose([
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)


class PipelineEvalNoCrop:
    """整幅图评估管线（不做任何裁剪/缩放），用于得到覆盖全图的指标。"""

    def __init__(self, mean=(33.6045, 33.9644, 27.2941), std=(19.3824, 19.3147, 20.1879)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)


def compute_class_weights(data_path, num_classes, mode="median", device=None,
                          cache=True):
    """统计训练集类别像素频率并构造损失权重（用于缓解长尾类别坍缩）。

    统计需要遍历全部训练标签（37,860 张），较慢；结果按 (数据路径, 类别数) 缓存到
    work_dir/class_freq.json，后续运行直接复用。
    """
    import json

    import numpy as np
    from PIL import Image

    ann_dir = os.path.join(data_path, "annotations", "training")
    cache_file = os.path.join("work_dir", "class_freq.json")
    key = f"{os.path.abspath(ann_dir)}|{num_classes}"
    hist = None

    if cache and os.path.isfile(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("key") == key:
                hist = np.array(cached["hist"], dtype=np.int64)
                print(f"  复用缓存 {cache_file}")
        except Exception:
            hist = None

    if hist is None:
        files = sorted(os.listdir(ann_dir))
        hist = np.zeros(num_classes, dtype=np.int64)
        for i, f in enumerate(files):
            a = np.asarray(Image.open(os.path.join(ann_dir, f)))
            valid = a[(a >= 0) & (a < num_classes)]
            if valid.size:
                hist += np.bincount(valid.ravel(), minlength=num_classes)[:num_classes]
            if (i + 1) % 10000 == 0:
                print(f"    已统计 {i + 1}/{len(files)} 张标签 ...")
        if cache:
            os.makedirs("work_dir", exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"key": key, "hist": hist.tolist()}, f)

    print(f"  训练集类别像素分布: {hist.tolist()}")
    return build_class_weights(hist, mode=mode, device=device), hist


def main(args):
    # device, batch size, number of classes and workers
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes
    # [SDCombo-patch] 原为 min(cpu, batch_size, 8)，被 batch size 压制。
    # 图片解码是 CPU 任务、不占显存，这里直接允许显式指定。
    num_workers = args.num_workers if args.num_workers >= 0 else min(
        os.cpu_count() or 1, max(batch_size, 1) * 2, 16)

    # record mark
    mark = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # record file
    results_file = "work_dir/evaluation/evaluation{}.txt".format(mark)

    # build dataset
    train_dataset = VKITTI(args.data_path, "training",
                           transforms=get_transform(True, args.base_size, args.crop_size))
    val_dataset = VKITTI(args.data_path, "validation",
                         transforms=get_transform(False, args.base_size, args.crop_size))

    persistent = num_workers > 0
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True,
                                               persistent_workers=persistent,
                                               prefetch_factor=4 if persistent else None,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.eval_batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             persistent_workers=persistent,
                                             collate_fn=val_dataset.collate_fn)

    # build the model from pretrained and local it on device
    pretrained = args.pretrained
    model, missing_keys, unexpected_keys = create_model(
        pretrained, num_classes, with_cp=args.with_cp)
    model.to(device)

    # collect the parameters to be optimized
    for n, p in model.named_parameters():
        if n in missing_keys:
            p.requires_grad = True
        elif n.split('.')[0] in args.module_trained and n not in missing_keys:
            p.requires_grad = True
        else:
            p.requires_grad = False

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params_to_optimize)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {n_train / 1e6:.2f} M / 总参数 {n_total / 1e6:.2f} M")
    print("Parameters unfreeze:")
    for i in [n for n, p in model.named_parameters() if p.requires_grad]:
        print(i)

    # 类别权重（缓解长尾类别坍缩）
    class_weights = None
    if args.class_weight != "none":
        print(f"\n统计类别频率以构造损失权重 (mode={args.class_weight}) ...")
        class_weights, hist = compute_class_weights(
            args.data_path, num_classes, mode=args.class_weight, device=device)
        print(f"  类别权重: {[round(float(w), 3) for w in class_weights]}")

    # set the optimizer
    # use AdamW；fused=True 在 CUDA 上更快（单卡无 AMP 兼容问题）
    try:
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=args.lr, betas=(0.9, 0.999),
            weight_decay=args.weight_decay, fused=True)
        print("优化器: AdamW(fused=True)")
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=args.lr, betas=(0.9, 0.999),
            weight_decay=args.weight_decay)
        print("优化器: AdamW(fused=False)")

    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    # 学习率：T_max = 总轮数（原实现把 --cos 的 half-life 当 T_max，与 --epochs 解耦）
    lr_scheduler = create_lr_scheduler(optimizer, args.epochs, args.lr_min,
                                       warmup_epochs=args.warmup_epochs)

    # log meta info
    with open(results_file, "a") as f:
        f.write(f"Pretrained: {pretrained}\n")
        f.write(f"Module to be trained: {args.module_trained}\n")
        f.write(f"batch_size={batch_size} base_size={args.base_size} "
                f"crop_size={args.crop_size} lr={args.lr} epochs={args.epochs} "
                f"warmup_epochs={args.warmup_epochs} amp={args.amp} "
                f"max_grad_norm={args.max_grad_norm} "
                f"class_weight={args.class_weight} with_cp={args.with_cp}\n\n")

    # start the training
    start_time = datetime.datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        # loss and learning rate
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            lr_scheduler=lr_scheduler, record_mark=mark, print_freq=args.print_freq,
            scaler=scaler, class_weights=class_weights,
            max_grad_norm=args.max_grad_norm)

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes,
                           record_mark=mark)
        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # record: train_loss, lr, val_set corresponding to each epoch
            train_info = f"[epoch: {epoch}]\n" \
                         f"time: {datetime.datetime.now().strftime('%H:%M:%S')}\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.8f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        torch.save(save_file, "work_dir/model/model_{}_{}.pth".format(mark, epoch))

    total_time = datetime.datetime.now() - start_time
    print("training time {}".format(total_time))


def _comma_list(s):
    return [x.strip() for x in s.split(',') if x.strip()]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SDCombo training")

    # ---- 数据 ----
    parser.add_argument("--data-path", default="datasets/VKITTI_II", help="Dataset root")
    parser.add_argument("--fold-num", default=0, type=int,
                        help="Training & Testing allocation:\n1: [[1, 2, 3, 4, 6], [5]],\n"
                             "2: [[1, 2, 3, 4, 6], [2, 4]],\n3: [[2, 4, 5], [1, 3, 6]]")
    parser.add_argument("--num-classes", default=15, type=int)
    parser.add_argument("--base-size", default=375, type=int,
                        help="RandomResize 的基准尺寸，缩放范围 [0.75x, 2.0x]；"
                             "想让 crop 看得更多应把它调到与 crop-size 相当")
    parser.add_argument("--crop-size", default=256, type=int, help="随机裁剪尺寸")
    parser.add_argument("--eval-batch-size", default=8, type=int,
                        help="训练中每个 epoch 验证时的 batch size"
                             "（原实现硬编码为 1，验证开销随图数线性增长）")
    parser.add_argument("--num-workers", default=-1, type=int,
                        help="-1 表示自动（min(cpu, 2*batch, 16)）")

    # ---- 硬件 ----
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--with-cp", action="store_true",
                        help="启用梯度检查点（省显存、略慢）")

    # ---- 优化 ----
    parser.add_argument("--epochs", default=10, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=6e-5, type=float,
                        help='初始学习率。注意官方 InternImage-S(AdamW) 用 6e-5；'
                             '原实现默认 1e-2 高 166 倍，实测会在 epoch 1 内发散为 nan')
    parser.add_argument('--lr-min', default=1e-6, type=float, help='余弦退火的最小学习率')
    parser.add_argument('--warmup-epochs', default=0, type=int,
                        help='线性 warmup 轮数（官方配 1500 iter，约 0.16 epoch）')
    parser.add_argument('--max-grad-norm', default=1.0, type=float,
                        help='梯度裁剪阈值；<=0 表示不裁剪（原实现无裁剪）')
    parser.add_argument('--class-weight', default="median",
                        choices=["none", "median", "inv"],
                        help='类别权重方式，缓解长尾类别坍缩')
    parser.add_argument('--wd', '--weight-decay', default=0.05, type=float,
                        metavar='W', help='weight decay (default: 0.05)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--module_trained',
                        default='internimage,upernet,SDHead',
                        type=_comma_list,
                        help='module to be trained, comma separated: '
                             'internimage,upernet,SDHead (default: all)')
    parser.add_argument("--pretrained", default=None,
                        help="Pretrained weight, best: model_20230729_183311_10.pth")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # work_dir 下的 logger / evaluation / model 三个子目录分别被
    # MetricLogger、结果记录和 checkpoint 保存直接写入，必须预先存在。
    for _sub in ("work_dir/logger", "work_dir/evaluation", "work_dir/model"):
        os.makedirs(_sub, exist_ok=True)

    main(args)
