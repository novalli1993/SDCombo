"""在 VKITTI 2 上训练论文主线模型 SDCombo（`Joint/model_DL4sDL.py`）。

与另外两个训练脚本的关系：
    train_HHA.py   —— Stanford2D3D + HHA 深度编码（论文 §6 的主线数据集）
    train_depth.py —— Stanford2D3D + 原始深度（论文 §6.3.1 的对照组）
    train_VKITTI.py—— **VKITTI 2 + 原始深度**（本机现有的数据集；模型完全相同）

设计要点（只借用「训练方法」，模型仍是本仓库论文主线那一个）：
    * 损失：Focal Loss(alpha=0.5, gamma=2)，`ignore_index=255`
      —— VKITTI 的 0..13 全是真实类别，255 是数据集 collate 的填充值
    * 优化：AdamW(betas=(0.9,0.999), weight_decay=0.01)、warmup + 余弦退火
    * 稳定：梯度裁剪（默认 max_norm=1.0）、AMP、类别权重（默认 median frequency）
    * 产物：work_dir/{logger,evaluation,model,board}，逐 epoch 存权重并记录指标

用法（仓库根目录，先 `conda activate SDCombo`）：
    python train_VKITTI.py --data-path datasets\\VKITTI_II --batch-size 32 --epochs 10
    python train_VKITTI.py --limit 512 --epochs 1 --batch-size 8     # 快速自检

本机实测（RTX 5090 D v2 / 24 GB，crop 256，AMP，`tools\\bench_train.py`）：
    batch 16 -> 231 img/s / 2.2 GB      batch 32 -> 430 img/s / 4.4 GB
    batch 48 -> 498 img/s / 6.3 GB      batch 64 -> 525 img/s / 8.3 GB
    37860 张训练图 => batch 32 时约 90 s/epoch，显存远未用满，故默认取 32（论文 §6.2 用 30）。
"""
import datetime
import os

import torch

import Dataset.transforms as T
from Dataset.dataset_VKITTI import VKITTI
from Joint.model_DL4sDL import _SDCombo as SDCombo
from Utils.train_val_vkitti import (build_class_weights, create_lr_scheduler, evaluate,
                                    load_class_histogram, train_one_epoch)

try:  # tensorboard 只是可选的观测手段，缺了也应该能训练
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

# 与 train_HHA.py / train_depth.py 保持一致：ImageNet 归一化
# （主干 MobileNetV3-Large 是 ImageNet 预训练的，用同一套统计量最匹配）
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def create_model(pretrained, num_classes):
    model = SDCombo(aux=True, num_classes=num_classes)
    missing_keys, unexpected_keys = [], []
    if pretrained:
        # PyTorch 2.6+ 的 torch.load 默认 weights_only=True，而权重里存了 argparse.Namespace
        weights_dict = torch.load(pretrained, map_location="cpu", weights_only=False)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if missing_keys:
            print("missing keys: ", ", ".join(missing_keys[:20]), "..." if len(missing_keys) > 20 else "")
        if unexpected_keys:
            print("unexpected_keys: ", ", ".join(unexpected_keys[:20]), "..." if len(unexpected_keys) > 20 else "")
    return model, missing_keys, unexpected_keys


class PipelineTrain:
    """训练管线：随机缩放 -> 随机水平翻转 -> 随机裁剪。

    VKITTI 原图 1242x375，缩放范围取 [crop_size, base_size]，默认 [256, 375]。
    """

    def __init__(self, base_size, crop_size, mean, std, hflip_prob=0.5):
        min_size = crop_size
        max_size = max(base_size, crop_size)
        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([T.RandomCrop(crop_size), T.ToTensor(), T.Normalize(mean=mean, std=std)])
        self.transforms = T.Compose(trans)

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)


class PipelineEval:
    """验证管线。

    默认 **中心裁剪**（同一 checkpoint 每次评估结果一致，便于挑选最佳权重）；
    加 `--eval-random-crop` 可切回论文 §6.4.1 的「随机裁剪」快速评估协议。
    """

    def __init__(self, crop_size, mean, std, random_crop=False):
        crop = T.RandomCrop(crop_size) if random_crop else T.CenterCrop(crop_size)
        self.transforms = T.Compose([crop, T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)


def get_transform(train, base_size=375, crop_size=256, random_crop=False):
    return PipelineTrain(base_size, crop_size, mean=MEAN, std=STD) if train \
        else PipelineEval(crop_size, mean=MEAN, std=STD, random_crop=random_crop)


def main(args):
    torch.cuda.empty_cache()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes
    num_workers = args.num_workers
    # 本机 Windows 分页文件偏小，pin_memory 下 worker 过多会静默卡死，默认 4 足够
    # （实测数据加载 ~0.0001 s/iter，瓶颈在 GPU 而不是数据管线）
    if args.test:
        mark = None
    else:
        mark = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 数据集 ----
    train_dataset = VKITTI(args.data_path, "training",
                           transforms=get_transform(True, args.base_size, args.crop_size))
    val_dataset = VKITTI(args.data_path, "validation",
                         transforms=get_transform(False, args.base_size, args.crop_size,
                                                  random_crop=args.eval_random_crop))
    if args.limit > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.limit, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(max(args.limit // 4, 8), len(val_dataset))))
    print("train={}  val={}  batch_size={}  crop={}  workers={}".format(
        len(train_dataset), len(val_dataset), batch_size, args.crop_size, num_workers))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        drop_last=(len(train_dataset) % batch_size == 1),  # 只丢掉"只剩 1 张"的尾巴（BatchNorm 需要 >1）
        shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=VKITTI.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=VKITTI.collate_fn)

    # ---- 模型 ----
    model, missing_keys, unexpected_keys = create_model(args.pretrained, num_classes)
    model.to(device)
    print("参数量: {:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # ---- 类别权重 ----
    class_weights = None
    if args.class_weight != "none":
        try:
            counts, source = load_class_histogram(args.class_stats, args.data_path, num_classes)
            class_weights = build_class_weights(counts, mode=args.class_weight, num_classes=num_classes)
            print("类别权重({}，来源 {}) = {}".format(
                args.class_weight, source, [round(float(w), 3) for w in class_weights]))
            class_weights = class_weights.to(device)
        except (FileNotFoundError, ValueError) as exc:
            print("[WARN] 类别权重不可用，已退化为不加权（仅用 Focal Loss）：{}".format(exc))
            print("       提示：把 prepare_vkitti.py 产出的 vkitti_report.json 放到 "
                  "datasets/vkitti_report.json，或用 --class-stats 指向它")
            class_weights = None

    # ---- 优化器 / 调度 / AMP ----
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    lr_scheduler, warmup_epochs, cosine_epochs = create_lr_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.lr_min)

    # ---- 断点续训 ----
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        mark = os.path.split(args.resume)[1][6:21]
        print("resume from {} (start_epoch={})".format(args.resume, args.start_epoch))

    # ---- 记录文件 ----
    results_file = None
    writer = None
    if mark is not None:
        results_file = "work_dir/evaluation/evaluation{}.txt".format(mark)
        args_dict = vars(args)
        with open(results_file, "a", encoding="utf-8") as f:
            f.write("# train_VKITTI.py\n")
            for key in args_dict:
                f.write("{}: {}\n".format(key, args_dict[key]))
            f.write("mark: {}\n".format(mark))
            f.write("warmup_epochs={} cosine_T_max={} ignore_index=255\n\n".format(warmup_epochs, cosine_epochs))
        if SummaryWriter is not None:
            writer = SummaryWriter("work_dir/board/" + mark)
        else:
            print("[WARN] 未安装 tensorboard，跳过 TensorBoard 记录（pip install tensorboard 可启用）")
    print(vars(args))

    # ---- 训练 ----
    start_time = datetime.datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, writer,
                                        lr_scheduler, mark, args.print_freq, scaler,
                                        class_weights=class_weights,
                                        max_grad_norm=args.max_grad_norm,
                                        ignore_index=255, alpha=args.focal_alpha,
                                        gamma=args.focal_gamma)
        if mark is not None:
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if scaler is not None:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, "work_dir/model/model_{}_{}.pth".format(mark, epoch))
            if writer is not None:
                writer.add_scalar("loss/epoch", mean_loss, epoch)
                writer.add_scalar("lr/epoch", lr, epoch)

        # ---- 逐 epoch 快速评估 ----
        if args.evaluation:
            torch.cuda.empty_cache()
            confmat = evaluate(model, val_loader, device=device, num_classes=num_classes,
                               record_mark=mark)
            val_info = str(confmat)
            print(val_info)
            if mark is not None:
                if writer is not None:
                    lines = val_info.split("\n")
                    try:
                        writer.add_scalar("Acc/epoch", float(lines[2].split(":")[1]), epoch)
                        writer.add_scalar("mIoU/epoch", float(lines[4].split(":")[1]), epoch)
                    except (IndexError, ValueError):
                        pass
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write("[epoch: {}]\n".format(epoch))
                    f.write("time: {}\n".format(datetime.datetime.now().strftime("%H:%M:%S")))
                    f.write("train_loss: {:.4f}\n".format(mean_loss))
                    f.write("lr: {:.4e}\n".format(lr))
                    f.write(val_info + "\n\n")

    total_time = datetime.datetime.now() - start_time
    print("training time {}".format(total_time))
    if mark is not None:
        with open(results_file, "a", encoding="utf-8") as f:
            f.write("training time {}\n\n".format(total_time))
    if writer is not None:
        writer.close()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SDCombo on VKITTI 2 (paper main model)")

    # meta
    parser.add_argument("--test", action="store_true", help="只跑不记录（不写 mark/日志/权重）")
    parser.add_argument("--message", default="VKITTI II. Raw depth. Focal loss.", type=str)

    # data
    parser.add_argument("--data-path", default="datasets/VKITTI_II", help="VKITTI_II 根目录")
    parser.add_argument("--num-classes", default=14, type=int, help="VKITTI 2 共 14 类(0..13)")
    parser.add_argument("--base_size", default=375, type=int, help="RandomResize 的上界（=原图高）")
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("-b", "--batch-size", default=32, type=int,
                        help="本机实测 batch 32/48/64 分别为 430/498/525 img/s（crop 256, AMP）")
    parser.add_argument("--eval-batch-size", default=8, type=int)
    parser.add_argument("--limit", default=0, type=int, help="只取前 N 张训练图（0=全部，用于快速自检）")
    parser.add_argument("--eval-random-crop", action="store_true",
                        help="验证时用随机裁剪（论文 §6.4.1 的快速评估协议）；默认中心裁剪")

    # training
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=5e-5, type=float, help="初始学习率（论文 5e-5）")
    parser.add_argument("--lr-min", default=1e-6, type=float, help="余弦退火下界")
    parser.add_argument("--warmup-epochs", default=1, type=int)
    parser.add_argument("--wd", "--weight-decay", default=0.01, type=float, dest="weight_decay",
                        help="权重衰减（论文 §6.2 用 0.01）")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="梯度裁剪阈值（0=不裁剪）")
    parser.add_argument("--class-weight", default="median", choices=["none", "median", "inverse"],
                        help="类别权重（VKITTI 类别极不平衡，默认 median frequency）")
    parser.add_argument("--class-stats", default="datasets/vkitti_report.json",
                        help="类别频率来源（prepare_vkitti.py 产出的报告）")
    parser.add_argument("--focal-alpha", default=0.5, type=float)
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--num-workers", default=4, type=int,
                        help="DataLoader worker 数；本机分页文件小，>8 有静默卡死风险")
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--no-amp", action="store_true", help="关闭混合精度（默认开启）")
    parser.add_argument("--resume", default="", help="从 checkpoint 续训")
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--evaluation", action="store_true", default=True, help="逐 epoch 评估（默认开）")
    parser.add_argument("--no-evaluation", action="store_false", dest="evaluation")
    parser.add_argument("--pretrained", default="",
                        help="预训练权重；例如 work_dir/model/init_from_DeepLabV3_inside.pth")

    args = parser.parse_args()
    args.amp = not args.no_amp
    return args


if __name__ == "__main__":
    args = parse_args()
    for d in ("work_dir", "work_dir/evaluation", "work_dir/logger", "work_dir/model", "work_dir/board"):
        os.makedirs(d, exist_ok=True)
    main(args)
