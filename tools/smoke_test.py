"""
训练/推理链路冒烟测试 —— 用合成数据跑通 SDCombo 前向 + 反向 + 评估。

不需要真实数据集，用于在正式训练前确认整条链路（数据集 -> 模型 -> 损失 -> 优化器 -> 评估）
在本机可正常运行，并测量显存占用以推荐 batch size。

用法（仓库根目录）:
    conda activate SDCombo
    python tools\\smoke_test.py                 # 默认 batch=1, crop=256
    python tools\\smoke_test.py --batch-size 4  # 测 batch=4 的显存
    python tools\\smoke_test.py --batch-size 2 --crop-size 128
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Dataset.dataset_VKITTI import VKITTI  # noqa: E402
from Joint.model import SDCombo  # noqa: E402
from Utils.train_val import criterion, create_lr_scheduler, evaluate, train_one_epoch  # noqa: E402
import Dataset.transforms as T  # noqa: E402


def build_synthetic_dataset(root, split, count, size=512):
    """在 root 下生成 VKITTI 目录结构的合成数据（images/annotations/depth/<split>）。"""
    paths = {k: os.path.join(root, k, split) for k in ("images", "annotations", "depth")}
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(count):
        name = f"rgb_S01_15l_c0_{i:03d}.png"
        rgb = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(os.path.join(paths["images"], name))

        # 语义标签：0..14，和 train.py 的 num_classes 一致；0 为 ignore_index
        sem = rng.integers(0, 15, (size, size), dtype=np.uint8)
        Image.fromarray(sem).save(os.path.join(paths["annotations"], name))

        # 深度：单通道灰度
        dep = rng.integers(1, 255, (size, size), dtype=np.uint8)
        Image.fromarray(dep).save(os.path.join(paths["depth"], name))
    return root


def get_transform(train, base_size=375, crop_size=256):
    mean = (33.6045, 33.9644, 27.2941)
    std = (19.3824, 19.3147, 20.1879)
    if train:
        trans = [T.RandomResize(int(0.75 * base_size), int(2.0 * base_size)),
                 T.RandomHorizontalFlip(0.5),
                 T.RandomCrop(crop_size),
                 T.ToTensor(),
                 T.Normalize(mean=mean, std=std)]
    else:
        trans = [T.RandomCrop(crop_size), T.ToTensor(), T.Normalize(mean=mean, std=std)]
    return T.Compose(trans)


def main():
    parser = argparse.ArgumentParser(description="SDCombo 训练/推理链路冒烟测试")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--crop-size", default=256, type=int)
    parser.add_argument("--num-classes", default=15, type=int)
    parser.add_argument("--samples", default=4, type=int, help="合成样本数量")
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "datasets", "VKITTI_II_smoke"))
    parser.add_argument("--amp", action="store_true", help="启用混合精度")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}  batch_size={args.batch_size}  crop_size={args.crop_size}  amp={args.amp}")

    # ---------------- 数据 ----------------
    print("\n[1/5] 生成/加载合成数据集 ...")
    # 关键：样本数必须 >= batch_size，否则一个 batch 装不满，
    # 实测出来的显存会被低估（曾因此得出"显存与 batch 无关"的错误结论）。
    need = max(args.samples, args.batch_size)
    existing = 0
    _tr = os.path.join(args.data_root, "images", "training")
    if os.path.isdir(_tr):
        existing = len(os.listdir(_tr))
    if existing < need:
        build_synthetic_dataset(args.data_root, "training", need)
        build_synthetic_dataset(args.data_root, "validation", 1)
    train_ds = VKITTI(args.data_root, "training", transforms=get_transform(True, crop_size=args.crop_size))
    val_ds = VKITTI(args.data_root, "validation", transforms=get_transform(False, crop_size=args.crop_size))
    print(f"    train={len(train_ds)} 样本, val={len(val_ds)} 样本, batch_size={args.batch_size}")
    if len(train_ds) < args.batch_size:
        print(f"    [WARN] 样本数 {len(train_ds)} < batch_size {args.batch_size}，"
              f"实际 batch 只有 {len(train_ds)}，测得的显存会偏小")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True,
                                               collate_fn=train_ds.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                             num_workers=0, pin_memory=True,
                                             collate_fn=val_ds.collate_fn)

    # ---------------- 模型 ----------------
    print("\n[2/5] 构建 SDCombo 模型 ...")
    t0 = time.perf_counter()
    model = SDCombo(args.num_classes).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"    参数量: {n_param / 1e6:.2f} M   (构建耗时 {time.perf_counter() - t0:.1f}s)")

    # ---------------- 单次前向 ----------------
    print("\n[3/5] 单次前向 ...")
    image, annotation, depth = next(iter(train_loader))
    image, annotation, depth = image.to(device), annotation.to(device), depth.to(device)
    print(f"    image={tuple(image.shape)} annotation={tuple(annotation.shape)} depth={tuple(depth.shape)}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        out = model(image, depth)
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"    output={tuple(out.shape)}  耗时 {time.perf_counter() - t0:.2f}s")

    # ---------------- 训练一步 ----------------
    print("\n[4/5] 训练一步（forward + backward + optimizer.step）...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    lr_scheduler = create_lr_scheduler(optimizer, 10, 1e-5, -1)
    t0 = time.perf_counter()
    mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, 0, lr_scheduler,
                                    record_mark="smoke", print_freq=1, scaler=scaler)
    print(f"    train_loss={mean_loss:.4f}  lr={lr:.2e}  耗时 {time.perf_counter() - t0:.2f}s")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        print(f"    训练峰值显存: allocated={peak:.2f} GiB, reserved={reserved:.2f} GiB")

    # ---------------- 评估 ----------------
    print("\n[5/5] 验证集评估 ...")
    with torch.no_grad():
        confmat = evaluate(model, val_loader, device=device, num_classes=args.num_classes, record_mark="smoke")
    print("    " + str(confmat).replace("\n", "\n    "))

    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else float("nan")
    reserved_gib = torch.cuda.max_memory_reserved() / 1024 ** 3 if device.type == "cuda" else float("nan")
    print("\nSMOKE_SUMMARY " + " ".join([
        f"batch={args.batch_size}",
        f"crop={args.crop_size}",
        f"amp={args.amp}",
        f"params_M={n_param / 1e6:.2f}",
        f"peak_allocated_GiB={peak_gib:.2f}",
        f"peak_reserved_GiB={reserved_gib:.2f}",
        f"train_loss={mean_loss:.4f}",
        f"mIoU={confmat.compute()[2].mean().item() * 100:.1f}",
    ]))

    print("\n" + "=" * 68)
    print("冒烟测试通过：数据集 -> 模型 -> 损失 -> 反向 -> 评估 全链路可用。")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
