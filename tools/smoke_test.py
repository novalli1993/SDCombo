"""训练/推理链路冒烟测试 —— 用合成数据跑通 SDCombo（论文主线模型）全链路。

不需要真实数据集：在 `datasets/VKITTI_II_smoke` 下生成 VKITTI 目录结构的合成数据，
依次验证 数据集 -> 模型(双分支+T融合) -> Focal Loss -> 反向 -> 优化器 -> 评估，
并汇报显存占用，用于在正式训练前确认环境可用以及选 batch size。

用法（仓库根目录，先 conda activate SDCombo）:
    python tools\\smoke_test.py                       # batch=2, crop=256
    python tools\\smoke_test.py --batch-size 16 --amp
    python tools\\smoke_test.py --batch-size 4 --crop-size 384
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import Dataset.transforms as T  # noqa: E402
from Dataset.dataset_VKITTI import VKITTI  # noqa: E402
from Joint.model_DL4sDL import _SDCombo as SDCombo  # noqa: E402
from Utils.train_val_vkitti import (build_class_weights, create_lr_scheduler, evaluate,  # noqa: E402
                                    train_one_epoch)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 14


def build_synthetic_dataset(root, split, count, size=512):
    """生成 VKITTI 目录结构的合成数据：images / annotations / depth / <split>。"""
    paths = {k: os.path.join(root, k, split) for k in ("images", "annotations", "depth")}
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        name = "rgb_S01_15l_c0_{:05d}.png".format(i)
        Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(
            os.path.join(paths["images"], name))
        # 标签 0..13（VKITTI 没有 ignore 类；255 是 collate 的填充值）
        Image.fromarray(rng.integers(0, NUM_CLASSES, (size, size), dtype=np.uint8)).save(
            os.path.join(paths["annotations"], name))
        # 深度：16bit，1 单位 = 1cm，远平面裁剪到 65535（与 VKITTI 真实数据一致）
        dep = (rng.random((size, size)) * 20000).astype(np.uint16)
        Image.fromarray(dep).save(os.path.join(paths["depth"], name))
    return root


def get_transform(train, base_size=375, crop_size=256):
    if train:
        trans = [T.RandomResize(crop_size, max(base_size, crop_size)),
                 T.RandomHorizontalFlip(0.5),
                 T.RandomCrop(crop_size), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    else:
        trans = [T.CenterCrop(crop_size), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    return T.Compose(trans)


def main():
    parser = argparse.ArgumentParser(description="SDCombo (paper main model) 训练链路冒烟测试")
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--crop-size", default=256, type=int)
    parser.add_argument("--num-classes", default=NUM_CLASSES, type=int)
    parser.add_argument("--samples", default=4, type=int, help="合成样本数量")
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "datasets", "VKITTI_II_smoke"))
    parser.add_argument("--amp", action="store_true", help="启用混合精度")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep", action="store_true", help="保留合成数据目录")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device={}  batch_size={}  crop_size={}  amp={}".format(
        device, args.batch_size, args.crop_size, args.amp))

    # ---------------- 数据 ----------------
    print("\n[1/5] 生成/加载合成数据集 ...")
    need = max(args.samples, args.batch_size)   # 样本数必须 >= batch，否则显存会被低估
    tr_dir = os.path.join(args.data_root, "images", "training")
    existing = len(os.listdir(tr_dir)) if os.path.isdir(tr_dir) else 0
    if existing < need:
        build_synthetic_dataset(args.data_root, "training", need)
        build_synthetic_dataset(args.data_root, "validation", 1)
    train_ds = VKITTI(args.data_root, "training", transforms=get_transform(True, crop_size=args.crop_size))
    val_ds = VKITTI(args.data_root, "validation", transforms=get_transform(False, crop_size=args.crop_size))
    print("    train={} 样本, val={} 样本, batch_size={}".format(len(train_ds), len(val_ds), args.batch_size))
    if len(train_ds) < args.batch_size:
        print("    [WARN] 样本数 {} < batch_size {}，测得的显存会偏小".format(len(train_ds), args.batch_size))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True,
                                               collate_fn=VKITTI.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                             num_workers=0, pin_memory=True,
                                             collate_fn=VKITTI.collate_fn)

    # ---------------- 模型 ----------------
    print("\n[2/5] 构建 SDCombo 模型（MobileNetV3 双分支 + 6 级 Fusion + DeepLabV3）...")
    t0 = time.perf_counter()
    model = SDCombo(aux=True, num_classes=args.num_classes).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print("    参数量: {:.2f} M   (构建耗时 {:.1f}s)".format(n_param / 1e6, time.perf_counter() - t0))

    # ---------------- 单次前向 ----------------
    print("\n[3/5] 单次前向 ...")
    image, annotation, depth = next(iter(train_loader))
    image, annotation, depth = image.to(device), annotation.to(device), depth.to(device)
    print("    image={} annotation={} depth={}".format(
        tuple(image.shape), tuple(annotation.shape), tuple(depth.shape)))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        out = model(image, depth)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("    out={}  aux={}  耗时 {:.2f}s".format(
        tuple(out["out"].shape), tuple(out["aux"].shape), time.perf_counter() - t0))

    # ---------------- 训练一步 ----------------
    print("\n[4/5] 训练一步（focal loss + backward + 梯度裁剪 + optimizer.step）...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    lr_scheduler, warm, cos = create_lr_scheduler(optimizer, epochs=10, warmup_epochs=1, lr_min=1e-6)
    # 合成数据的类别直方图是均匀的，直接给一组示例权重，只为跑通权重分支
    class_weights = build_class_weights([1000] * args.num_classes, mode="median",
                                        num_classes=args.num_classes).to(device)
    t0 = time.perf_counter()
    mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, 0, None, lr_scheduler,
                                    "smoke", print_freq=1, scaler=scaler,
                                    class_weights=class_weights, max_grad_norm=1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("    train_loss={:.4f}  lr={:.2e}  耗时 {:.2f}s  (warmup={}ep, cosine_T_max={})".format(
        mean_loss, lr, time.perf_counter() - t0, warm, cos))
    if device.type == "cuda":
        print("    训练峰值显存: allocated={:.2f} GiB, reserved={:.2f} GiB".format(
            torch.cuda.max_memory_allocated() / 1024 ** 3,
            torch.cuda.max_memory_reserved() / 1024 ** 3))

    # ---------------- 评估 ----------------
    print("\n[5/5] 验证集评估 ...")
    confmat = evaluate(model, val_loader, device=device, num_classes=args.num_classes, record_mark="smoke")
    print("    " + str(confmat).replace("\n", "\n    "))

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else float("nan")
    reserved = torch.cuda.max_memory_reserved() / 1024 ** 3 if device.type == "cuda" else float("nan")
    print("\nSMOKE_SUMMARY " + " ".join([
        "batch={}".format(args.batch_size),
        "crop={}".format(args.crop_size),
        "amp={}".format(args.amp),
        "params_M={:.2f}".format(n_param / 1e6),
        "peak_allocated_GiB={:.2f}".format(peak),
        "peak_reserved_GiB={:.2f}".format(reserved),
        "train_loss={:.4f}".format(mean_loss),
    ]))
    print("\n" + "=" * 68)
    print("冒烟测试通过：数据集 -> 模型 -> Focal Loss -> 反向 -> 优化 -> 评估 全链路可用。")
    print("=" * 68)

    # 清理（保留日志文件以便排查；合成数据默认删掉）
    if not args.keep:
        import shutil
        shutil.rmtree(args.data_root, ignore_errors=True)
        log = os.path.join(REPO_ROOT, "work_dir", "logger", "loggerssmoke.txt")
        if os.path.isfile(log):
            os.remove(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
