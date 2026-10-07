import os
import datetime

from Dataset.dataset_VKITTI import VKITTI
from Utils.train_val import *

# 复用 train.py 的验证管线（确定性中心裁剪）与模型构建，避免两处实现漂移
from train import PipelineEval, PipelineEvalNoCrop, create_model


def get_transform(crop_size=375, full_res=False):
    mean = (33.6045, 33.9644, 27.2941)
    std = (19.3824, 19.3147, 20.1879)
    if full_res:
        return PipelineEvalNoCrop(mean=mean, std=std)
    return PipelineEval(crop_size, mean=mean, std=std)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes
    num_workers = args.num_workers if args.num_workers >= 0 else min(
        os.cpu_count() or 1, max(batch_size, 1) * 2, 16)

    mark = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = "work_dir/evaluation/evaluation{}.txt".format(mark)

    val_dataset = VKITTI(args.data_path, "validation",
                         transforms=get_transform(args.crop_size, args.full_res))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             persistent_workers=num_workers > 0,
                                             collate_fn=val_dataset.collate_fn)

    pretrained = args.pretrained
    model, missing_keys, unexpected_keys = create_model(pretrained, num_classes)
    model.to(device)
    model.eval()

    with open(results_file, "a") as f:
        f.write(f"Pretrained: {pretrained}\n")
        f.write(f"batch_size={batch_size} crop_size={args.crop_size} "
                f"num_classes={num_classes}\n\n")

    start_time = datetime.datetime.now()
    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes,
                       record_mark=mark)
    val_info = str(confmat)
    print(val_info)
    with open(results_file, "a") as f:
        f.write(val_info + "\n\n")

    total_time = datetime.datetime.now() - start_time
    print("evaluation time {}".format(total_time))

    # 导出逐类指标，便于脚本化比较
    acc_global, acc, iu = confmat.compute()
    names = ["Terrain", "Tree", "Vegetation", "Building", "Road", "GuardRail",
             "TrafficSign", "TrafficLight", "Pole", "Misc", "Truck", "Car",
             "Van", "Undefined", "Unknown"]
    print("\n逐类 IoU:")
    for i in range(num_classes):
        print(f"  {i:>2} {names[i] if i < len(names) else '?':<14} "
              f"{iu[i].item() * 100:>6.1f}")
    print(f"\nmean IoU = {confmat.mean_iou().item() * 100:.2f}")
    return 0


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SDCombo evaluation")
    parser.add_argument("--data-path", default="datasets/VKITTI_II", help="Dataset root")
    parser.add_argument("--num-classes", default=15, type=int)
    parser.add_argument("--device", default="cuda", help="evaluation device")
    parser.add_argument("-b", "--batch-size", default=8, type=int,
                        help="验证集 batch size（原实现硬编码为 1，严重限制吞吐）")
    parser.add_argument("--crop-size", default=375, type=int,
                        help="评估裁剪尺寸。原实现用 256（仅覆盖约 14%% 视野）；"
                             "VKITTI 图高 375，设为 375 可纵向全覆盖")
    parser.add_argument("--full-res", action="store_true",
                        help="不做任何裁剪，评估整幅图（1242x375，覆盖 100%% 像素）")
    parser.add_argument("--num-workers", default=-1, type=int)
    parser.add_argument("--pretrained", default=None,
                        help="Pretrained weight, e.g. work_dir/model/model_xxx_9.pth")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    for _sub in ("work_dir/logger", "work_dir/evaluation", "work_dir/model"):
        os.makedirs(_sub, exist_ok=True)
    main(args)
