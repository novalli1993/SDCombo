"""在 VKITTI 2 上对训练好的 SDCombo 权重做独立评估（论文主线模型）。

与 evaluation_HHA.py / evaluation_depth.py 同构，区别：
    * 数据集用 `Dataset/dataset_VKITTI.py`（VKITTI 2，14 类，原始深度）
    * 默认 **整图评估**（`--crop_size 0`）：VKITTI 整图 1242x375 显存占用很小，
      不需要论文时代 RTX 2060 6GB 那种折衷；`--crop_size 256` 则回到裁剪评估
    * 权重加载显式 `weights_only=False`（checkpoint 里存了 argparse.Namespace，
      PyTorch 2.6+ 的默认行为会直接反序列化失败）

用法：
    python evaluation_VKITTI.py --weight work_dir/model/model_20261007_xxxxxx_9.pth
    python evaluation_VKITTI.py --weight ... --crop_size 256 --eval-random-crop
"""
import datetime
import os

import torch

import Dataset.transforms as T
from Dataset.dataset_VKITTI import VKITTI
from Joint.model_DL4sDL import _SDCombo as SDCombo
from Utils.train_val_vkitti import evaluate

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def create_model(pretrained, num_classes):
    model = SDCombo(aux=True, num_classes=num_classes)
    weights_dict = torch.load(pretrained, map_location="cpu", weights_only=False)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    for label, keys in (("missing", missing_keys), ("unexpected", unexpected_keys)):
        if keys:
            print("{} keys: {}".format(label, ", ".join(keys[:20])))
    return model


class PipelineEval:
    """crop_size <= 0 时不做裁剪，整图送入（VKITTI 原图 1242x375）。"""

    def __init__(self, crop_size, mean, std, random_crop=False):
        trans = []
        if crop_size and crop_size > 0:
            trans.append(T.RandomCrop(crop_size) if random_crop else T.CenterCrop(crop_size))
        trans.extend([T.ToTensor(), T.Normalize(mean=mean, std=std)])
        self.transforms = T.Compose(trans)

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)


def evaluation(args):
    torch.cuda.empty_cache()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mark = "Full_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    weight = torch.load(args.weight, map_location="cpu", weights_only=False)
    train_info = vars(weight["args"])
    num_classes = train_info["num_classes"]
    model = create_model(args.weight, num_classes).to(device)
    print("权重: {}  (训练时 num_classes={}, epochs={}, lr={})".format(
        os.path.basename(args.weight), num_classes, train_info.get("epochs"), train_info.get("lr")))

    val_dataset = VKITTI(args.data_path, "validation",
                         transforms=PipelineEval(args.crop_size, MEAN, STD, args.eval_random_crop))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True,
                                             collate_fn=VKITTI.collate_fn)

    results_file = "work_dir/evaluation/evaluation{}.txt".format(mark)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write("# evaluation_VKITTI.py\n")
        f.write("weight: {}\n".format(args.weight))
        f.write("data_path: {}\ncrop_size: {}\nbatch_size: {}\nmark: {}\n\n".format(
            args.data_path, args.crop_size, args.batch_size, mark))
    print(vars(args))

    start_time = datetime.datetime.now()
    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes, record_mark=mark)
    val_info = str(confmat)
    print(val_info)
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(val_info + "\n\n")
    print("Evaluation time {}".format(datetime.datetime.now() - start_time))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SDCombo VKITTI 2 evaluation")
    parser.add_argument("--weight", required=True, help="work_dir/model/model_<mark>_<epoch>.pth")
    parser.add_argument("--data-path", default="datasets/VKITTI_II")
    parser.add_argument("--crop_size", default=0, type=int, help="0=整图评估（默认），256=裁剪评估")
    parser.add_argument("--eval-random-crop", action="store_true", help="裁剪时用随机裁剪而非中心裁剪")
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size > 1 and args.crop_size != 0:
        # 裁剪评估时不同图片尺寸一致，可以堆 batch；整图评估尺寸也一样(1242x375)，同样可以
        pass
    return args


if __name__ == "__main__":
    args = parse_args()
    for d in ("work_dir", "work_dir/evaluation", "work_dir/logger"):
        os.makedirs(d, exist_ok=True)
    evaluation(args)
