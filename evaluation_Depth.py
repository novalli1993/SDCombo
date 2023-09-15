import torch
import os
import datetime
from Dataset.dataset_Stanford2D3D import Stanford2D3D
import Dataset.transforms as T
from Joint.model_DL4sDL import _SDCombo as SDCombo
from Utils.train_val import evaluate


def create_model(pretrained, num_classes):
    model = SDCombo(aux=True, num_classes=num_classes)
    missing_keys = []
    unexpected_keys = []
    if pretrained:
        weights_dict = torch.load(pretrained, map_location='cpu')['model']
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


def get_transform(train, base_size=1024, crop_size=1080):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return PipelineEval(crop_size, mean=mean, std=std)


class PipelineEval:
    def __init__(self, crop_size, mean, std):
        self.transforms = T.Compose([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)


def Evalution(args):
    torch.cuda.empty_cache()
    # device, number of classes and workers
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), 8])
    # record mark
    mark = "Full_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # load weight file
    weight = torch.load(args.weight, map_location='cpu')
    train_info = vars(weight["args"])
    num_classes = train_info["num_classes"]
    # build the model from pretrained and local it on device
    model, missing_keys, unexpected_keys = create_model(args.weight, num_classes)
    model.to(device)

    # build dataset
    val_dataset = Stanford2D3D(args.data_path, args.fold_num - 1, "validation",
                               transforms=get_transform(False, 1080, args.crop_size))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # record file
    results_file = "work_dir/evaluation/evaluation{}.txt".format(mark)
    # log meta info
    args_dict = vars(args)
    train_meta = [i + ": " + str(args_dict[i]) + '\n' for i in args_dict.keys()]
    train_meta_str = ""
    for s in train_meta:
        train_meta_str += s
    train_meta_str += "mark: " + mark + '\n'
    with (open(results_file, "a") as f):
        f.write(train_meta_str + "\n")
    print(vars(args))

    # start evaluating
    start_time = datetime.datetime.now()
    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes, record_mark=mark)
    val_info = str(confmat)
    print(val_info)
    # write into txt
    with open(results_file, "a") as f:
        f.write(val_info + "\n\n")

    total_time = datetime.datetime.now() - start_time
    print("Evaluation time {}".format(total_time))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SDCombo evaluation")

    parser.add_argument("--weight", default="work_dir/model/model_20230830_211613_9.pth",
                        help="Pretrained weight, work_dir/model/model_20230830_151212_8.pth")
    parser.add_argument("--data-path", default="J:/Dataset/Stanford2D3D", help="Dataset root")
    parser.add_argument("--fold-num", default=1, type=int,
                        help="Training & Testing allocation:\n"
                             "1: [[1, 2, 3, 4, 6], [5]],\n2: [[1, 3, 5, 6], [2, 4]],\n"
                             "3: [[2, 4, 5], [1, 3, 6]],\n4: [[4], [3]]\n5: [[3], [3]]\n"
                             "Data size: [10327, 15714, 3704, 13268, 17593, 9890]"
                             "1: [52903, 17593], 2: [41514, 28982], 3: [46575, 23921], 4: [13268, 3704], 5: test")
    parser.add_argument("--crop_size", default=1080, type=int)
    parser.add_argument("--device", default="cuda", help="training device")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("work_dir"):
        os.mkdir("work_dir")
    if not os.path.exists("work_dir/evaluation"):
        os.mkdir("work_dir/evaluation")

    Evalution(args)
