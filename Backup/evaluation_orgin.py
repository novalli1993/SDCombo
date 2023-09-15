import torch
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from Dataset.dataset_HHA import Stanford2D3D
import Dataset.transforms as T
from Joint.model_DL4sDL import _SDCombo as SDCombo
from Utils.train_val import evaluate


def create_model(pretrained, num_classes):
    model = SDCombo(aux=False, num_classes=num_classes)
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


def get_transform(train, base_size=1024, crop_size=512):
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
    # device, number of classes and workers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count()])
    # load weight file
    weight = torch.load(args, map_location='cpu')
    train_info = vars(weight["args"])
    num_classes = train_info["num_classes"]
    lr = weight["lr_scheduler"]['_last_lr'][0]
    # build the model from pretrained and local it on device
    model, missing_keys, unexpected_keys = create_model(args, num_classes)
    model.to(device)

    # build dataset
    val_dataset = Stanford2D3D(train_info["data_path"], train_info["fold_num"] - 1, "validation",
                               transforms=get_transform(False, train_info["base_size"], train_info["crop_size"]))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=4,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    # record mark
    mark = os.path.split(args)[1][6:21]
    epoch = int(os.path.splitext(os.path.split(args)[1])[0].split("_")[3])
    # record file
    results_file = "work_dir/evaluation/evaluation{}.txt".format(mark)
    # log meta info
    # train_meta = [i + ": " + str(train_info[i]) + '\n' for i in train_info.keys()]
    # train_meta_str = ""
    # for s in train_meta:
    #     train_meta_str += s
    # train_meta_str += "mark: " + mark + '\n'
    # with (open(results_file, "a") as f):
    #     f.write(train_meta_str + "\n")
    print(train_info)

    writer = SummaryWriter("work_dir/board/" + mark)
    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes, record_mark=mark)
    val_info = str(confmat)

    print(val_info)
    writer.add_scalar("Acc/epoch", float(val_info.split('\n')[2].split(':')[1]), epoch)
    writer.add_scalar("mIoU/epoch", float(val_info.split('\n')[4].split(':')[1]), epoch)
    iou = [float(i) for i in val_info.split('\n')[3].split(':')[1][2:-1].replace("'", '').split(',')]
    cls = val_info.split('\n')[0].split(':')[1][2:-2].replace("'", '').split(',')
    for i in range(len(iou)):
        writer.add_scalar("IoU/" + cls[i], iou[i], epoch)
    # write into txt
    with open(results_file, "a") as f:
        # record: train_loss, lr, val_set corresponding to each epoch
        train_info = f"[epoch: {epoch}]\n" \
                     f"time: {datetime.datetime.now().strftime('%H:%M:%S')}\n" \
                     f"lr: {lr:.4e}\n"
        f.write(train_info + val_info + "\n\n")


if __name__ == '__main__':
    if not os.path.exists("../work_dir"):
        os.mkdir("../work_dir")
    if not os.path.exists("../work_dir/evaluation"):
        os.mkdir("../work_dir/evaluation")

    # Pretrained weight
    start_time = datetime.datetime.now()
    # for i in reversed(range(10)):
    #     weights = "work_dir/model/model_20230823_213627_{}.pth".format(i)
    #     Evalution(weights)
    Evalution("work_dir/model/model_20230826_073826_0.pth")
    total_time = datetime.datetime.now() - start_time
    print("Evaluation time {}".format(total_time))
