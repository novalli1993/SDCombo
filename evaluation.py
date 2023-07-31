import os
import datetime

import Dataset.transforms as T
from Dataset.dataset_VKITTI import VKITTI
from Joint.model import SDCombo
from Utils.train_val import *

def create_model(pretrained, num_classes):
    model = SDCombo(num_classes)
    missing_keys = unexpected_keys = []
    if pretrained is not None:
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


def get_transform(train):
    base_size = 375
    crop_size = 256
    mean = (33.6045, 33.9644, 27.2941)
    std = (19.3824, 19.3147, 20.1879)
    return PipelineTrain(base_size, crop_size, mean=mean, std=std) if train else PipelineEval(crop_size, mean=mean,
                                                                                              std=std)


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
    def __init__(self, crop_size, mean=(33.6045, 33.9644, 27.2941), std=(19.3824, 19.3147, 20.1879)):
        self.transforms = T.Compose([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)




def main(args):
    # device, batch size, number of classes and workers
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    # record mark
    mark = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # record file
    results_file = "work_dir/evaluation/evaluation{}.txt".format(mark)

    # build dataset
    val_dataset = VKITTI(args.data_path, "validation", transforms=get_transform(train=False))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # build the model from pretrained and local it on device
    pretrained = args.pretrained
    model, missing_keys, unexpected_keys = create_model(pretrained, num_classes)
    model.to(device)

    # log meta info
    with open(results_file, "a") as f:
        train_meta = "Pretrained:" + pretrained
        f.write(train_meta + "\n\n")

    start_time = datetime.datetime.now()
    confmat = evaluate(model, val_loader, device=device, num_classes=num_classes, record_mark=mark)
    val_info = str(confmat)
    print(val_info)
    # write into txt
    with open(results_file, "a") as f:
        # record: train_loss, lr, val_set corresponding to each epoch
        f.write(val_info + "\n\n")

    total_time = datetime.datetime.now() - start_time
    print("training time {}".format(total_time))



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SDCombo evaluation")

    parser.add_argument("--data-path", default="J:/Dataset/VKITTIS", help="Dataset root")
    parser.add_argument("--num-classes", default=15, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--pretrained", default="work_dir/model/model_20230727_230404_9.pth",
                        help="Pretrained weight, best: model_20230727_230404_9.pth")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("work_dir"):
        os.mkdir("work_dir")

    main(args)
