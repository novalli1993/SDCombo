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
    # train_dataset = Stanford2D3D(args.data_path, args.fold_num-1, "training", transforms=get_transform(train=True))
    # val_dataset = Stanford2D3D(args.data_path, args.fold_num-1, "validation", transforms=get_transform(train=False))
    train_dataset = VKITTI(args.data_path, "training", transforms=get_transform(train=True))
    val_dataset = VKITTI(args.data_path, "validation", transforms=get_transform(train=False))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # build the model from pretrained and local it on device
    pretrained = args.pretrained
    model, missing_keys, unexpected_keys = create_model(pretrained, num_classes)
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
    print("Parameters unfreeze:")
    for i in [n for n, p in model.named_parameters() if p.requires_grad]:
        print(i)

    # set the optimizer
    # use AdamW
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # update each iteration by CosineAnnealingLR
    lr_scheduler = create_lr_scheduler(optimizer, args.cos[0], args.cos[1], args.cos[2])

    # log meta info
    with open(results_file, "a") as f:
        if pretrained is None:
            train_meta = "Pretrained: None" + '\n' + "Module to be trained: " + str(args.module_trained)
        else:
            train_meta = "Pretrained:" + pretrained + '\n' + "Module to be trained: " + str(args.module_trained)
        f.write(train_meta + "\n\n")

    # start the training
    start_time = datetime.datetime.now()
    for epoch in range(args.start_epoch, args.epochs):
        # loss and learning rate
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, record_mark=mark, print_freq=args.print_freq,
                                        scaler=scaler)

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes, record_mark=mark)
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
    total_time = datetime.datetime.now() - start_time
    print("training time {}".format(total_time))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SDCombo training")

    parser.add_argument("--data-path", default="J:/Dataset/VKITTI_II", help="Dataset root")
    parser.add_argument("--fold-num", default=0, type=int,
                        help="Training & Testing allocation:\n1: [[1, 2, 3, 4, 6], [5]],\n"
                             "2: [[1, 2, 3, 4, 6], [2, 4]],\n3: [[2, 4, 5], [1, 3, 6]]")
    parser.add_argument("--num-classes", default=15, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxiliary loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=10, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--cos', default=[10, 1e-5, -1], type=list,
                        help='[half-life epoch, minimum learning rate, last epoch]')
    parser.add_argument('--wd', '--weight-decay', default=0.05, type=float,
                        metavar='W', help='weight decay (default: 0.05)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--module_trained',
                        default=['internimage', 'upernet', 'SDHead'],
                        help='module to be trained: internimage, upernet, SDHead')
    parser.add_argument("--pretrained", default=None,
                        help="Pretrained weight, best: model_20230729_183311_10.pth")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("work_dir"):
        os.mkdir("work_dir")

    main(args)
