import os
import datetime
from torch.utils.tensorboard import SummaryWriter

import Dataset.transforms as T
from Dataset.dataset_HHA import Stanford2D3D
from Utils.train_val import *
from Joint.model_DL4sDL import _SDCombo as SDCombo


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


def get_transform(train, base_size=1080, crop_size=256):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return PipelineTrain(base_size, crop_size, mean=mean, std=std) if train \
        else PipelineEval(crop_size, mean=mean, std=std)


class PipelineTrain:
    def __init__(self, base_size, crop_size, mean, std, hflip_prob=0.5):
        min_size = crop_size
        max_size = base_size

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
    def __init__(self, crop_size, mean, std):
        self.transforms = T.Compose([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image, annotation, depth):
        return self.transforms(image, annotation, depth)


def main(args):
    torch.cuda.empty_cache()
    # device, batch size, number of classes and workers
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0])
    # record mark
    if args.test:
        mark = None
    else:
        mark = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # build dataset
    train_dataset = Stanford2D3D(args.data_path, args.fold_num - 1, "training",
                                 transforms=get_transform(True, args.base_size, args.crop_size))
    val_dataset = Stanford2D3D(args.data_path, args.fold_num - 1, "validation",
                               transforms=get_transform(False, args.base_size, args.crop_size))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=max(2**18//(args.crop_size**2), 1),
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # build the model from pretrained and local it on device
    model, missing_keys, unexpected_keys = create_model(args.pretrained, num_classes)
    model.to(device)

    # collect the parameters to be optimized
    if args.pretrained is not None:
        for n, p in model.named_parameters():
            if n in missing_keys:
                p.requires_grad = True
            elif n.split('.')[0] in args.module_freezed and n not in missing_keys:
                p.requires_grad = False
            else:
                p.requires_grad = True

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    print("Parameters unfreezed:")
    print([n for n, p in model.named_parameters() if p.requires_grad])

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
    lr_scheduler = create_lr_scheduler(optimizer, args.cos[0], args.cos[1])

    # resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        _, file = os.path.split(args.resume)
        mark = file[6:21]

    # record file
    if mark is not None:
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

    # start the training
    start_time = datetime.datetime.now()
    if mark is not None:
        writer = SummaryWriter("work_dir/board/" + mark)
    else:
        writer = None
    for epoch in range(args.start_epoch, args.epochs):
        # loss and learning rate
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, writer, lr_scheduler, mark,
                                        args.print_freq, scaler)
        if mark is not None:
            # save the weight
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
            if args.amp:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, "work_dir/model/model_{}_{}.pth".format(mark, epoch))
            # tensorboard of loss and lr
            writer.add_scalar("loss/epoch", mean_loss, epoch)
            writer.add_scalar("lr/epoch", lr, epoch)

        # evaluation
        if args.evaluation:
            torch.cuda.empty_cache()
            confmat = evaluate(model, val_loader, device=device, num_classes=num_classes, record_mark=mark)
            val_info = str(confmat)
            print(val_info)

            if mark is not None:
                # tensorboard of Acc, mIoU and IoUs
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
                                 f"train_loss: {mean_loss:.4f}\n" \
                                 f"lr: {lr:.4e}\n"
                    f.write(train_info + val_info + "\n\n")

    total_time = datetime.datetime.now() - start_time
    if mark is not None:
        with open(results_file, "a") as f: f.write("training time {}".format(total_time) + "\n\n")
    print("training time {}".format(total_time))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="SDCombo training")

    # meta info
    parser.add_argument("--test", default=False, type=bool, help="Test")
    parser.add_argument('--evaluation', default=True, type=bool, help='evaluation')
    parser.add_argument("--message", default="HHA. Focal loss.", type=str, help="Message.")

    # dataset
    parser.add_argument("--data-path", default="J:/Dataset/Stanford2D3D", help="Dataset root")
    parser.add_argument("--fold-num", default=4, type=int,
                        help="Training & Testing allocation:\n"
                             "1: [[1, 2, 3, 4, 6], [5]],\n2: [[1, 3, 5, 6], [2, 4]],\n"
                             "3: [[2, 4, 5], [1, 3, 6]],\n4: [[4], [3]]\n5: [[3], [3]]\n"
                             "Data size: [10327, 15714, 3704, 13268, 17593, 9890]"
                             "1: [52903, 17593], 2: [41514, 28982], 3: [46575, 23921], 4: [13268, 3704], 5: test")
    parser.add_argument("--num-classes", default=14, type=int)
    parser.add_argument("--base_size", default=1080, type=int)
    parser.add_argument("--crop_size", default=256, type=int)
    parser.add_argument("-b", "--batch-size", default=30, type=int)

    # training setting
    parser.add_argument("--aux", default=True, type=bool, help="auxiliary loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--epochs", default=10, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--cos', default=[9, 1e-7], type=list,
                        help='[half-life epoch, minimum learning rate]')
    parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
                        metavar='W', help='weight decay (default: 0.01)',
                        dest='weight_decay')

    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument('--module_freezed',
                        default=[],
                        help='module freezed: internimage')
    parser.add_argument("--pretrained", default="work_dir/model/init_from_DeepLabV3_inside.pth",
                        help="Pretrained weight, "
                             "work_dir/model/init_from_ade20k_internimage.pth,"
                             "work_dir/model/init_from_deeplabv3_mobilenet_v3_large.pth,"
                             "work_dir/model/init_from_DeepLabV3_inside.pth.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("work_dir"):
        os.mkdir("work_dir")
    if not os.path.exists("work_dir/evaluation"):
        os.mkdir("work_dir/evaluation")
    if not os.path.exists("work_dir/logger"):
        os.mkdir("work_dir/logger")
    if not os.path.exists("work_dir/model"):
        os.mkdir("work_dir/model")
    if not os.path.exists("work_dir/board"):
        os.mkdir("work_dir/board")

    main(args)
