import os

class Args(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def parse_args():
    parser = {}

    parser["test"] = False
    parser["meaasge"] = "ResNet block = [3, 3, 5, 2]"

    parser["data_path"] = "/kaggle/input/stanford2d3ds"
    parser["fold_num"] = 1
    # "Training & Testing allocation:\n"
    # "1: [[1, 2, 3, 4, 6], [5]],\n2: [[1, 3, 5, 6], [2, 4]],\n"
    # "3: [[2, 4, 5], [1, 3, 6]],\n4: [[4], [3]]\n5: [[7], [8]]\n"
    # "Data size: [10327, 15714, 3704, 13268, 17593, 9890]"
    # "1: [52903, 17593], 2: [41514, 28982], 3: [46575, 23921], 4: [13268, 3704]")
    parser["num_classes"] = 14
    parser["batch_size"] = 16

    parser["aux"] = True
    parser["device"] = "cuda"
    parser["epochs"] = 10  # number of total epochs to train
    parser["lr"] = 5e-5  # initial learning rate
    parser["cos"] = [9, 1e-7]  # [half-life epoch, minimum learning rate]
    parser["weight_decay"] = 0.05  # weight decay (default: 0.05)

    parser["print_freq"] = 100  # print frequency
    parser["resume"] = ''  # resume from checkpoint
    parser["start_epoch"] = 0  # start epoch

    # Mixed precision training parameters
    parser["amp"] = True  # Use torch.cuda.amp for mixed precision training
    parser["module_trained"] = ['internimage', 'SDHead',
                                'SDBottleneck']  # module to be trained: internimage, upernet, SDHead, SDBottleneck
    parser["pretrained"] = None  # Pretrained weight, work_dir/model/init_from_ade20k_internimage.pth

    args = Args(parser)

    return args

args = parse_args()

args_dict = args.copy()
train_meta = [i + ": " + str(args_dict[i]) + '\n' for i in args_dict.keys()]
train_meta_str = ""
for s in train_meta:
    train_meta_str += s
print(train_meta_str)