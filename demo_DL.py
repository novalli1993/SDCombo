import datetime
import math
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PALETTE = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128)]


def data_select(area, pic_num):
    dataset = "J:/Dataset/Stanford2D3D"
    areas = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
    types = ["rgb", "semantic", "depth", "predict"]

    file_name_prefix = os.listdir(os.path.join(dataset, areas[area], types[0]))[pic_num][:-7]
    file_name_suffix = '.png'
    rgb = os.path.join(dataset, areas[area], types[0], file_name_prefix + types[0] + file_name_suffix)
    semantic = os.path.join(dataset, areas[area], types[1], file_name_prefix + types[1] + file_name_suffix)
    depth = os.path.join(dataset, areas[area], types[2], file_name_prefix + types[2] + file_name_suffix)
    return rgb, depth, semantic, file_name_prefix


def create_model(pretrained, num_classes):
    from Segmentation.DeepLabV3 import deeplabv3_mobilenetv3_large
    model = deeplabv3_mobilenetv3_large(aux=False, num_classes=num_classes, pretrain_backbone=pretrained)
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


def cam_mask(mask, palette):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    n = len(palette)
    for c in range(n):
        seg_img[:, :, 0] += ((mask[:, :] == c) * (palette[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c) * (palette[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c) * (palette[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask


def confusion_matrix(label_true, label_pred, n_class):
    result = torch.zeros((n_class, n_class), dtype=torch.int64, device=label_true.device)
    k = (label_true >= 0) & (label_true < n_class)
    inds = n_class * label_true[k].to(torch.int64) + label_pred[k]
    result += torch.bincount(inds, minlength=n_class ** 2).reshape(n_class, n_class)
    return result


def iou(confusion_matrix):
    iu = torch.diag(confusion_matrix) / (
            confusion_matrix.sum(1) + confusion_matrix.sum(0) - torch.diag(confusion_matrix))
    return iu


def miou(iu):
    result = []
    for i in iu:
        if not math.isnan(i):
            result.append(i)
    result = sum(result) / len(result) * 100
    return result


def inference(model, image, depth, device='cpu'):
    image = Image.open(image)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    image = data_transform(image)
    image = torch.unsqueeze(image, dim=0)

    model.eval()
    with torch.no_grad():
        output = model(image.to(device))['out']
        prediction = output.argmax(1).squeeze(0)

    return prediction.to("cpu").numpy().astype(np.uint8)

if __name__ == '__main__':
    demo_path = "Demo"
    if not os.path.exists(demo_path):
        os.mkdir(demo_path)

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    mark = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    start_time = datetime.datetime.now()
    # input weight and data
    weight = "work_dir/model/model_DL_20230831_070701_18.pth"
    save_path = os.path.join(demo_path, os.path.split(weight)[1][6:-4])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # load weight file
    train_info = torch.load(weight, map_location='cpu')
    train_info = vars(train_info["args"])
    num_classes = train_info["num_classes"]
    # build the model from pretrained and local it on device
    # weight = "work_dir/model/DL_model_DL_20230827_185154_8.pth"
    model, missing_keys, unexpected_keys = create_model(weight, num_classes)
    model.to(device)

    areas = [0,1,2,3,4,5]
    for area in areas:
        save_path = os.path.join(demo_path, os.path.split(weight)[1][6:-4], "area_" + str(area + 1))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in range(100):
            # input data
            image, depth, annotation, prefix = data_select(area, i)
            # inference and mask
            infer = inference(model, image, depth, device)
            cam_mask(infer, PALETTE).save(os.path.join(save_path, prefix + "demo.png"))

    total_time = datetime.datetime.now() - start_time
    print(total_time)
