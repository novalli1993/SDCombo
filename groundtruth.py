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
    types = ["rgb", "semantic", "HHA", "predict"]

    file_name_prefix = os.listdir(os.path.join(dataset, areas[area], types[0]))[pic_num][:-7]
    file_name_suffix = '.png'
    rgb = os.path.join(dataset, areas[area], types[0], file_name_prefix + types[0] + file_name_suffix)
    semantic = os.path.join(dataset, areas[area], types[1], file_name_prefix + types[1] + file_name_suffix)
    depth = os.path.join(dataset, areas[area], types[2], file_name_prefix + types[2] + file_name_suffix)
    return rgb, depth, semantic, file_name_prefix


def cam_mask(mask, palette):
    rows, cols = mask.shape
    colorized_mask = torch.zeros(rows, cols, 3).type(torch.int).to(device)
    for i in range(len(palette)):
        colorized_mask[:, :, 0] += (mask[:, :] == i) * palette[i][0]
        colorized_mask[:, :, 1] += (mask[:, :] == i) * palette[i][1]
        colorized_mask[:, :, 2] += (mask[:, :] == i) * palette[i][2]
    colorized_mask = (colorized_mask.cpu().numpy()).astype(np.uint8)
    colorized_mask = Image.fromarray(colorized_mask)
    return colorized_mask


if __name__ == '__main__':
    demo_path = "Demo"
    if not os.path.exists(demo_path):
        os.mkdir(demo_path)
    if not os.path.exists(os.path.join(demo_path, "Ground Truth")):
        os.mkdir(os.path.join(demo_path, "Ground Truth"))

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    start_time = datetime.datetime.now()
    areas = [0, 1, 2, 3, 4, 5]

    for area in areas:
        save_path = os.path.join(demo_path, "Ground Truth", "area_" + str(area + 1))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in range(100):
            # input data
            _, _, annotation, prefix = data_select(area, i)
            annotation = torch.from_numpy(np.array(Image.open(annotation)).astype(np.int8)).to(device)
            cam_mask(annotation, PALETTE).save(os.path.join(save_path, prefix + "gt.png"))
    total_time = datetime.datetime.now() - start_time
    print(total_time)
