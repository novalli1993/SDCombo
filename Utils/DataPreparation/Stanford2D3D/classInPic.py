import os
import numpy as np
import torch
from PIL import Image

from utils import *

device = 'cuda'
dataset = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]

instance_class = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door',
                  'floor', 'sofa', 'table', 'wall', 'window']
label_file = "semantic_labels.json"
labels = load_labels(label_file)

annotation_path = os.path.join(dataset, area[0], "semantic")
file_list = os.listdir(annotation_path)
for i in range(len(file_list)):
    annotation_file = os.path.join(annotation_path, file_list[i])
    print(annotation_file)
    annotation = torch.from_numpy(np.array(Image.open(annotation_file)).astype(np.int8)).to(device)
    if 7 in annotation:
        print(annotation)
        result = {}
        sum = 0
        for i in range(len(instance_class)):
            result[instance_class[i]] = torch.sum(annotation[:, :] == i).item()
            sum += result[instance_class[i]]
        print(result)
        print(sum)
    else:
        pass
