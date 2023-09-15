import os

import numpy as np
import torch
from PIL import Image

dataset_path = "T:/Dataset/VKITTIS/images" # mean:  [33.6045 33.9644 27.2941], [19.3824 19.3147 20.1879]
pipeline = ["training", "validation"]

mean = torch.zeros(3)
std = torch.zeros(3)
a = 0

for p in pipeline:
    work_dirs = os.path.join(dataset_path,p)
    for _,_,l in os.walk(work_dirs):
        for n in l:
            img_path = os.path.join(work_dirs, n)
            img = Image.open(img_path)
            arr = np.array(img)
            for d in range(3):
                mean[d] += arr[:, :, d].mean()
                std[d] += arr[:, :, d].std()
                a += 1
print(a)
mean /= a
std /= a

print("mean=[", mean.numpy()[0], ", ", mean.numpy()[1], ", ", mean.numpy()[2], "]")
print("std=[", std.numpy(), ", ", std.numpy()[1], ", ", std.numpy()[2], "]")