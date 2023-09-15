import os
import json
import time

import cv2
import numpy as np

from getHHA import getHHA, getCameraParam

dataset = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
types = ["rgb", "semantic", "depth", "pose", "HHA"]

select = {'area': 2}
file_list = os.listdir(os.path.join(dataset, area[select['area']], types[2]))

save_dir = os.path.join(dataset, area[select['area']], "HHA2")

if os.path.exists(save_dir):
    pass
else:
    os.mkdir(save_dir)

total = 0
start = time.time()
for file in file_list:
    prefix = file[:-9]
    save_path = os.path.join(save_dir, prefix + types[4] + '.png')
    if os.path.exists(save_path):
        pass
    else:
        depth = os.path.join(dataset, area[select['area']], types[2], file)
        with open(os.path.join(dataset, area[select['area']], types[3], prefix + types[3] + '.json')) as f:
            camera = json.load(f)
        camera_matrix = np.array(camera["camera_k_matrix"])
        D = cv2.imread(depth, cv2.COLOR_BGR2GRAY) / 512  # 1 in depth refer 1/512m
        # RD = cv2.imread("demo/0_raw.png", cv2.COLOR_BGR2GRAY)/512
        hha = getHHA(camera_matrix, D, RD=D)
        cv2.imwrite(save_path, hha)
    total += 1
    if total % 100 == 0:
        used_time = time.time() - start
        m, s = divmod(used_time, 60)
        h, m = divmod(m, 60)
        used_time = "%d:%02d:%02d" % (h, m, s)
        print("Finished:", str(total) + "/" + str(len(file_list)) + ", used time:", used_time)
print("Complete!")
print("Total time: " + str(time.time() - start) + '.')
print("Total images: " + str(len(file_list)) + '.')
