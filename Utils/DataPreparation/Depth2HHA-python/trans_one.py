import os
import json
import cv2
import numpy as np

from getHHA import getHHA, getCameraParam

dataset = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
types = ["rgb", "semantic", "depth", "pose", "HHA"]

select = {'area': 4, 'pic_num': 12}
file_name_prefix = os.listdir(os.path.join(dataset, area[select['area']], types[0]))[select['pic_num']][:-7]
file_name_suffix = '.png'
rgb = os.path.join(dataset, area[select['area']], types[0], file_name_prefix + types[0] + file_name_suffix)
semantic = os.path.join(dataset, area[select['area']], types[1], file_name_prefix + types[1] + file_name_suffix)
depth = os.path.join(dataset, area[select['area']], types[2], file_name_prefix + types[2] + file_name_suffix)
with open(os.path.join(dataset, area[select['area']], types[3], file_name_prefix + types[3] + '.json')) as f:
    camera = json.load(f)
# HHA = os.path.join(dataset, area[select['area']], types[4], file_name_prefix + types[4] + file_name_suffix)
HHA = os.path.join("demo", file_name_prefix + types[4] + file_name_suffix)

D = cv2.imread(depth, cv2.COLOR_BGR2GRAY) / 512
# RD = cv2.imread("demo/0_raw.png", cv2.COLOR_BGR2GRAY)/512
camera_matrix = np.array(camera["camera_k_matrix"])
hha = getHHA(camera_matrix, D, RD=D)

print(np.max(hha[2]))

# cv2.imwrite(HHA, hha)