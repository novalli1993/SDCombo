import os
import time

import numpy as np
from PIL import Image

from_path = "T:/Dataset/VKITTI_II/annotations_RGB"
to_path = "T:/Dataset/VKITTI_II/annotations/"
dir = ["training", "validation"]
classes = {tuple([210, 0, 200]): 0, tuple([90, 200, 255]): 1, tuple([0, 199, 0]): 2, tuple([90, 240, 0]): 3, tuple([140, 140, 140]): 4, tuple([100, 60, 100]): 5, tuple([250, 100, 255]): 6, tuple([255, 255, 0]): 7, tuple([200, 200, 0]): 8, tuple([255, 130, 0]): 9, tuple([80, 80, 80]): 10, tuple([160, 60, 60]): 11, tuple([255, 127, 80]): 12, tuple([0, 139, 139]): 13, tuple([0, 0, 0]): 14}

if os.path.exists(to_path):
    pass
else:
    os.mkdir(to_path)
    for d in dir:
        os.mkdir(to_path + "\\" + d)

for d in dir:
    if os.path.exists(to_path + "\\" + d):
        pass
    else:
        os.mkdir(to_path + "\\" + d)


def transform(img_path):
    img = Image.open(img_path)  # 读取RGB图像
    arr = np.array(img)  # 转换为数组
    for h in range(len(arr)):
        for w in range(len(arr[0])):
            arr[h,w,]=classes[tuple(arr[h][w])]
    arr=arr[:,:,1]
    image = Image.fromarray(arr)  # 转换为图像
    return image


a = 0
time_start = time.perf_counter()
time_end = time.perf_counter()
for d in dir:
    work_dir = from_path + "\\" + d
    # work_dir = from_path + "\\100"
    keep_dir = to_path + "\\" + d
    print(work_dir)
    for _, _, p in os.walk(work_dir):
        for i in p:
            if i.count("S06")!=1 or a<= 14020:
                a += 1
                pass
            else:
                image_dir = work_dir + "\\" + i
                gray = transform(image_dir)
                gray.save(keep_dir + "\\" + i)
                a += 1
                print(i)
                if a % 100 == 0:
                    print(a, end="")
                    print("/42520")
                    time_end = time.perf_counter()
                    print(time_end - time_start)
print("Complete!")

# import numpy as np
# from PIL import Image
# img = Image.open('T:/Dataset/VKITTI_II/annotations_RGB/100/rgb_S02_15l_c0_000.png') # 读取RGB图像
# arr1 = np.array(img) # 转换为数组
# print(arr1.shape)
# for h in range(len(arr1)):
#     for w in range(len(arr1[0])):
#         arr1[h,w,]=classes[tuple(arr1[h][w])]
# arr1 = arr1[:,:,1]
# print(arr1[34,221,])
