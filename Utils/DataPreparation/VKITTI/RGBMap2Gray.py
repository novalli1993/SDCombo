import os
import time

import numpy as np
from PIL import Image
from torchvision import transforms

# import numpy as np
# from PIL import Image
# img = Image.open('T:/Dataset/VKITTI_II/annotations_RGB/100/rgb_S02_15l_c0_000.png') # 读取RGB图像
# arr = np.array(img) # 转换为数组
# gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2] # 计算灰度值
# gray = gray.astype(np.uint8) # 转换为无符号整数类型
# image = Image.fromarray(gray) # 转换为灰度图像
# image.show() # 显示灰度图像

from_path = "T:/Dataset/VKITTI_II/annotations_RGB"
to_path = "T:/Dataset/VKITTI_II/annotations/0"
dir = ["training", "validation"]

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


# transform RGB to gray. num_output_channels = 1
# transform = transforms.Grayscale(num_output_channels=1)
def transform(img_path):
    img = Image.open(img_path)  # 读取RGB图像
    arr = np.array(img)  # 转换为数组
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]  # 计算灰度值
    gray = gray.astype(np.uint8)  # 转换为无符号整数类型
    image = Image.fromarray(gray)  # 转换为灰度图像
    return image


a = 0
time_start = time.perf_counter()
for d in dir:
    work_dir = from_path + "\\" + d
    keep_dir = to_path + "\\" + d
    print(work_dir)
    for _, _, p in os.walk(work_dir):
        for i in p:
            image_dir = work_dir + "\\" + i
            gray = transform(image_dir)
            gray.save(keep_dir + "\\" + i)
            a += 1
            if a % 100 == 0:
                print(a, end="")
                print("/42520")
                time_end = time.perf_counter()
                print(time_end - time_start)
print("Complete!")
