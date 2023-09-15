import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# def plt_image():
#     # image_dir = 'T:\\Dataset\\ADEChallengeData2016\\annotations\\training\\ADE_train_00000001.png'
#     image_dir = 'T:\\Dataset\\\VKITTI_II\\annotations\\training\\rgb_S01_15l_c0_000.png'
#     image = Image.open(image_dir)
#     transform = transforms.Grayscale(num_output_channels=1)  # 彩色图像转灰度图像num_output_channels默认1
#     image_gray =transform(image)
#     image_gray = np.array(image_gray)
#     print("image_shape: ", image_gray.shape)
#     print("image_dtype: ", image_gray.dtype)
#     print("image_type: ", type(image_gray))
#     plt.imshow(image_gray)
#     plt.show()
#
#
# plt_image()

# from_path = "T:\\Dataset\\VKITTI_II\\annotations_RGB"
# to_path = "T:\\Dataset\\VKITTI_II\\annotations"
# dir = ["training", "validation"]
#
# if os.path.exists(to_path):
#     pass
# else:
#     os.mkdir(to_path)
#     for d in dir:
#         os.mkdir(to_path + "\\" + d)
#
# for d in dir:
#     if os.path.exists(to_path + "\\" +d):
#         pass
#     else:
#         os.mkdir(to_path + "\\" +d)
#
# # transform RGB to gray. num_output_channels = 1
# transform = transforms.Grayscale(num_output_channels=1)
#
# a = 0
# time_start = time.perf_counter()
# for d in dir:
#     work_dir = from_path + "\\" + d
#     keep_dir = to_path + "\\" + d
#     print(work_dir)
#     for _,_,p in os.walk(work_dir):
#         for i in p:
#             image_dir = work_dir + "\\" + i
#             image = Image.open(image_dir)
#             gray = transform(image)
#             gray.save(keep_dir + "\\" + i)
#             a += 1
#             if a%100==0:
#                 print(a,end="")
#                 print("/42520")
#                 time_end=time.perf_counter()
#                 print(time_end-time_start)
# print("Complete!")
