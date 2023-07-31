import os
import numpy as np
import cv2

depth_dir = "J:/Dataset/VKITTIS/depth/training"
depth_png_filename = depth_dir + "/rgb_S01_15l_c0_000.png"
depth = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
print(depth)