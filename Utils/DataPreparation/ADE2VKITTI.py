import os
import time

import numpy as np
from PIL import Image

# import numpy as np
# from PIL import Image
# img = Image.open('T:/Dataset/VKITTI_II/annotations_RGB/100/rgb_S02_15l_c0_000.png') # 读取RGB图像
# arr = np.array(img) # 转换为数组
# gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2] # 计算灰度值
# gray = gray.astype(np.uint8) # 转换为无符号整数类型
# image = Image.fromarray(gray) # 转换为灰度图像
# image.show() # 显示灰度图像

from_path = "T:/Dataset/ADEChallengeData2016/annotations/validation"
to_path = "T:/Dataset/ADEtest/annotations/validation"
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
    pic = Image.open(img_path)
    arr = np.array(pic)  # 转换为数组
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == 1: arr[i][j] = 255
            elif arr[i][j] == 2: arr[i][j] = 4
            elif arr[i][j] == 3: arr[i][j] = 1
            elif arr[i][j] == 4: arr[i][j] = 255
            elif arr[i][j] == 5: arr[i][j] = 2
            elif arr[i][j] == 6: arr[i][j] = 255
            elif arr[i][j] == 7: arr[i][j] = 5
            elif arr[i][j] == 8: arr[i][j] = 255
            elif arr[i][j] == 9: arr[i][j] = 255
            elif arr[i][j] == 10: arr[i][j] = 3
            elif arr[i][j] == 11: arr[i][j] = 255
            elif arr[i][j] == 12: arr[i][j] = 0
            elif arr[i][j] == 13: arr[i][j] = 255
            elif arr[i][j] == 14: arr[i][j] = 0
            elif arr[i][j] == 15: arr[i][j] = 255
            elif arr[i][j] == 16: arr[i][j] = 255
            elif arr[i][j] == 17: arr[i][j] = 0
            elif arr[i][j] == 18: arr[i][j] = 3
            elif arr[i][j] == 19: arr[i][j] = 255
            elif arr[i][j] == 20: arr[i][j] = 255
            elif arr[i][j] == 21: arr[i][j] = 12
            elif arr[i][j] == 22: arr[i][j] = 255
            elif arr[i][j] == 23: arr[i][j] = 255
            elif arr[i][j] == 24: arr[i][j] = 255
            elif arr[i][j] == 25: arr[i][j] = 255
            elif arr[i][j] == 26: arr[i][j] = 255
            elif arr[i][j] == 27: arr[i][j] = 255
            elif arr[i][j] == 28: arr[i][j] = 255
            elif arr[i][j] == 29: arr[i][j] = 255
            elif arr[i][j] == 30: arr[i][j] = 0
            elif arr[i][j] == 31: arr[i][j] = 255
            elif arr[i][j] == 32: arr[i][j] = 255
            elif arr[i][j] == 33: arr[i][j] = 255
            elif arr[i][j] == 34: arr[i][j] = 255
            elif arr[i][j] == 35: arr[i][j] = 255
            elif arr[i][j] == 36: arr[i][j] = 255
            elif arr[i][j] == 37: arr[i][j] = 255
            elif arr[i][j] == 38: arr[i][j] = 255
            elif arr[i][j] == 39: arr[i][j] = 255
            elif arr[i][j] == 40: arr[i][j] = 255
            elif arr[i][j] == 41: arr[i][j] = 255
            elif arr[i][j] == 42: arr[i][j] = 255
            elif arr[i][j] == 43: arr[i][j] = 255
            elif arr[i][j] == 44: arr[i][j] = 255
            elif arr[i][j] == 45: arr[i][j] = 255
            elif arr[i][j] == 46: arr[i][j] = 255
            elif arr[i][j] == 47: arr[i][j] = 255
            elif arr[i][j] == 48: arr[i][j] = 255
            elif arr[i][j] == 49: arr[i][j] = 255
            elif arr[i][j] == 50: arr[i][j] = 255
            elif arr[i][j] == 51: arr[i][j] = 255
            elif arr[i][j] == 52: arr[i][j] = 255
            elif arr[i][j] == 53: arr[i][j] = 0
            elif arr[i][j] == 54: arr[i][j] = 255
            elif arr[i][j] == 55: arr[i][j] = 255
            elif arr[i][j] == 56: arr[i][j] = 255
            elif arr[i][j] == 57: arr[i][j] = 255
            elif arr[i][j] == 58: arr[i][j] = 255
            elif arr[i][j] == 59: arr[i][j] = 255
            elif arr[i][j] == 60: arr[i][j] = 255
            elif arr[i][j] == 61: arr[i][j] = 255
            elif arr[i][j] == 62: arr[i][j] = 255
            elif arr[i][j] == 63: arr[i][j] = 255
            elif arr[i][j] == 64: arr[i][j] = 255
            elif arr[i][j] == 65: arr[i][j] = 255
            elif arr[i][j] == 66: arr[i][j] = 255
            elif arr[i][j] == 67: arr[i][j] = 3
            elif arr[i][j] == 68: arr[i][j] = 255
            elif arr[i][j] == 69: arr[i][j] = 255
            elif arr[i][j] == 70: arr[i][j] = 255
            elif arr[i][j] == 71: arr[i][j] = 255
            elif arr[i][j] == 72: arr[i][j] = 255
            elif arr[i][j] == 73: arr[i][j] = 255
            elif arr[i][j] == 74: arr[i][j] = 255
            elif arr[i][j] == 75: arr[i][j] = 255
            elif arr[i][j] == 76: arr[i][j] = 255
            elif arr[i][j] == 77: arr[i][j] = 255
            elif arr[i][j] == 78: arr[i][j] = 255
            elif arr[i][j] == 79: arr[i][j] = 255
            elif arr[i][j] == 80: arr[i][j] = 255
            elif arr[i][j] == 81: arr[i][j] = 255
            elif arr[i][j] == 82: arr[i][j] = 255
            elif arr[i][j] == 83: arr[i][j] = 255
            elif arr[i][j] == 84: arr[i][j] = 11
            elif arr[i][j] == 85: arr[i][j] = 255
            elif arr[i][j] == 86: arr[i][j] = 255
            elif arr[i][j] == 87: arr[i][j] = 255
            elif arr[i][j] == 88: arr[i][j] = 255
            elif arr[i][j] == 89: arr[i][j] = 255
            elif arr[i][j] == 90: arr[i][j] = 255
            elif arr[i][j] == 91: arr[i][j] = 255
            elif arr[i][j] == 92: arr[i][j] = 255
            elif arr[i][j] == 93: arr[i][j] = 255
            elif arr[i][j] == 94: arr[i][j] = 9
            elif arr[i][j] == 95: arr[i][j] = 0
            elif arr[i][j] == 96: arr[i][j] = 255
            elif arr[i][j] == 97: arr[i][j] = 255
            elif arr[i][j] == 98: arr[i][j] = 255
            elif arr[i][j] == 99: arr[i][j] = 255
            elif arr[i][j] == 100: arr[i][j] = 255
            elif arr[i][j] == 101: arr[i][j] = 255
            elif arr[i][j] == 102: arr[i][j] = 255
            elif arr[i][j] == 103: arr[i][j] = 13
            elif arr[i][j] == 104: arr[i][j] = 255
            elif arr[i][j] == 105: arr[i][j] = 255
            elif arr[i][j] == 106: arr[i][j] = 255
            elif arr[i][j] == 107: arr[i][j] = 255
            elif arr[i][j] == 108: arr[i][j] = 255
            elif arr[i][j] == 109: arr[i][j] = 255
            elif arr[i][j] == 110: arr[i][j] = 255
            elif arr[i][j] == 111: arr[i][j] = 255
            elif arr[i][j] == 112: arr[i][j] = 255
            elif arr[i][j] == 113: arr[i][j] = 255
            elif arr[i][j] == 114: arr[i][j] = 255
            elif arr[i][j] == 115: arr[i][j] = 255
            elif arr[i][j] == 116: arr[i][j] = 255
            elif arr[i][j] == 117: arr[i][j] = 255
            elif arr[i][j] == 118: arr[i][j] = 255
            elif arr[i][j] == 119: arr[i][j] = 255
            elif arr[i][j] == 120: arr[i][j] = 255
            elif arr[i][j] == 121: arr[i][j] = 255
            elif arr[i][j] == 122: arr[i][j] = 255
            elif arr[i][j] == 123: arr[i][j] = 255
            elif arr[i][j] == 124: arr[i][j] = 255
            elif arr[i][j] == 125: arr[i][j] = 255
            elif arr[i][j] == 126: arr[i][j] = 255
            elif arr[i][j] == 127: arr[i][j] = 255
            elif arr[i][j] == 128: arr[i][j] = 255
            elif arr[i][j] == 129: arr[i][j] = 255
            elif arr[i][j] == 130: arr[i][j] = 255
            elif arr[i][j] == 131: arr[i][j] = 255
            elif arr[i][j] == 132: arr[i][j] = 255
            elif arr[i][j] == 133: arr[i][j] = 255
            elif arr[i][j] == 134: arr[i][j] = 255
            elif arr[i][j] == 135: arr[i][j] = 255
            elif arr[i][j] == 136: arr[i][j] = 255
            elif arr[i][j] == 137: arr[i][j] = 8
            elif arr[i][j] == 138: arr[i][j] = 255
            elif arr[i][j] == 139: arr[i][j] = 255
            elif arr[i][j] == 140: arr[i][j] = 255
            elif arr[i][j] == 141: arr[i][j] = 255
            elif arr[i][j] == 142: arr[i][j] = 255
            elif arr[i][j] == 143: arr[i][j] = 255
            elif arr[i][j] == 144: arr[i][j] = 255
            elif arr[i][j] == 145: arr[i][j] = 255
            elif arr[i][j] == 146: arr[i][j] = 255
            elif arr[i][j] == 147: arr[i][j] = 255
            elif arr[i][j] == 148: arr[i][j] = 255
            elif arr[i][j] == 149: arr[i][j] = 255
            elif arr[i][j] == 150: arr[i][j] = 255
    image = Image.fromarray(arr)  # 转换为灰度图像
    return image

a = 0
time_start = time.perf_counter()
for d in dir:
    work_dir = from_path
    keep_dir = to_path
    print(work_dir)
    for _, _, p in os.walk(work_dir):
        for i in p:
            image_dir = work_dir + "\\" + i
            gray = transform(image_dir)
            gray.save(keep_dir + "\\" + i)
            a += 1
            if a % 100 == 0:
                print(a, end="")
                print("/2000")
                time_end = time.perf_counter()
                print(time_end - time_start)
print("Complete!")
