import os

import numpy as np
from PIL import Image
import cv2


def get_colour(num):
    B = num % 256
    G = (num >> 8) % 256
    R = (num >> 16) % 256
    return [R, G, B]


def DepthMapPseudoColorize(depth_map, output_path="", max_depth=None, min_depth=None):
    # Load 16 units depth_map(element ranges from 0~65535),
    # Convert it to 8 units (element ranges frome 0~255) and pseudo colorize
    if isinstance(depth_map, np.ndarray):
        uint16_img = depth_map
    # Indicate the argument load mode -1, otherwise loading default 8 units
    if isinstance(depth_map, str):
        uint16_img = cv2.imread(depth_map, -1)
    if max_depth is None:
        max_depth = uint16_img.max()
    if min_depth is None:
        min_depth = uint16_img.min()

    uint16_img -= min_depth
    uint16_img = uint16_img / (max_depth - min_depth)
    uint16_img *= 255

    # cv2.COLORMAP_JET, blue represents a higher depth value, and red represents a lower depth value
    # The value of alpha in the cv.convertScaleAbs() function is related to the effective distance in the depth map. If like me, all the depth values
    # in the default depth map are within the effective distance, and the 16-bit depth has been manually converted to 8-bit depth. , then alpha can be set to 1.
    im_color = cv2.applyColorMap(cv2.convertScaleAbs(uint16_img, alpha=15), cv2.COLORMAP_JET)
    # convert to mat png
    im = Image.fromarray(im_color)
    return im
    # im.save(output_path)


device = 'cuda'
dataset = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
depth_path = os.path.join(dataset, area[5], "depth")
file_list = os.listdir(depth_path)
depth_file = os.path.join(depth_path, file_list[122])

# depth = torch.from_numpy(np.array(Image.open(depth_file))).to(device)
depth = np.array(Image.open(depth_file))

# index = {}
# for i in range(torch.max(depth)):
#     if i > 2000 and torch.sum(depth == i) == 0:
#         break
#     else:
#         index[i] = torch.sum(depth == i)
# head = next(i for i in range(len(index)) if index[i] != 0)
# print(index)
#
# palette = torch.zeros(depth.size()[0], depth.size()[1], 3).type(torch.int).to(device)
#
# for i in range(65535):
#     palette[:, :, 0] += (depth[:, :] == i) * get_colour(i)[0]
#     palette[:, :, 1] += (depth[:, :] == i) * get_colour(i)[1]
#     palette[:, :, 2] += (depth[:, :] == i) * get_colour(i)[2]
#
# palette = (palette.cpu().numpy()).astype(np.uint8)
# print(palette)
# palette = Image.fromarray(palette)
# palette.show()

colored = DepthMapPseudoColorize(depth)
colored.show()