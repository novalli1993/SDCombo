import os
import shutil
import time
import json

import PIL.Image
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

""" Semantics """


def get_index(color):
    ''' Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    '''
    return color[0] * 256 * 256 + color[1] * 256 + color[2]


def get_color(i):
    ''' Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    '''
    b = (i) % 256  # least significant byte
    g = (i >> 8) % 256
    r = (i >> 16) % 256  # most significant byte
    return r, g, b


""" Label functions """


def load_labels(label_file):
    """ Convenience function for loading JSON labels """
    with open(label_file) as f:
        return json.load(f)


def parse_label(label):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split("_")
    res['instance_class'] = clazz
    res['instance_num'] = int(instance_num)
    res['room_type'] = room_type
    res['room_num'] = int(room_num)
    res['area_num'] = int(area_num)
    return res


device = 'cuda'
label_file = "J:\Model\SDCombo\Dataset\semantic_labels.json"
depth = Image.open(
    "J:/Dataset/Stanford2D3D/semantic_map/area_1/semantic/camera_00d10d86db1e435081a837ced388375f_office_24_frame_19_domain_semantic.png")
instance_class = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door',
                  'floor', 'sofa', 'table', 'wall', 'window']
labels = load_labels(label_file)

# image map
depth = np.array(depth)
print(depth)
print(depth.shape)
rows, cols, _ = depth.shape
print(rows)
print(cols)
depth_map = np.zeros((rows, cols))
print(depth_map.shape)
start = time.time()
class_index = []
# for label in labels:
#     class_index.append(instance_class.index(parse_label(label)['instance_class']))
for i in range(rows):
    for j in range(cols):
        if get_index(depth[i, j]) > len(labels):
            depth_map[i, j] = 0
        else:
            depth[i, j] = instance_class.index(
                parse_label(labels[get_index(depth[i, j])])['instance_class'])
            # depth_map[i, j] = class_index[get_index(depth[i, j]).astype('uint8')]
end = time.time() - start
print(end)
print(depth_map.shape, '\n', depth_map)
depth_map = Image.fromarray(depth_map)
print(depth.size, type(depth))
print(depth_map.size, type(depth))

# labels count
# label_collection = []
# for label in labels:
#     instance_class = parse_label(label)['instance_class']
#     if instance_class in label_collection:
#         pass
#     else:
#         label_collection.append(instance_class)
# print(label_collection)
# print(len(label_collection))
