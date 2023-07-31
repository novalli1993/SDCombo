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
annotation = Image.open(
    "J:/Dataset/Stanford2D3D/semantic_map/area_5/semantic/camera_0b2396756adf4a76bbb985af92f534af_office_7_frame_4_domain_semantic.png")
instance_class = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door',
                  'floor', 'sofa', 'table', 'wall', 'window']
labels = load_labels(label_file)

# annotation map
# annotation = torch.from_numpy(np.array(annotation).astype(np.int32)).to(device)
# print(annotation)
# rows, cols, _ = annotation.shape
# annotation_map = torch.zeros((rows, cols), dtype=torch.int32).to(device)
# print(annotation_map.shape)
# start = time.time()
# class_index = []
# for label in labels:
#     class_index.append(instance_class.index(parse_label(label)['instance_class']))
# class_index = torch.from_numpy(np.array(class_index)).to(device)
# print(class_index)
# print(class_index.shape)
# annotation_index = get_index([annotation[:, :, 0], annotation[:, :, 1], annotation[:, :, 2]])
# for i in range(len(class_index)):
#     annotation_map[:, :] += (((annotation_index <= len(labels)) * annotation_index)[:, :] == i) * class_index[i]
# end = time.time() - start
# print(end)
# print(annotation_map.shape, '\n', annotation_map)
# annotation_map = annotation_map.cpu().numpy().astype(np.int8)
# print(annotation_map.dtype)
# annotation_map = Image.fromarray(annotation_map)
# print(annotation_map)

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

# count annotation_map
annotation = np.array(annotation)
print(annotation)
rows, cols = annotation.shape
class_count = []
for i in range(len(instance_class)):
    class_count.append(sum(sum(annotation[:,:]==i)))
    print(instance_class[i]+':', class_count[i])
print('Total: ', sum(class_count))
print(1080*1080)