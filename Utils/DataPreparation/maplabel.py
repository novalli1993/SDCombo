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
from_path = "J:/Dataset/Stanford2D3D"
to_path = "J:/Dataset/Stanford2D3D/semantic_map"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
label_file = "J:/Model/SDCombo/Dataset/semantic_labels.json"
instance_class = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door',
                  'floor', 'sofa', 'table', 'wall', 'window']
labels = load_labels(label_file)
class_index = []
for label in labels:
    class_index.append(instance_class.index(parse_label(label)['instance_class']))
class_index = torch.from_numpy(np.array(class_index)).to(device)

if os.path.exists(to_path):
    pass
else:
    os.mkdir(to_path)

# image map
image_num = 0
start = time.time()
print("Start of iteration.")
for a in area:
    a_path = os.path.join(to_path, a)
    if os.path.exists(a_path):
        pass
    else:
        os.mkdir(a_path)
    s_path = os.path.join(a_path, 'semantic')
    if os.path.exists(s_path):
        pass
    else:
        os.mkdir(s_path)
    images_path = os.path.join(from_path, a, 'semantic')
    print("Working path: ", images_path)
    for _, _, p in os.walk(images_path):
        for f in p:
            if os.path.exists(os.path.join(s_path, f)):
                pass
            else:
                annotation = Image.open(os.path.join(images_path, f))
                annotation = torch.from_numpy(np.array(annotation).astype(np.int32)).to(device)
                rows, cols, _ = annotation.shape
                annotation_map = torch.zeros((rows, cols), dtype=torch.int32).to(device)
                annotation_index = get_index([annotation[:, :, 0], annotation[:, :, 1], annotation[:, :, 2]])
                for i in range(len(class_index)):
                    annotation_map[:, :] += (((annotation_index <= len(labels)) * annotation_index)[:, :] == i) * class_index[i]
                annotation_map = annotation_map.cpu().numpy().astype(np.int8)
                annotation_map = Image.fromarray(annotation_map)
                annotation_map.save(os.path.join(s_path, f))
            image_num += 1
            if image_num % 50 == 0:
                used_time = time.time() - start
                print("Finished: ", image_num, ", used time: ", used_time)
print("Complete!")
print("Total time: ", time.time() - start, '.')
print("Total images: ", image_num, '.')
