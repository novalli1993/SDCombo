import os
import random
import time

import numpy as np
import torch
from PIL import Image
from Utils.DataPreparation.Stanford2D3D.utils import get_index, get_color, load_labels, parse_label

device = 'cuda'
from_path = "J:/Dataset/Stanford2D3D"
to_path = "J:/Dataset/Stanford2D3D/semantic_map"
area = ["area_3"]

label_file = "semantic_labels.json"
instance_class = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door',
                  'floor', 'sofa', 'table', 'wall', 'window']
labels = load_labels(label_file)

class_index = []
for label in labels:
    class_index.append(instance_class.index(parse_label(label)['instance_class']))
class_index = torch.from_numpy(np.array(class_index)).to(device)
# class_count = {}
# for i in range(len(instance_class)):
#     class_count[instance_class[i]] = torch.sum(class_index[:] == i).item()
# print(class_count)

example = 10
error = []

# image map
# start = time.time()
# print("Start of iteration.")
# image_num = 0
# for a in area:
#     a_path = os.path.join(to_path, a)
#     s_path = os.path.join(a_path, 'semantic')
#     images_path = os.path.join(from_path, a, 'semantic')
#     file_count = len(os.listdir(images_path))
#     print("Working path: ", images_path)
#     for _, _, p in os.walk(images_path):
#         for f in p:
#             anno = Image.open(os.path.join(images_path, f))
#             anno = torch.from_numpy(np.array(anno).astype(np.int32)).to(device)
#             map = Image.open(os.path.join(s_path, f))
#             map = torch.from_numpy(np.array(map).astype(np.int32)).to(device)
#             rows, cols, _ = anno.shape
#             index = [(random.randint(0, rows - 1), random.randint(0, cols - 1)) for i in range(example)]
#             anno_index = [get_index(c) for c in [c for c in [anno[x, y,] for (x, y) in index]]]
#             for i in range(len(anno_index)):
#                 if anno_index[i] > len(class_index):
#                     anno_index[i] = 0
#             anno_class = [class_index[i].item() for i in anno_index]
#             map_class = [cls.item() for cls in [map[x, y,] for (x, y) in index]]
#             eva = sum([a - m for a,m in [[anno_class[i], map_class[i]] for i in range(example)]])
#             if eva != 0:
#                 error.append(f)
#             image_num += 1
#             if image_num % 50 == 0:
#                 used_time = time.time() - start
#                 m, s = divmod(used_time, 60)
#                 h, m = divmod(m, 60)
#                 used_time = "%d:%02d:%02d" % (h, m, s)
#                 print("Finished:", str(image_num) + "/" + str(file_count) + ", used time:", used_time)
#                 print("Error images: "+str(len(error))+'.')
#                 if len(error) != 0:
#                     print(error)
# print("Complete!")
# print("Total time: "+str(time.time() - start)+'.')
# print("Total images: "+str(image_num)+'.')
# print("Error images: "+str(len(error))+'.')
# if len(error) != 0:
#     print(error)