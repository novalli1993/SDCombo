import os
import json

import numpy as np
import torch.utils.data as data
from PIL import Image


class Stanford2D3D(data.Dataset):
    def __init__(self, data_path, flod_num, pipeline, transforms=None):
        super().__init__()
        fold = [[[1, 2, 3, 4, 6], [5]],
                [[1, 2, 3, 4, 6], [2, 4]],
                [[2, 4, 5], [1, 3, 6]]]
        area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
        types = ["rgb", "semantic", "depth"]
        fold_type = 0
        if pipeline == "training":
            fold_type = 0
        elif pipeline == "validation":
            fold_type = 1

        for f in fold[flod_num][fold_type]:
            for t in types:
                image_dir = os.path.join(data_path, area[f - 1], t)
                assert os.path.exists(image_dir), "path '{}' does not exist.".format(image_dir)
                for _, _, p in os.walk(image_dir):
                    if t == "rgb":
                        self.images = [os.path.join(image_dir, x) for x in p]
                    elif t == "semantic":
                        self.annotations = [os.path.join(image_dir, x) for x in p]
                    elif t == "depth":
                        self.depth = [os.path.join(image_dir, x) for x in p]
        assert (len(self.images) == len(self.annotations))
        assert (len(self.images) == len(self.depth))
        self.transforms = transforms
        self.labels = load_labels()
        self.instance_class = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter', 'column', 'door',
                               'floor', 'sofa', 'table', 'wall', 'window']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, annotation, depth)
            image: RGB
            annotation: semantic
            depth: depth map
        """
        image = Image.open(self.images[index])

        annotation = Image.open(self.annotations[index])
        annotation = np.array(annotation)
        rows, cols, _ = annotation.shape
        annotation_map = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if get_index(annotation[i, j]) > len(self.labels):
                    annotation_map[i, j] = 0
                else:
                    annotation_map[i, j] = self.instance_class.index(parse_label(self.labels[get_index(annotation[i, j])])['instance_class'])
        annotation = Image.fromarray(annotation_map)

        depth = Image.open(self.depth[index])

        if self.transforms is not None:
            image, annotation, depth = self.transforms(image, annotation, depth)

        return image, annotation, depth

    @staticmethod
    def collate_fn(batch):
        images, annotations, depth = list(zip(*batch))
        batched_images = cat_list(images, fill_value=0)
        batched_annotations = cat_list(annotations, fill_value=0)
        batched_depth = cat_list(depth, fill_value=255)
        return batched_images, batched_annotations, batched_depth


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


" Semantics "


def get_index(color):
    """ Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    """
    return color[0] * 256 * 256 + color[1] * 256 + color[2]


def get_color(i):
    """ Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        i: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    """
    b = (i) % 256  # least significant byte
    g = (i >> 8) % 256
    r = (i >> 16) % 256  # most significant byte
    return r, g, b


" Label functions "


def load_labels():
    """ Convenience function for loading JSON labels """
    with open("./Dataset/semantic_labels.json") as f:
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
