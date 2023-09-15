import os
import json

import numpy as np
import torch.utils.data as data
from PIL import Image

CLASSES = ['<UNK>', 'beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter',
           'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']
CLASSES_NUM = {'<UNK>': [0, 0, 0, 0, 0, 1], 'beam': [61, 11, 13, 3, 3, 68], 'board': [27, 17, 12, 10, 42, 29],
               'bookcase': [90, 48, 41, 98, 217, 90], 'ceiling': [55, 81, 37, 73, 76, 63],
               'chair': [155, 545, 67, 159, 258, 179], 'clutter': [775, 495, 335, 671, 922, 684],
               'column': [57, 19, 12, 38, 74, 54], 'door': [86, 93, 37, 107, 127, 93],
               'floor': [44, 50, 23, 50, 68, 49], 'sofa': [6, 6, 9, 14, 11, 9], 'table': [69, 46, 30, 79, 154, 77],
               'wall': [234, 283, 159, 280, 343, 247], 'window': [29, 8, 8, 40, 52, 31]}
PALETTE = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128)]


class Stanford2D3D(data.Dataset):
    def __init__(self, data_path, flod_num, pipeline, transforms=None):
        super().__init__()
        fold = {"training": [[1, 2, 3, 4, 6], [1, 3, 4, 6], [2, 4, 5], [4], [3]],
                "validation": [[5], [2, 4], [1, 3, 6], [3], [3]]}
        area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6", "test", "eval"]
        types = ["rgb", "semantic", "HHA"]

        self.images = []
        self.annotations = []
        self.depth = []
        for f in fold[pipeline][flod_num]:
            for t in types:
                image_dir = os.path.join(data_path, area[f - 1], t)
                assert os.path.exists(image_dir), "path '{}' does not exist.".format(image_dir)
                for _, _, p in os.walk(image_dir):
                    if t == "rgb":
                        self.images.extend([os.path.join(image_dir, x) for x in p])
                    elif t == "semantic":
                        self.annotations.extend([os.path.join(image_dir, x) for x in p])
                    elif t == "HHA":
                        self.depth.extend([os.path.join(image_dir, x) for x in p])
        assert (len(self.images) == len(self.annotations))
        assert (len(self.images) == len(self.depth))

        self.transforms = transforms

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
        # name = os.path.split(self.images[index])[1][:-7]
        # path = os.path.split(os.path.split(self.images[index])[0])[0]
        # image = os.path.join(path, "rgb", name + "rgb.png")
        # annotation = os.path.join(path, "semantic", name + "semantic.png")
        # depth = os.path.join(path, "HHA", name + "depth.png")
        image = os.path.join(self.images[index])
        annotation = os.path.join(self.annotations[index])
        depth = os.path.join(self.depth[index])
        assert os.path.split(image)[1][:-7] == os.path.split(annotation)[1][:-12], \
            "RGB {} doesn't match annotation {}.".format(image, annotation)
        assert os.path.split(image)[1][:-7] == os.path.split(depth)[1][:-7], \
            "RGB {} doesn't match depth {}.".format(image, depth)
        image = Image.open(image)
        annotation = Image.open(annotation)
        depth = Image.open(depth)

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
