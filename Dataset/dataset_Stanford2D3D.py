import os
import json

import numpy as np
import torch.utils.data as data
from PIL import Image


class Stanford2D3D(data.Dataset):
    def __init__(self, data_path, flod_num, pipeline, transforms=None):
        super().__init__()
        fold = {"training": [[1, 2, 3, 4, 6], [1, 2, 3, 4, 6], [2, 4, 5]],
                "validation": [[5], [2, 4], [1, 3, 6]]}
        area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
        types = ["rgb", "semantic", "depth"]

        for f in fold[pipeline][flod_num]:
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
