import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, [0, 0, padw, padh], fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, annotation, depth):
        for t in self.transforms:
            image, annotation, depth = t(image, annotation, depth)
        return image, annotation, depth


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, annotation, depth):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        annotation = F.resize(annotation, size, interpolation=T.InterpolationMode.NEAREST)
        depth = F.resize(depth, size, interpolation=T.InterpolationMode.NEAREST)
        return image, annotation, depth


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, annotation, depth):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            annotation = F.hflip(annotation)
            depth = F.hflip(depth)
        return image, annotation, depth


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, annotation, depth):
        image = pad_if_smaller(image, self.size)
        annotation = pad_if_smaller(annotation, self.size, fill=0)
        depth = pad_if_smaller(depth, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        annotation = F.crop(annotation, *crop_params)
        depth = F.crop(depth, *crop_params)
        return image, annotation, depth


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, annotation, depth):
        image = F.center_crop(image, self.size)
        annotation = F.center_crop(annotation, self.size)
        depth = F.center_crop(depth, self.size)
        return image, annotation, depth


class ToTensor(object):
    def __call__(self, image, annotation, depth):
        image = F.to_tensor(image)
        annotation = torch.as_tensor(np.array(annotation), dtype=torch.int64)
        depth = torch.as_tensor(np.array(depth), dtype=torch.int64)
        return image, annotation, depth


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, annotation, depth):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, annotation, depth
