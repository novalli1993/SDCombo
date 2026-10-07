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
    """图像与深度都转成 float32 张量；标签保持 int64。

    深度必须是 float，理由有二：
      1. 数据集里深度是 16bit 厘米整数，`F.normalize` 只接受浮点张量；
      2. 模型 `SDCombo.forward` 对单通道深度会做 `F.normalize(depth.float())*256`
         再 `depth_conv(1->3)` —— 先转 float 与该流程一致，不改变数值语义。
    """

    def __call__(self, image, annotation, depth):
        image = F.to_tensor(image)
        annotation = torch.as_tensor(np.array(annotation), dtype=torch.int64)
        if depth.mode == 'RGB':
            depth = F.to_tensor(depth)
        else:
            depth = torch.as_tensor(np.array(depth), dtype=torch.float32)
        return image, annotation, depth


class Normalize(object):
    """对 RGB 做 ImageNet 归一化；深度是否归一化由 `depth` 开关决定。

    * `depth=True`（原始深度）：深度是单通道 int64（厘米），照旧做 ImageNet 归一化，
      再由「模型内」的 `F.normalize(depth)*256 -> depth_conv(1->3)` 适配
      —— 与论文 §6.3.1 的代码片段一致。
    * `depth=False`（HHA）：HHA 是 3 通道 uint8 的**几何编码图**，
      三通道分别表示视差/离地高度/与重力夹角，不是自然图像的颜色通道，
      套用 ImageNet 统计量会破坏其几何语义（论文也只说它"格式与 RGB 相同"，
      并未说要用 RGB 的均值方差）。因此 HHA 只做 `ToTensor`（缩放到 [0,1]），
      与 RGB 一样直接作为 3 通道图送进 depth 分支。
    """

    def __init__(self, mean, std, depth=True):
        self.mean = mean
        self.std = std
        self.depth = depth

    def __call__(self, image, annotation, depth):
        image = F.normalize(image, mean=self.mean, std=self.std)
        # 只对 3 通道（HHA，形状 (C,H,W)）做归一化。
        # 单通道原始深度形状是 (H,W)，没有通道维：
        #   * 它不是"图片"，套 ImageNet 统计量没有意义；
        #   * 模型内部会自己做 `F.normalize(depth)*256`（论文 §6.3.1 的写法）。
        # 这里其实用不上 self.depth 开关（形状判断已足够），保留它是为了显式记录意图。
        if self.depth and depth.dim() == 3:
            depth = F.normalize(depth, mean=self.mean, std=self.std)
        return image, annotation, depth
