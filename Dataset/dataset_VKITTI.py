import os

import torch.utils.data as data
from PIL import Image

# VKITTI 2 的 14 个类别（顺序由 tools/prepare_vkitti.py 的 PALETTE 决定）。
# 注意：0..13 全是真实类别，没有 Stanford2D3D 那种 <UNK>；
#       255 只是 collate_fn 的填充值，所以损失与指标用 ignore_index=255。
CLASSES = ['Terrain', 'Tree', 'Vegetation', 'Building', 'Road', 'GuardRail',
           'TrafficSign', 'TrafficLight', 'Pole', 'Misc', 'Truck', 'Car',
           'Van', 'Undefined']

# 深度表示方式 -> 存放深度图的子目录名。
#   raw   : depth/<split>/xxx.png    16bit 单通道，单位厘米
#   hha   : HHA/<split>/xxx.png      8bit 三通道，R=disparity G=height B=angle
#                                    （由 tools/prepare_hha_vkitti.py 生成）
DEPTH_DIRS = {"raw": "depth", "hha": "HHA"}


class VKITTI(data.Dataset):
    """VKITTI 2 语义分割数据集（RGB + 深度 + 标签三元组）。

    Args:
        voc_root:  数据集根目录（含 images/ annotations/ 以及 depth/ 或 HHA/）
        pipeline:  'training' 或 'validation'
        transforms: 同时作用于 (image, annotation, depth) 的三元变换
        depth_mode: 'raw' -> 读 depth/<pipeline>/ 的单通道 16bit 深度（单位厘米）
                    'hha' -> 读 HHA/<pipeline>/  的三通道 uint8 编码
    """

    def __init__(self, voc_root, pipeline, transforms=None, depth_mode="raw"):
        super(VKITTI, self).__init__()
        if depth_mode not in DEPTH_DIRS:
            raise ValueError("depth_mode 必须是 {} 之一，收到 {!r}".format(
                sorted(DEPTH_DIRS), depth_mode))
        self.depth_mode = depth_mode
        types = ["images", "annotations", DEPTH_DIRS[depth_mode]]
        listing = {}
        for t in types:
            image_dir = os.path.join(voc_root, t, pipeline)
            assert os.path.exists(image_dir), "path '{}' does not exist.".format(image_dir)
            for _, _, p in os.walk(image_dir):
                listing[t] = [os.path.join(image_dir, x) for x in p]
        self.images = listing["images"]
        self.annotations = listing["annotations"]
        self.depth = listing[DEPTH_DIRS[depth_mode]]
        assert (len(self.images) == len(self.annotations))
        assert (len(self.images) == len(self.depth))

        # 独立性校验：三条列表是各自 os.walk 出来的，顺序一致**不是**自动成立的。
        # 这里显式比较文件名（忽略扩展名），不一致就直接报错，避免 RGB/深度错配
        # 这种静默的、会毁掉整个实验的错误。
        stems = {}
        for name, paths in (("images", self.images), ("annotations", self.annotations),
                            (DEPTH_DIRS[depth_mode], self.depth)):
            stems[name] = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        ref = stems["images"]
        for name, s in stems.items():
            if s != ref:
                bad = next((i for i, (a, b) in enumerate(zip(ref, s)) if a != b), None)
                raise RuntimeError(
                    "{} 与 images 的文件名序列不一致（首个不同位置: {}，{!r} vs {!r}）。"
                    "这会导致 RGB 与深度错配，已中止。".format(
                        name, bad, ref[bad] if bad is not None else None,
                        s[bad] if bad is not None else None))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
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
        batched_annotations = cat_list(annotations, fill_value=255)
        # 深度在 crop 之后尺寸已经一致；HHA 是 3 通道 float，raw 是单通道 int64。
        # 两者都按同样方式堆叠即可（cat_list 会先取各维最大值再填充）。
        batched_depth = cat_list(depth, fill_value=0)
        return batched_images, batched_annotations, batched_depth


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
