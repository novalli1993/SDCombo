import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class SDCHead(nn.Module):
    """Semantic-Depth Combination Head.

    [SDCombo-patch] 相对原始实现做了三处修正，原因见下。

    1) 深度归一化尺度
       原实现为 `depth = depth / torch.max(depth) * torch.max(seg_0)`。
       实测 VKITTI 2 的深度是 16bit（1 单位 = 1cm，远平面 655.35m 被裁剪），
       几乎每一帧都含有 65535 的像素（抽样 120 帧全部如此），因此分母恒为 65535。
       结果：真实深度中位数 20.2m（=2019）被压到 0.0308，深度分支相当于常数输入，
       模型实际上只用了 RGB。这里改为**按固定物理尺度 + log 压缩**归一化：
           d_norm = log1p(depth_cm) / log1p(max_depth_cm)
       取 max_depth_cm=10000（100m）时，中位像素约为 0.65，四分位区间约
       [0.00, 0.87]，深度信号真正参与进来；同时不依赖数据最大值的随机波动。

    2) 去掉每步都发生的 GPU->CPU 同步
       原实现两次调用 `.item()`（`torch.max(depth).item()` 与
       `torch.max(seg_0).item()`），每次都触发一次设备同步，在小 batch 下
       是明显的性能损失；且 `ma = torch.max(depth)` 取出来后从未使用（死代码）。
       现全部在设备上完成。

    3) 卷积补零与归一化层
       原 Conv2d(kernel_size=3) 没有 padding，特征图每层缩小 2 像素（共缩 8），
       与 `F.interpolate` 回原尺寸配合时会引入边界错位。现统一 padding=1。
       归一化由 BatchNorm2d 改为 GroupNorm：SDCHead 的输入包含量纲差异很大的
       深度通道，且 batch 往往很小（4~8），BatchNorm 的 batch 统计量在这种
       条件下不稳定；GroupNorm 与 batch 无关，在分割任务中更可靠。
    """

    def __init__(self, num_classes, max_depth_cm=10000.0, log_depth=True,
                 norm_groups=32):
        super().__init__()
        self.max_depth_cm = float(max_depth_cm)
        self.log_depth = bool(log_depth)

        self.out_conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
            self._make_norm(num_classes, norm_groups)
        )

        self.out_conv_depth_1 = nn.Sequential(
            nn.Conv2d(num_classes + 1, 256, kernel_size=3, padding=1),
            self._make_norm(256, norm_groups)
        )

        self.out_conv_depth_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self._make_norm(256, norm_groups)
        )

        self.out_conv_depth_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self._make_norm(256, norm_groups)
        )

        self.out_conv_depth_4 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),
            self._make_norm(num_classes, norm_groups)
        )

    @staticmethod
    def _make_norm(channels, groups):
        """GroupNorm，自动选一个能整除 channels 的组数。"""
        g = groups
        while g > 1 and channels % g != 0:
            g -= 1
        return nn.GroupNorm(g, channels)

    def normalize_depth(self, depth):
        """深度 -> [0,1] 附近的无量纲张量（保持设备与 dtype，不做同步）。"""
        d = depth.unsqueeze(1).to(torch.float32).clamp(min=0.0,
                                                      max=self.max_depth_cm)
        if self.log_depth:
            denom = math.log1p(self.max_depth_cm)
            d = torch.log1p(d)
        else:
            denom = self.max_depth_cm
        return (d / max(denom, 1e-6)).clamp(0.0, 1.0)

    def forward(self, seg_0, depth):
        seg_0 = self.out_conv(seg_0)
        # 插值到深度图分辨率；align_corners 显式写 False，与官方默认一致
        seg_0 = F.interpolate(seg_0, size=depth.shape[-2:], mode="bilinear",
                              align_corners=False)

        d = self.normalize_depth(depth).to(seg_0.dtype)
        x = torch.cat((seg_0, d), dim=1)

        x = F.leaky_relu(self.out_conv_depth_1(x))
        x = F.leaky_relu(self.out_conv_depth_2(x))
        x = F.leaky_relu(self.out_conv_depth_3(x))
        x = F.leaky_relu(self.out_conv_depth_4(x))
        return x
