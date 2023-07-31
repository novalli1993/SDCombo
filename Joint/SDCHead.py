import torch
import torch.nn as nn
from torch.nn import functional as F

class SDCHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.out_conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3),
            nn.BatchNorm2d(num_classes)
        )

        self.out_conv_depth_1 = nn.Sequential(
            nn.Conv2d(num_classes + 1, 256, kernel_size=3),
            nn.BatchNorm2d(256)
        )

        self.out_conv_depth_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256)
        )

        self.out_conv_depth_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256)
        )

        self.out_conv_depth_4 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes)
        )

    def forward(self, seg_0, depth):
        seg_0 = self.out_conv(seg_0)
        seg_0 = F.interpolate(seg_0, size=depth.shape[-2:], mode="bilinear")
        depth_limit = torch.max(depth).item()
        depth = depth / depth_limit * torch.max(seg_0).item()
        ma = torch.max(depth)
        depth = depth.unsqueeze(1)

        x = torch.cat((seg_0, depth), dim=1)

        x = F.leaky_relu(self.out_conv_depth_1(x))
        x = F.leaky_relu(self.out_conv_depth_2(x))
        x = F.leaky_relu(self.out_conv_depth_3(x))
        x = F.leaky_relu(self.out_conv_depth_4(x))
        return x
