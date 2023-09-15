import torch
import torch.nn as nn
from torch.nn import functional as F


class SDCBottleneck_NoDepth(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.in_channel = in_channel
        self.depth_feature = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(1),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(1),
        )
        self.mix_in = []
        for i in in_channel:
            self.mix_in.append(nn.Sequential(
                nn.Conv2d(i, i, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(i),
                nn.LeakyReLU(),
                nn.Conv2d(i, i, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(i),
                nn.LeakyReLU(),
                nn.Conv2d(i, i, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(i),
                nn.LeakyReLU(),
                nn.Conv2d(i, i, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(i),
                nn.LeakyReLU()
            ))
        self.mix_in = nn.ModuleList(self.mix_in)

    def forward(self, seg_0, depth):
        mix = []
        for i in range(len(seg_0)):
            mix.append(seg_0[i])

        x = []
        for i in range(len(mix)):
            x.append(self.mix_in[i](mix[i]))

        return x
