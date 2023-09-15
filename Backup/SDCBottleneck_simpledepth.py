import torch
import torch.nn as nn
from torch.nn import functional as F


class SDBottleneck(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.in_channel = in_channel
        self.mix_in = []
        for i in in_channel:
            self.mix_in.append(nn.Sequential(
                nn.Conv2d(i + 1, i, kernel_size=3, padding=1, bias=True),
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
        depth_limit = torch.max(depth).item()
        depth = depth / (depth_limit * torch.max(seg_0[0]).item())
        depth = depth.unsqueeze(1)

        mix = []
        for i in range(len(seg_0)):
            mix.append(torch.cat((seg_0[i], F.interpolate(depth, size=seg_0[i].shape[-2:], mode="nearest")), dim=1))

        x = []
        for i in range(len(mix)):
            x.append(self.mix_in[i](mix[i]))

        return x
