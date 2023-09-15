import torch
import torch.nn as nn
from torch.nn import functional as F


class SDBottleneck(nn.Module):
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
        depth = F.normalize(depth.type(torch.float)) * 256
        depth = depth.unsqueeze(1)
        depth = self.depth_feature(depth)
        depth = F.adaptive_avg_pool2d(depth, seg_0[0].shape[-2:])

        mix = []
        for i in range(len(seg_0)):
            mix.append(torch.cat((seg_0[i], depth), dim=1))
            depth = self.downsample(depth)

        x = []
        for i in range(len(mix)):
            x.append(self.mix_in[i](mix[i]))

        return x


class SDBottleneckTest(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        rh_kernal = torch.Tensor([[[0, -1, 0], [-1, -5, -1], [0, -1, 0]],
                                  [[0, -1, 0], [-1, -5, -1], [0, -1, 0]],
                                  [[0, -1, 0], [-1, -5, -1], [0, -1, 0]]])
        rh_kernal = rh_kernal.reshape((1, 3, 3, 3))

        by_kernal = torch.Tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                                  [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
                                  [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
                                  ])
        by_kernal = by_kernal.reshape((1, 3, 3, 3))

        self.in_channel = in_channel
        self.depth_feature = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=True),
            # nn.LayerNorm(1)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.LayerNorm(1)
        )

        self.org_seg = []
        for i in in_channel:
            self.org_seg.append(nn.Sequential(
                nn.Conv2d(i, in_channel[0], kernel_size=3, padding=1, bias=True)
            ))
        self.org_seg = nn.ModuleList(self.org_seg)

        self.mix_in = []
        for i in in_channel:
            self.mix_in.append(nn.Sequential(
                nn.Conv2d(i + 1, i, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, i, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, i, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(i),
                nn.Conv2d(i, i, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(i),
            ))
        self.mix_in = nn.ModuleList(self.mix_in)

    def forward(self, seg_0, depth):
        depth = F.normalize(depth.type(torch.float))
        depth = ((depth - torch.min(depth)) * 256).type(torch.int).type(torch.float)
        depth = depth.unsqueeze(1)
        depth = self.depth_feature(depth)

        x = []
        x.append(depth)
        x.append(torch.zeros(seg_0[0].size()).to(seg_0[0].device))

        mix = []
        depth = F.adaptive_max_pool2d(depth, seg_0[0].shape[-2:])
        for i in range(len(seg_0)):
            mix.append(torch.cat((seg_0[i], depth), dim=1))
            x[1] += F.interpolate(self.org_seg[i](seg_0[i]), x[1].size()[-2:], mode='bilinear')
            depth = self.downsample(depth)

        for i in range(len(mix)):
            x.append(self.mix_in[i](mix[i]))

        return x
