import time

import torch
import torch.nn as nn
from torch.nn import functional as F


class SDCHead(nn.Module):
    def __init__(self, in_classes, class_num):
        super().__init__()

        self.in_conv = []
        for i in range(len(in_classes)-1):
            self.in_conv.append(nn.Sequential(
                nn.Conv2d(in_classes[i+1], in_classes[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(in_classes[i])
            ))
        self.in_conv = nn.ModuleList(self.in_conv)

        self.org_mix = nn.Sequential(
            nn.Conv2d(in_classes[0]+1+in_classes[0],in_classes[0], kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_classes[0], in_classes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(in_classes[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_classes[0], class_num, kernel_size=1, bias=True)
        )

    def forward(self, mix, input_shape):
        f = mix[-1]
        # f_list = []
        depth = F.interpolate(mix[0], mix[2].size()[-2:], mode='bilinear', align_corners=False)
        seg_0 = F.interpolate(mix[1], mix[2].size()[-2:], mode='bilinear', align_corners=False)
        mix = mix[2:]
        for i in reversed(range(len(mix) - 1)):
            conv_x = self.in_conv[i](f)
            conv_x = F.interpolate(conv_x, mix[i].size()[-2:], mode='bilinear', align_corners=False)
            f = conv_x + mix[i]
            # f_list.append(f)

        f = torch.cat((f, depth, seg_0),dim=1)
        f = self.org_mix(f)
        x = self.out_conv(f)

        return x
