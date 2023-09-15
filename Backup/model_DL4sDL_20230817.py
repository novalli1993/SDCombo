import torch
import torch.nn as nn

from Segmentation.DeepLabV3 import deeplabv3_mobilenetv3_large


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        width = int(out_channel / self.expansion)

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=width * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(width * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class Fusion(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.in_conv_rgb = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel / 2), kernel_size=1, bias=True),
            nn.BatchNorm2d(int(in_channel / 2))
        )

        self.in_conv_depth = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel / 2), kernel_size=1, bias=True),
            nn.BatchNorm2d(int(in_channel / 2))
        )

        self.mix_conv = nn.Sequential(Bottleneck(in_channel, in_channel))

        self.resize = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channel * 2)
        )

        self.resize_m = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channel)
        )

    def forward(self, rgb, depth):
        fusion = torch.cat((self.in_conv_rgb(rgb), self.in_conv_depth(depth)), dim=1)
        fusion = self.mix_conv(fusion)

        return self.resize_m(fusion), self.resize(fusion + rgb), self.resize(fusion + depth)


class SDCombo(nn.Module):
    def __init__(self, class_num=14, mark=None):
        super().__init__()
        block = [3, 3, 5, 2]
        channels = [64, 128, 256, 512]

        self.deeplab_rgb = deeplabv3_mobilenetv3_large(aux=None, num_classes=64)

        self.deeplab_depth = deeplabv3_mobilenetv3_large(aux=None, num_classes=64)

        self.in_conv_depth = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=False)
        )

        res_block = []
        self.res_depth = nn.ModuleList()

        for b in range(block[0]):
            res_block.append(BasicBlock(channels[0], channels[0]))
        self.res_depth.append(nn.Sequential(*res_block))
        res_block.clear()

        fusion = [Fusion(channels[0])]
        for i in range(len(channels) - 1):
            downsample = nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=1),
                nn.BatchNorm2d(channels[i + 1])
            )
            for b in range(block[i + 1]):
                res_block.append(BasicBlock(channels[i + 1], channels[i + 1]))
            self.res_depth.append(nn.Sequential(Bottleneck(channels[i], channels[i + 1], downsample=downsample),
                                                nn.Sequential(*res_block)))
            res_block.clear()

            fusion.append(Fusion(channels[i + 1]))

        self.fusion = nn.ModuleList(fusion)

        self.in_conv = []
        for i in range(len(channels) - 1):
            self.in_conv.append(nn.Sequential(
                nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(channels[i])
            ))
        self.in_conv = nn.ModuleList(self.in_conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], class_num, kernel_size=1, bias=True)
        )

    def forward(self, rgb, depth):
        input_size = rgb.size()[-2:]

        depth = nn.functional.normalize(depth.type(torch.float)) * 256
        depth = self.in_conv_depth(depth.unsqueeze(1))

        in_rgb = self.deeplab_rgb(rgb)['out']
        in_depth = self.deeplab_depth(depth)['out']

        mix = []
        for i in range(len(self.fusion)):
            m, in_rgb, in_depth = self.fusion[i](in_rgb, in_depth)
            mix.append(m)

        f = mix[-1]
        for i in reversed(range(len(mix) - 1)):
            conv_x = self.in_conv[i](f)
            conv_x = nn.functional.interpolate(conv_x, mix[i].size()[-2:], mode='bilinear', align_corners=True)
            f = conv_x + mix[i]

        f = nn.functional.interpolate(f, input_size, mode='bilinear', align_corners=True)
        output = self.out_conv(f)
        return output
