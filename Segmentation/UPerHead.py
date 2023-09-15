import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.SyncBatchNorm(out_planes),
        nn.ReLU(inplace=True),
    )


# upernet
class UPerNet(nn.Module):
    def __init__(self, nr_class, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=512):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            # we use the feature map size instead of input image size, so down_scale = 1.0
            # self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.)) # can't compile the PrRoIPooling
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.SyncBatchNorm(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales) * 512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                nn.SyncBatchNorm(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)

        self.nr_class = nr_class

        # input: Fusion out, input_dim: fpn_dim
        self.object_head = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, self.nr_class, kernel_size=1, bias=True)
        )

    def forward(self, conv_out, seg_size=None):

        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        roi = []  # fake rois, just used for pooling
        for i in range(input_size[0]):  # batch size
            roi.append(torch.Tensor([i, 0, 0, input_size[3], input_size[2]]).view(1, -1))  # b, x0, y0, x1, y1
        roi = torch.cat(roi, dim=0).type_as(conv5)
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(F.interpolate(
                # pool_scale(conv5, roi.detach()), # can't compile the PrRoIPooling
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]

        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_fusion(fusion_out)

        output = self.object_head(x)

        if self.use_softmax:  # is True during inference

            x = output
            x = F.interpolate(x, size=seg_size, mode='bilinear', align_corners=False)
            x = F.softmax(x, dim=1)
            output = x

        else:  # Training

            x = output
            x = F.log_softmax(x, dim=1)
            output = x

        return output
