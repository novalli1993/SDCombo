from collections import OrderedDict

from typing import Dict, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from Backbone.MobileNetV3 import mobilenet_v3_large


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, rbg_branch: nn.Module, depth_branch: nn.Module, fusion: nn.Module, stage_indices: [int],
                 return_layers: Dict[str, str]):
        if not set(return_layers).issubset([name for name, _ in rbg_branch.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        self.rgb_layers = OrderedDict()
        for name, module in rbg_branch.named_children():
            self.rgb_layers[name] = module
            layers["rgb_layers_" + name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        return_layers = {str(k): str(v) for k, v in orig_return_layers.items()}
        self.depth_layers = OrderedDict()
        for name, module in depth_branch.named_children():
            self.depth_layers[name] = module
            layers["depth_layers_" + name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        self.fusion_layers = OrderedDict()
        for name, module in fusion.named_children():
            self.fusion_layers[stage_indices[int(name)]] = module
            layers["fusion_layers_" + name] = module

        super().__init__(layers)
        self.return_layers = orig_return_layers
        self.stage_indices = stage_indices

    def forward(self, rgb: Tensor, depth: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module_rbg in self.rgb_layers.items():
            module_depth = self.depth_layers[name]
            rgb = module_rbg(rgb)
            depth = module_depth(depth)
            if int(name) in self.stage_indices:
                x, rgb, depth = self.fusion_layers[int(name)](rgb, depth)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


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
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel // 2)
        )

        self.in_conv_depth = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel // 2)
        )

        self.in_mix_rgb = nn.Sequential(Bottleneck(in_channel, in_channel))

        self.in_mix_depth = nn.Sequential(Bottleneck(in_channel, in_channel))

        self.out_conv_rgb = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel // 2)
        )

        self.out_conv_depth = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channel // 2)
        )

        self.out_mix_rgb = nn.Sequential(Bottleneck(in_channel, in_channel))

        self.out_mix_depth = nn.Sequential(Bottleneck(in_channel, in_channel))

    def forward(self, rgb, depth):
        fusion = torch.cat((self.in_conv_rgb(rgb), self.in_conv_depth(depth)), dim=1)

        fusion_rgb = self.in_mix_rgb(fusion)
        fusion_depth = self.in_mix_depth(fusion)
        fusion_rgb = self.out_mix_rgb(torch.cat((self.out_conv_rgb(fusion_rgb), self.in_conv_depth(depth)), dim=1))
        fusion_depth = self.out_mix_depth(torch.cat((self.out_conv_depth(fusion_depth), self.in_conv_rgb(rgb)), dim=1))

        fusion = fusion_rgb + fusion_depth

        return fusion, fusion + rgb, fusion + depth


class SDCombo(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super().__init__()
        self.depth_conv = nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=False)
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, rgb: Tensor, depth: Tensor) -> Dict[str, Tensor]:
        input_shape = rgb.shape[-2:]
        # contract: features is a dict of tensors
        if len(depth.size()) == 3:
            depth = nn.functional.normalize(depth.type(torch.float)) * 256
            depth = self.depth_conv(depth.unsqueeze(1))
        features = self.backbone(rgb, depth)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        # 使用双线性插值还原回原图尺度
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 使用双线性插值还原回原图尺度
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


def _SDCombo(aux, num_classes=14):
    # 'mobilenetv3_large_imagenet': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
    # 'depv3_mobilenetv3_large_coco': "https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth"
    rbg_branch = mobilenet_v3_large(num_classes=14, dilated=True)
    depth_branch = mobilenet_v3_large(num_classes=14, dilated=True)

    rbg_branch = rbg_branch.features
    depth_branch = depth_branch.features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(rbg_branch) if getattr(b, "is_strided", False)] + [
        len(rbg_branch) - 1]

    # fusion
    fusion = []
    for i in stage_indices:
        in_channel = rbg_branch[i].out_channels
        fusion.append(Fusion(in_channel))
    fusion = nn.ModuleList(fusion)

    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = rbg_branch[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = rbg_branch[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux:
        return_layers[str(aux_pos)] = "aux"

    backbone = IntermediateLayerGetter(rbg_branch, depth_branch, fusion, stage_indices=stage_indices,
                                       return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHead(out_inplanes, num_classes)

    model = SDCombo(backbone, classifier, aux_classifier)

    return model
