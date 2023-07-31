import datetime
import torch.nn as nn
from torch.nn import functional as F

from Segmentation.Models.InternImage.intern_image import InternImage
from Segmentation.Models.UPerHead import UPerNet
from .SDCHead import SDCHead


class SDCombo(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.internimage = InternImage(core_op='DCNv3',
                                       channels=64,
                                       depths=[4, 4, 18, 4],
                                       groups=[4, 8, 16, 32],
                                       mlp_ratio=4.,
                                       drop_path_rate=0.2,
                                       norm_layer='LN',
                                       layer_scale=1.0,
                                       offset_scale=1.0,
                                       post_norm=False,
                                       with_cp=False,
                                       out_indices=(0, 1, 2, 3),
                                       )
        self.upernet = UPerNet(nr_class=self.class_num,
                               fc_dim=512,
                               fpn_inplanes=(64, 128, 256, 512))

        self.SDHead = SDCHead(class_num)

    def forward(self, image, depth):
        input_shape = image.shape[-2:]

        seq_out = self.internimage(image)
        seq_out = self.upernet(seq_out, 256)
        output = self.SDHead(seq_out, depth)

        output = F.interpolate(output, input_shape, mode='bilinear', align_corners=False)

        return output
