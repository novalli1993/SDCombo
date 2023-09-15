import torch.nn as nn
from torch.nn import functional as F

from Segmentation.Models.InternImage.intern_image import InternImage
from Backup.SDCBottleneck import SDBottleneckTest
from Backup.SDCHead import SDCHead


class SDCombo(nn.Module):
    def __init__(self, class_num, mark=None):
        super().__init__()
        self.internimage = InternImage(core_op='DCNv3',
                                       channels=64,
                                       depths=[4, 4, 18, 4],
                                       groups=[4, 8, 16,  32],
                                       mlp_ratio=4.,
                                       drop_path_rate=0.2,
                                       norm_layer='LN',
                                       layer_scale=1.0,
                                       offset_scale=1.0,
                                       post_norm=False,
                                       with_cp=False,
                                       out_indices=(0, 1, 2, 3),
                                       )

        # self.SDBottleneck = SDBottleneck(in_channel=[64, 128, 256, 512])
        self.SDBottleneck = SDBottleneckTest(in_channel=[64, 128, 256, 512])
        # self.SDBottleneck = SDCBottleneck_NoDepth(in_channel=[64, 128, 256, 512])

        # self.UPerHead = UPerNet(nr_class=class_num,
        #                                fc_dim=512,
        #                                fpn_inplanes=(64, 128, 256, 512))

        self.SDHead = SDCHead([64, 128, 256, 512], class_num)

    def forward(self, image, depth):
        input_shape = image.shape[-2:]

        seq_out = self.internimage(image)

        seq_out = self.SDBottleneck(seq_out, depth)

        output = self.SDHead(seq_out, input_shape)

        output = F.interpolate(output, input_shape, mode='bilinear', align_corners=False)

        return output
