import torch
import torch.nn as nn
from torch.nn import functional as F

from Segmentation.Models.InternImage.intern_image import InternImage
from Segmentation.Models.UPerHead import UPerNet
from .SDCHead import SDCHead


class SDCombo(nn.Module):
    """语义分割 + 深度联合模型：InternImage(DCNv3) 主干 + UPerNet + SDCHead。

    [SDCombo-patch] 相对原始实现的改动：

    * `forward` 现在返回**原始 logits**（形状 (B, num_classes, H, W)，与输入同分辨率）。
      原实现里 UPerHead 先做 log_softmax，再用 `F.interpolate` 双线性上采样 ——
      对已经归一化的概率做插值再送进 cross_entropy（内部又做一次 log_softmax），
      既损失数值性质又压缩梯度。现在归一化统一交给损失函数/推理端。
      需要概率时用 `predict_proba`，需要类别图时用 `predict`。

    * 去掉了对 `UPerNet(seq_out, 256)` 的第二个实参。原 `seg_size=256` 只在
      `use_softmax=True` 分支被使用，而该分支从未启用（默认 False），
      在所有实际路径中都是无效参数，容易误导。

    * 支持 `with_cp` 与 `drop_path_rate` 配置，便于在更大 crop 下用
      梯度检查点换取显存（RTX 2060 时代为了速度关闭了它，现在可按需开启）。

    * 主干输出通道按 `channels` 推导，不再硬编码 fpn_inplanes，
      避免改 channels 时 UPerNet 输入维度不匹配。
    """

    def __init__(self, class_num, channels=64, depths=(4, 4, 18, 4),
                 groups=(4, 8, 16, 32), drop_path_rate=0.2, with_cp=False,
                 sdhead_kwargs=None):
        super().__init__()
        self.class_num = class_num
        self.internimage = InternImage(core_op='DCNv3',
                                       channels=channels,
                                       depths=list(depths),
                                       groups=list(groups),
                                       mlp_ratio=4.,
                                       drop_path_rate=drop_path_rate,
                                       norm_layer='LN',
                                       layer_scale=1.0,
                                       offset_scale=1.0,
                                       post_norm=False,
                                       with_cp=with_cp,
                                       out_indices=(0, 1, 2, 3),
                                       )
        fpn_inplanes = tuple(int(channels * 2 ** i) for i in range(len(depths)))
        self.upernet = UPerNet(nr_class=self.class_num,
                               fc_dim=channels * 2 ** (len(depths) - 1),
                               fpn_inplanes=fpn_inplanes)

        self.SDHead = SDCHead(class_num, **(sdhead_kwargs or {}))

    def forward(self, image, depth):
        """返回原始 logits，形状 (B, num_classes, H, W)，H/W 与输入一致。"""
        input_shape = image.shape[-2:]

        seq_out = self.internimage(image)
        seq_out = self.upernet(seq_out)
        logits = self.SDHead(seq_out, depth)

        # 对 logits 做双线性上采样（保持 logits 语义，不做 softmax）
        if logits.shape[-2:] != input_shape:
            logits = F.interpolate(logits, input_shape, mode='bilinear',
                                   align_corners=False)
        return logits

    @torch.no_grad()
    def predict(self, image, depth):
        """推理便捷接口：返回 (B, H, W) 的类别图。"""
        return self.forward(image, depth).argmax(1)

    @torch.no_grad()
    def predict_proba(self, image, depth):
        """推理便捷接口：返回 (B, num_classes, H, W) 的概率图。"""
        return self.forward(image, depth).softmax(1)
