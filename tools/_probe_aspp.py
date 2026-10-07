"""临时探针：ASPP 膨胀率与下采样后的特征图尺寸是否匹配。

论文主线分类头是 DeepLabHead(ASPP([12,24,36]))，主干 output_stride=16。
如果 crop/H 在 stage C5 处只剩 16x16，那么 3x3 空洞卷积（dilation=24/36）的
有效核 49/73 会大于特征图本身 —— 卷积主要在"零填充"上做运算。

本探针直接数一数每个空洞分支的输入里，有多少比例是 padding。
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from Joint.model_DL4sDL import _SDCombo  # noqa: E402


def aspp_padding_fraction(h, w, dilation, k=3):
    """单边 total padding = dilation*(k-1)；返回 padding 元素占(含pad后)元素的比例。"""
    p = dilation * (k - 1)
    H, W = h + 2 * p, w + 2 * p
    inside = h * w
    return 1.0 - inside / float(H * W), p


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = _SDCombo(aux=True, num_classes=14).to(dev).eval()

    print("ASPP 膨胀分支: [12, 24, 36], 卷积核 3x3\n")
    print("{:>12} {:>10} {:>12}   {}".format("crop", "C5 featmap", "out_stride", "各膨胀分支 padding 占比"))
    for crop in (128, 256, 375, 512, 768, 1080):
        x = torch.randn(1, 3, crop, crop, device=dev)
        # backbone 期望 depth 已经是 3 通道（SDCombo.forward 里由 depth_conv 1->3 得到）
        d = torch.randn(1, 3, crop, crop, device=dev)
        with torch.no_grad():
            feats = model.backbone(x, d)
            feat = feats["out"]
        h, w = feat.shape[-2:]
        stride = crop / float(h)
        fracs = []
        for dil in (12, 24, 36):
            f, p = aspp_padding_fraction(h, w, dil)
            fracs.append("{:6.1%}(pad{:>3})".format(f, p))
        print("{:>12} {:>10} {:>12}   {}".format(
            "{}x{}".format(crop, crop), "{}x{}".format(h, w), "{:.0f}".format(stride), "  ".join(fracs)))
        del x, d, feats, feat
        torch.cuda.empty_cache()

    print("\n说明：padding 占比 = 1 - (h*w) / ((h+2p)*(w+2p))，p = dilation*(3-1)。")
    print("占比越高，该分支的卷积越多地在零填充上做运算，等效于丢掉细节/引入边界伪影。")


if __name__ == "__main__":
    main()
