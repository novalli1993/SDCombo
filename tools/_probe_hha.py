"""临时探针：验证能否把 VKITTI 的 16bit 原始深度转成 HHA（论文 §5.4/§6.1 的编码）。

只用仓库自带的 Depth2HHA-python 实现，不修改任何仓库代码。
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HHA_DIR = os.path.join(HERE, "..", "Utils", "DataPreparation", "Depth2HHA-python")
sys.path.insert(0, os.path.abspath(HHA_DIR))

from getHHA import getHHA  # noqa: E402

# VKITTI / KITTI 虚拟相机内参（1242x375）。VKITTI 由 Unity 虚拟相机渲染，
# 官方给的 FOV 为 81 度水平 -> fx = (1242/2)/tan(40.5deg) ~= 725。
W, H = 1242, 375
FX = FY = 725.0087
CX, CY = 620.5, 187.0
C = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def load_depth_cm(path, crop=None):
    """读 16bit 深度（单位 cm），可选中心裁剪以加速探针。"""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert d is not None, path
    if crop is not None:
        h, w = d.shape
        y0 = max((h - crop[0]) // 2, 0)
        x0 = max((w - crop[1]) // 2, 0)
        d = d[y0:y0 + crop[0], x0:x0 + crop[1]]
        c = C.copy()
        c[0, 2] -= x0
        c[1, 2] -= y0
        return d.astype(np.float64), c
    return d.astype(np.float64), C


def main():
    root = os.path.join(HERE, "..", "datasets", "VKITTI_II")
    split = "training"
    names = sorted(os.listdir(os.path.join(root, "depth", split)))[:6]

    for crop in [(256, 256), (375, 1242)]:
        print("=" * 70)
        print("crop =", crop)
        for name in names[:3] if crop == (256, 256) else names[:1]:
            path = os.path.join(root, "depth", split, name)
            D, c = load_depth_cm(path, crop=crop)
            missing = (D == 0) | (D >= 65535)          # 0=无效, 65535=远平面裁剪
            RD = D.copy()
            RD[missing] = 0
            t = time.time()
            hha = getHHA(c, D / 100.0, RD / 100.0)     # getHHA 需要"米"
            dt = time.time() - t
            print(" {}  {}  clipped={:.1%}  shape={}  t={:.2f}s".format(
                name, hha.shape, float(missing.mean()), crop, dt))
            for i, ch in enumerate("HHA"):
                a = hha[:, :, i]
                print("    ch{} {}: min={:3d} max={:3d} mean={:6.1f} std={:5.1f} unique={}".format(
                    i, ch, a.min(), a.max(), a.mean(), a.std(), len(np.unique(a))))
            out = os.path.join(HERE, "_probe_hha_{}.png".format(name.split(".")[0]))
            cv2.imwrite(out, hha)
            print("    -> {}".format(out))


if __name__ == "__main__":
    main()
