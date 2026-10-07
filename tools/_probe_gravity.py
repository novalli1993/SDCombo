"""临时探针：诊断 getHHA 在 VKITTI 上的重力方向估计。

上游 getHHA 的假设是「相机坐标系 Y 朝上」并且相机是**倾斜**的（Stanford2D3D 室内手持），
所以它用 normal 场**迭代估计**重力方向。
VKITTI 是车载相机，光轴水平、相机坐标系 Y 轴恰好就是重力方向 [0,1,0]，
属于上游代码里的退化情形（y0 与真实重力平行）。

本探针把三者摆在一起比较：
  (a) 上游 getYDir 估计出来的 yDir；
  (b) 车载相机的解析重力 [0,1,0]；
  (c) 由「地面平面」拟合出来的重力（用最下方像素的平面法向）。
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "Utils", "DataPreparation", "Depth2HHA-python")))
from utils.rgbd_util import computeNormalsSquareSupport, getPointCloudFromZ  # noqa: E402
from utils.util import getYDir  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "datasets", "VKITTI_II"))
W, H = 1242, 375
FX = FY = 725.0087
CX, CY = 620.5, 187.0


def main():
    C = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    names = sorted(os.listdir(os.path.join(ROOT, "depth", "training")))[:3]
    for name in names:
        D = cv2.imread(os.path.join(ROOT, "depth", "training", name), cv2.IMREAD_UNCHANGED).astype(np.float64)
        # 用左上角 375x512，保留道路
        D = D[:375, :512]
        Cc = C.copy()
        Cc[0, 2] = CX
        Cc[1, 2] = CY
        missing = ((D == 0) | (D >= 65535)).astype(np.float64)

        z_cm = D                      # 厘米
        N1, _ = computeNormalsSquareSupport(z_cm / 100.0, missing, 3, 1, Cc, np.ones(D.shape))
        N2, _ = computeNormalsSquareSupport(z_cm / 100.0, missing, 10, 1, Cc, np.ones(D.shape))

        y_est = getYDir(N2, np.array([45, 15]), np.array([5, 5]), np.array([0, 1, 0]))
        y_fix = np.array([[0.0], [1.0], [0.0]])

        print("=== %s ===" % name)
        print("  yDir(上游估计) = [% .3f % .3f % .3f]  |y|=%.3f" % (
            y_est[0], y_est[1], y_est[2], float(np.linalg.norm(y_est))))
        print("  yDir(解析[0,1,0]) = [0 1 0]")
        cos = float(np.abs(np.dot(y_est.flatten(), y_fix.flatten())) /
                    max(np.linalg.norm(y_est), 1e-9))
        print("  两者夹角 = %.1f 度" % np.degrees(np.arccos(np.clip(cos, -1, 1))))

        # 用最下方 30 行（近处地面）拟合平面法向作为"重力"参考
        hh, ww = D.shape
        rows = np.arange(hh - 30, hh)
        cols = np.arange(0, ww, 8)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        z = D[rr, cc]
        keep = (z > 0) & (z < 65535)
        if keep.sum() > 50:
            x3, y3, z3 = getPointCloudFromZ(z, Cc, 1)
            X = np.stack([x3[keep], y3[keep], z3[keep]], axis=1)
            cen = X.mean(0)
            _, _, vt = np.linalg.svd(X - cen)
            n = vt[-1]
            if n[1] > 0:
                n = -n
            print("  地面拟合法向(取 y<0 定向) = [% .3f % .3f % .3f]" % (n[0], n[1], n[2]))
            print("    与 [0,-1,0] 夹角 = %.1f 度" % np.degrees(np.arccos(
                np.clip(np.dot(n, [0, -1, 0]), -1, 1))))
        print()


if __name__ == "__main__":
    main()
