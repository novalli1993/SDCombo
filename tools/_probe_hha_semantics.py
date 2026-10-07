"""临时探针：核验 VKITTI HHA 三个通道的**物理语义**是否正确。

只做视觉检查不够 —— 需要确认：
  1. 视差(D)通道是否与深度的倒数单调（近亮远暗）；
  2. 高度(H)通道是否随图像行号（越靠下越接近地面）单调减小；
  3. 角度(A)通道在地面区域是否接近"与重力垂直"的预期值。
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "Utils", "DataPreparation", "Depth2HHA-python")))
from getHHA import getHHA  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "datasets", "VKITTI_II"))
W, H = 1242, 375
FX = FY = 725.0087
CX, CY = 620.5, 187.0


def build(C, D, crop=None):
    if crop is not None:
        h, w = D.shape
        y0, x0 = max((h - crop[0]) // 2, 0), max((w - crop[1]) // 2, 0)
        D = D[y0:y0 + crop[0], x0:x0 + crop[1]]
        C = C.copy()
        C[0, 2] -= x0
        C[1, 2] -= y0
    C = C.copy()
    C[1, 2] = D.shape[0] - (H - CY) if crop else CY   # 裁剪后主点随窗口平移
    C[0, 2] = (W - CX) if crop is None else CX
    return D, C


def main():
    C = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    name = sorted(os.listdir(os.path.join(ROOT, "depth", "training")))[0]
    D = cv2.imread(os.path.join(ROOT, "depth", "training", name), cv2.IMREAD_UNCHANGED).astype(np.float64)

    # 用一个**故意简单**的裁剪（左上角 512x375，主点相应左移）避免中心裁剪把道路切掉
    y0, x0, ch, cw = 0, 0, 375, 512
    Dc = D[y0:y0 + ch, x0:x0 + cw]
    Cc = C.copy()
    Cc[0, 2] = CX - x0
    Cc[1, 2] = CY - y0

    missing = (Dc == 0) | (Dc >= 65535)
    RD = Dc.copy()
    RD[missing] = 0
    hha = getHHA(Cc, Dc / 100.0, RD / 100.0)   # 输入单位：米

    disp = hha[:, :, 0].astype(np.float64)     # B 通道 = 视差
    height = hha[:, :, 1].astype(np.float64)   # G 通道 = 高度
    ang = hha[:, :, 2].astype(np.float64)      # R 通道 = 与重力夹角

    valid = ~missing

    def corr(a, b, mask):
        a, b = a[mask], b[mask]
        if a.size < 10:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    rows = np.repeat(np.arange(ch)[:, None], cw, axis=1).astype(np.float64)
    close = valid & (Dc < 1500) & (Dc > 0)      # 15 m 以内，且非远平面

    print("=== 通道物理语义核验 ===")
    print("1) 视差 vs 1/深度（应强正相关，越近视差越大）")
    print("   corr(disp, 1/z)      = %+.3f   (n=%d)" % (corr(disp, 1.0 / np.maximum(Dc, 1), valid), valid.sum()))
    print("   corr(disp, z)        = %+.3f   (应为负)" % corr(disp, Dc, valid))

    print("2) 高度 vs 图像行号（应负相关：越靠下越接近地面 -> 高度越小）")
    print("   corr(height, row)    = %+.3f" % corr(height, rows, close))
    # 取几行看看实际数值
    for r in (ch - 5, ch // 2, 5):
        m = close[r]
        if m.sum() > 5:
            print("     row %3d: height mean=%6.1f  disp mean=%6.1f  (n=%d)" % (
                r, height[r][m].mean(), disp[r][m].mean(), m.sum()))

    print("3) 地面区域的角度应接近 90 度（法向与重力垂直）")
    # 用"最下方 40 行里高度最小的那部分"近似地面
    ground = close.copy()
    ground[:ch - 60] = False
    if ground.sum() > 20:
        print("   ground ang: mean=%6.1f  median=%6.1f  std=%5.1f" % (
            ang[ground].mean(), np.median(ang[ground]), ang[ground].std()))
    print("   NOTE: getHHA 存的是 (angle + 128 - 90)，故 90 度对应像素值 128")

    print("\n4) 各通道取值范围与唯一值数")
    for nm, ch_arr in (("disp", disp), ("height", height), ("angle", ang)):
        print("   %-6s min=%3.0f max=%3.0f mean=%6.1f  unique=%d" % (
            nm, ch_arr.min(), ch_arr.max(), ch_arr.mean(), len(np.unique(ch_arr))))

    print("\n5) 远平面 (=65535) 像素占比: %.1f%%" % (missing.mean() * 100))

    # 保存三通道分别可视化（对比度拉伸）
    os.makedirs(os.path.join(HERE, "_hha_vis"), exist_ok=True)

    def stretch(a):
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        return np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    rgb = cv2.imread(os.path.join(ROOT, "images", "training", name.replace(".png", ".jpg")), cv2.IMREAD_UNCHANGED)
    rgb = rgb[y0:y0 + ch, x0:x0 + cw]
    panel = np.concatenate([rgb, cv2.cvtColor(hha, cv2.COLOR_BGR2RGB),
                            cv2.cvtColor(np.stack([stretch(disp)] * 3, -1), cv2.COLOR_BGR2RGB),
                            cv2.cvtColor(np.stack([stretch(height)] * 3, -1), cv2.COLOR_BGR2RGB),
                            cv2.cvtColor(np.stack([stretch(ang)] * 3, -1), cv2.COLOR_BGR2RGB)], axis=0)
    out = os.path.join(HERE, "_hha_vis", "channels.png")
    cv2.imwrite(out, panel)
    print("\n可视化已保存: %s" % out)
    print("顺序: RGB / HHA(BGR->RGB) / 视差 / 高度 / 角度")


if __name__ == "__main__":
    main()
