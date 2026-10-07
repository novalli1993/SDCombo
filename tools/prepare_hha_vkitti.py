"""把 VKITTI 2 的 16bit 原始深度编码成 HHA（论文 §5.4 / §6.1）。

背景
----
论文 §5.4 把深度编码成 HHA（Horizontal disparity / Height above ground /
Angle with gravity），§6.3.1 用「原始深度」作对照组，§6.4.3 给出结论：
**HHA 36.5 mIoU 明显优于原始深度 17.6 mIoU**。

本仓库原有的 HHA 脚本（`Utils/DataPreparation/Depth2HHA-python/trans_HHA2_3.py`）
是给 Stanford2D3D 写的，不适用于 VKITTI。本脚本是 VKITTI 适配版，
复用上游 `getHHA` 的几何实现，只替换数据集相关的三处：

1. **相机内参**：Stanford2D3D 是 1080x1080 室内全景；VKITTI 是 1242x375 车载相机。
   这里用 KITTI 系内参（fx=fy=725.0087, cx=620.5, cy=187.0）。
   该值经核验与 VKITTI 官方给出的 81 度水平视场角自洽
   （fx = (1242/2)/tan(40.5deg) = 725.2）。
2. **无效掩码**：VKITTI 除 `0` 之外，**远平面外的像素被写成 65535**（约占 15%），
   这些像素的深度不可信，必须一并当作缺失，否则会污染法向量与重力估计。
   （上游脚本只把 `RD == 0` 当缺失。）
3. **通道顺序**：上游 `getHHA` 返回的是 OpenCV 的 BGR 数组，
   而 `cv2.imwrite` 再按 BGR 写盘 -> 落盘文件的 R/G/B 恰好是
   **R=disparity, G=height, B=angle**，正是 HHA 的语义顺序。
   本脚本显式做该转换，不依赖隐式的两重反转。

上游 `getHHA` 的一个已核验的**单位契约**（容易踩）：
入口参数 `D/RD` 的单位必须是**米**，函数内部会 `D*100` 转厘米。
按米传参才正确；按厘米传会让地面高度项饱和到常数。

输出
----
```
<out>/HHA/<split>/<stem>.png     3 通道 uint8，R=disparity G=height B=angle
```
与 `images/`、`annotations/`、`depth/` 同级，便于 `--data-path` 直接复用。

用法
----
    python tools/prepare_hha_vkitti.py                      # 全量，自动选进程数
    python tools/prepare_hha_vkitti.py --limit 64           # 小样本自检
    python tools/prepare_hha_vkitti.py --workers 16         # 指定并行度
    python tools/prepare_hha_vkitti.py --overwrite          # 重算已存在文件

特性：**可续跑**（默认跳过已存在的输出，Ctrl-C 后重跑即可续）、多进程、失败不中断。
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Utils", "DataPreparation", "Depth2HHA-python"))

# VKITTI / KITTI 虚拟相机内参（1242x375）
DEFAULT_FX = 725.0087
DEFAULT_FY = 725.0087
DEFAULT_CX = 620.5
DEFAULT_CY = 187.0

FAR_CLIP = 65535          # 远平面裁剪值，视为无效
_CAM = None               # 每个 worker 进程内的缓存


def _patch_scipy_convolve2d():
    """把上游 `utils/util.py` 用的 `scipy.signal.convolve2d` 换成 `fftconvolve`。

    动机（实测剖析，单帧 1242x375 共 3.4 s）：`convolve2d` 独占 **2.5 s（74%）**，
    它在 21x21 的核上走直接卷积、且是单线程，8 个进程也只能吃掉约 1.8 个核。

    为什么可以换：`fftconvolve(mode='same')` 在本库的输入上与 `convolve2d`
    **逐元素一致到 6e-13**（实测 max|diff| = 5.969e-13），远小于 uint8 的量化步长，
    取整后结果**逐位相同**；而速度提升约 20 倍。

    为什么不用 `cv2.filter2D`：它在边界按 BORDER_REFLECT101 处理，
    与 `convolve2d` 的零填充不同，实测边界处 max|diff| 高达 167，
    **会真正改变 HHA 边界像素**，因此不可用。

    上游参考实现（`Utils/DataPreparation/Depth2HHA-python/`）保持原样不动，
    该替换只发生在本脚本的运行期内。
    """
    from scipy import signal
    if getattr(signal, "_dsh_fftconvolve_patch", False):
        return
    _orig = signal.convolve2d

    def convolve2d(in1, in2, mode="full", boundary="fill", fillvalue=0):
        if boundary == "fill" and fillvalue == 0:
            return signal.fftconvolve(in1, in2, mode=mode)
        return _orig(in1, in2, mode=mode, boundary=boundary, fillvalue=fillvalue)

    signal.convolve2d = convolve2d
    signal._dsh_fftconvolve_patch = True


def _build_cam(args):
    return np.array([[args.fx, 0.0, args.cx],
                     [0.0, args.fy, args.cy],
                     [0.0, 0.0, 1.0]])


def hha_from_depth_cm(depth_cm, cam):
    """把单位是厘米的深度图（uint16/float）编码成 HHA。

    Returns:
        (hha_uint8, invalid_ratio)    hha 为 (H,W,3) uint8，通道序 R=disparity G=height B=angle
    """
    import cv2
    _patch_scipy_convolve2d()      # 必须在 getHHA 被导入之前（它内部 from .util import *）
    from getHHA import getHHA

    depth_cm = depth_cm.astype(np.float64)
    invalid = (depth_cm <= 0) | (depth_cm >= FAR_CLIP)

    # 上游约定：入口 depth 单位是米
    D_m = depth_cm / 100.0
    RD_m = depth_cm.copy()
    RD_m[invalid] = 0.0
    RD_m = RD_m / 100.0

    hha_bgr = getHHA(cam, D_m, RD_m)      # 返回 BGR: [0]=angle [1]=height [2]=disparity
    hha_rgb = cv2.cvtColor(hha_bgr, cv2.COLOR_BGR2RGB)   # -> R=disparity G=height B=angle
    return hha_rgb, float(invalid.mean())


def _init_worker(fx, fy, cx, cy):
    global _CAM
    _patch_scipy_convolve2d()
    _CAM = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _process_chunk(job):
    """worker 入口：处理一批文件，返回 (成功数, 跳过数, 失败列表, 无效像素占比之和)。"""
    import cv2

    src_dir, dst_dir, names, overwrite = job
    ok = skipped = 0
    fails = []
    invalid_sum = 0.0
    for name in names:
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(dst_dir, stem + ".png")
        if not overwrite and os.path.exists(out_path):
            skipped += 1
            continue
        try:
            depth = cv2.imread(os.path.join(src_dir, name), cv2.IMREAD_UNCHANGED)
            if depth is None:
                fails.append((name, "unreadable"))
                continue
            hha, inv = hha_from_depth_cm(depth, _CAM)
            cv2.imwrite(out_path, cv2.cvtColor(hha, cv2.COLOR_RGB2BGR))
            ok += 1
            invalid_sum += inv
        except Exception as exc:                      # noqa: BLE001 - 单个失败不该中断全量
            fails.append((name, "{}: {}".format(type(exc).__name__, exc)))
    return ok, skipped, fails, invalid_sum


def collect_names(out_dir, split, limit=0):
    src_dir = os.path.join(out_dir, "depth", split)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError("找不到深度目录: {}".format(src_dir))
    names = sorted(os.listdir(src_dir))
    if limit and limit > 0:
        names = names[:limit]
    return src_dir, names


def main():
    ap = argparse.ArgumentParser(description="VKITTI 2 深度 -> HHA（论文 §5.4）")
    ap.add_argument("--data-path", default=os.path.join("datasets", "VKITTI_II"),
                    help="VKITTI 根目录（含 images/ annotations/ depth/）")
    ap.add_argument("--out", default="", help="输出根目录（默认与 --data-path 相同）")
    ap.add_argument("--splits", nargs="+", default=["training", "validation"])
    ap.add_argument("--workers", type=int, default=0, help="并行进程数（0=自动：CPU 核数-1）")
    ap.add_argument("--chunk", type=int, default=16, help="每个任务的文件数（减少进程间通信）")
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只处理前 N 张（自检用）")
    ap.add_argument("--overwrite", action="store_true", help="重算已存在的输出")
    ap.add_argument("--fx", type=float, default=DEFAULT_FX)
    ap.add_argument("--fy", type=float, default=DEFAULT_FY)
    ap.add_argument("--cx", type=float, default=DEFAULT_CX)
    ap.add_argument("--cy", type=float, default=DEFAULT_CY)
    args = ap.parse_args()

    data_path = os.path.abspath(args.data_path)
    out_root = os.path.abspath(args.out) if args.out else data_path
    # 实测：本机 8 物理核 / 16 逻辑核，8 进程即已饱和（8/12/15 进程吞吐都是 ~3.5 帧/s），
    # 再多只会加剧上下文切换。故默认按**物理核**数取，而不是逻辑核。
    default_workers = max(min((os.cpu_count() or 8) // 2, 8), 1)
    workers = args.workers if args.workers > 0 else default_workers

    print("数据根目录 : {}".format(data_path))
    print("输出根目录 : {}".format(out_root))
    print("内参       : fx={} fy={} cx={} cy={}".format(args.fx, args.fy, args.cx, args.cy))
    print("并行进程   : {}".format(workers))
    print()

    grand_ok = grand_skip = grand_fail = 0
    t_all = time.time()
    for split in args.splits:
        src_dir, names = collect_names(data_path, split, args.limit)
        dst_dir = os.path.join(out_root, "HHA", split)
        os.makedirs(dst_dir, exist_ok=True)

        pending = [n for n in names
                   if args.overwrite or not os.path.exists(
                       os.path.join(dst_dir, os.path.splitext(n)[0] + ".png"))]
        print("== {} : 共 {} 张，待处理 {} 张".format(split, len(names), len(pending)))
        if not pending:
            grand_skip += len(names)
            continue

        chunks = [(src_dir, dst_dir, pending[i:i + args.chunk], args.overwrite)
                  for i in range(0, len(pending), args.chunk)]

        t0 = time.time()
        done = ok = skipped = 0
        fails = []
        invalid_sum = 0.0
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(args.fx, args.fy, args.cx, args.cy)) as pool:
            for c_ok, c_skip, c_fail, c_inv in pool.imap_unordered(_process_chunk, chunks):
                ok += c_ok
                skipped += c_skip
                invalid_sum += c_inv
                fails.extend(c_fail)
                done += len(c_fail) + c_ok + c_skip
                if done % (args.chunk * 20) < args.chunk or done >= len(pending):
                    el = time.time() - t0
                    rate = done / max(el, 1e-6)
                    eta = (len(pending) - done) / max(rate, 1e-6)
                    print("   {}/{}  已写 {}  跳过 {}  失败 {}  {:.1f} 帧/s  ETA {:.0f}s".format(
                        done, len(pending), ok, skipped, len(fails), rate, eta), flush=True)

        el = time.time() - t0
        print("   完成: 写 {} / 跳过 {} / 失败 {}，用时 {:.0f}s ({:.1f} 帧/s)".format(
            ok, skipped, len(fails), el, (ok + skipped) / max(el, 1e-6)))
        if ok:
            print("   平均无效像素占比（深度为 0 或 65535）: {:.1%}".format(invalid_sum / ok))
        if fails:
            print("   [!] 失败样例（最多 5 个）:")
            for n, e in fails[:5]:
                print("       {} -> {}".format(n, e))
        grand_ok += ok
        grand_skip += skipped
        grand_fail += len(fails)

    print()
    print("全部完成: 写 {} / 跳过 {} / 失败 {}，总用时 {:.1f} min".format(
        grand_ok, grand_skip, grand_fail, (time.time() - t_all) / 60.0))
    if grand_fail:
        print("[!] 有失败项；重新运行本脚本会自动只补算缺失文件。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
