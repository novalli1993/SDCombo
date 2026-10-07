#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_vkitti.py —— 从 VKITTI 2 原始 tar 包一站式生成 SDCombo 训练/验证数据集。

取代 Utils/DataPreparation 下需要依次手工执行、且含严重缺陷的 5 个脚本
（FileRename.py -> Allocation.py -> 3to1_S0x.py x5 -> SameName.py）。

原始链路的问题（详见 ReadMe_环境配置.md §5.3）：
  * 3to1_S0x.py 用 Python 双重循环逐像素查字典（1242x375 = 46.6 万次/图），
    在 2 万张图上要跑几十小时；本脚本改用 24-bit LUT 向量化，快约 3 个数量级。
  * 3to1_S01/S06/S20.py 用 `if ... or a<= N:` 续跑计数，但由于 os.walk 的
    遍历顺序在多目录间不保证，重复运行会静默丢弃部分样本（S02/S18 无此问题，
    可见是中断后临时加的续跑逻辑被遗留）。
  * Allocation.py 里 rgb/cls/ins 的复制语句全被注释，只有 depth 会真正复制。
  * norm_para.py 的计数器 a 在通道循环内自增，导致 mean/std 被除以 3N 而非 N，
    即 train.py 里那组归一化常数整体偏小 3 倍（详见文档）。

用法:
    # 1) 指定 3 个 tar 与输出目录
    python tools/prepare_vkitti.py ^
        --rgb  D:\\dl\\vkitti_2.0.3_rgb.tar ^
        --depth D:\\dl\\vkitti_2.0.3_depth.tar ^
        --classseg D:\\dl\\vkitti_2.0.3_classSegmentation.tar ^
        --out datasets\\VKITTI_II

    # 2) 只处理一个场景做试跑
    python tools/prepare_vkitti.py --rgb ... --depth ... --classseg ... ^
        --out datasets\\VKITTI_II --scenes S01 --limit 200

输出结构（dataset_VKITTI.py 直接可读）:
    <out>/images/{training,validation}/rgb_<scene>_<cond>_<cam>_<frame>.jpg
    <out>/annotations/{training,validation}/<同名>.png
    <out>/depth/{training,validation}/<同名>.png

依赖: numpy, Pillow
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- 常量映射

# 场景 -> 短名（与 FileRename.py 一致）
SCENE_MAP = {
    "Scene01": "S01", "Scene02": "S02", "Scene06": "S06",
    "Scene18": "S18", "Scene20": "S20",
}
# 天气/视角 -> 短名
COND_MAP = {
    "15-deg-left": "15l", "15-deg-right": "15r",
    "30-deg-left": "30l", "30-deg-right": "30r",
    "clone": "clo", "fog": "fog", "morning": "mor",
    "overcast": "ove", "rain": "rai", "sunset": "sun",
}
CAM_MAP = {"Camera_0": "c0", "Camera_1": "c1"}

# 训练/验证划分。与 Dataset/dataset_Stanford2D3D.py 的 fold 定义保持一致：
#   training   = Scene 1,2,3,4,6 -> 实际存在的是 S01/S02/S06/S18/S20 中的部分
#   validation = 其余
# 但 Dataset/dataset_VKITTI.py 不接 fold 参数，所以这里沿用准备脚本的惯例
# （Allocation.py）：S01/S06/S18/S20 -> training，其余 -> validation。
DEFAULT_TRAIN_SCENES = ["S01", "S06", "S18", "S20"]

# 官方 15 类调色板（与 Utils/DataPreparation/3to1_S0x.py 的 classes 字典一致）
PALETTE = {
    (210, 0, 200): 0,    # Terrain
    (90, 200, 255): 1,   # Tree
    (0, 199, 0): 2,      # Vegetation
    (90, 240, 0): 3,     # Building
    (140, 140, 140): 4,  # Road
    (100, 60, 100): 5,   # GuardRail
    (250, 100, 255): 6,  # TrafficSign
    (255, 255, 0): 7,    # TrafficLight
    (200, 200, 0): 8,    # Pole
    (255, 130, 0): 9,    # Misc
    (80, 80, 80): 10,    # Truck
    (160, 60, 60): 11,   # Car
    (255, 127, 80): 12,  # Van
    (0, 139, 139): 13,   # Undefined
    (0, 0, 0): 14,       # 未标注/背景色
}


def build_lut(palette: dict) -> np.ndarray:
    """构造 2**24 的 24-bit RGB -> 类别号 LUT。

    向量化替代逐像素 `classes[tuple(arr[h][w])]`。未收录的颜色返回 -1。
    """
    lut = np.full(1 << 24, -1, dtype=np.int8)
    keys = np.array([(r << 16) | (g << 8) | b for (r, g, b) in palette], dtype=np.int64)
    vals = np.array(list(palette.values()), dtype=np.int8)
    # 若存在碰撞（不同颜色打包后同键）会在 assert 中暴露
    assert len(np.unique(keys)) == len(keys), "调色板存在 key 碰撞"
    lut[keys] = vals
    return lut


def rgb_to_class(arr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB -> (H,W) int8 类别号；未匹配像素为 -1。"""
    a = arr
    if a.ndim == 3 and a.shape[2] == 4:      # 去掉 alpha
        a = a[:, :, :3]
    if a.ndim == 3 and a.shape[2] == 1:      # 灰度当作 R=G=B
        a = np.repeat(a, 3, axis=2)
    key = (a[:, :, 0].astype(np.int64) << 16) | \
          (a[:, :, 1].astype(np.int64) << 8) | \
           a[:, :, 2].astype(np.int64)
    return lut[key.ravel()].reshape(key.shape)


# ---------------------------------------------------------------- tar 遍历

MEMBER_RE = re.compile(
    r"^(?P<scene>Scene\d+)/(?P<cond>[^/]+)/frames/(?P<kind>[^/]+)/"
    r"(?P<cam>Camera_\d+)/(?P<file>[^/]+)$"
)


def iter_members(tar: tarfile.TarFile, kind: str, scenes: set | None, suffix_re: re.Pattern):
    """按 (scene, cond, cam, frame) 产出 tar 成员信息。"""
    for m in tar:
        if not m.isfile():
            continue
        mo = MEMBER_RE.match(m.name)
        if not mo:
            continue
        if mo.group("kind") != kind:
            continue
        scene = SCENE_MAP.get(mo.group("scene"))
        if scene is None:
            continue                      # 不在本数据集场景集合内，跳过
        if scenes and scene not in scenes:
            continue
        cond = COND_MAP.get(mo.group("cond"))
        cam = CAM_MAP.get(mo.group("cam"))
        if cond is None or cam is None:
            continue
        fm = suffix_re.search(mo.group("file"))
        if not fm:
            continue
        yield scene, cond, cam, fm.group(1), m


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 VKITTI 2 原始 tar 生成 SDCombo 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb", required=True, help="vkitti_2.0.3_rgb.tar")
    ap.add_argument("--depth", required=True, help="vkitti_2.0.3_depth.tar")
    ap.add_argument("--classseg", required=True, help="vkitti_2.0.3_classSegmentation.tar")
    ap.add_argument("--out", required=True, help="输出根目录")
    ap.add_argument("--train-scenes", default=",".join(DEFAULT_TRAIN_SCENES),
                    help="训练场景，逗号分隔（其余归 validation）")
    ap.add_argument("--scenes", default=None,
                    help="只处理这些场景（调试用），默认全部")
    ap.add_argument("--limit", default=0, type=int,
                    help="每个 split 最多处理多少帧（调试用），0=不限")
    ap.add_argument("--jpeg-quality", default=95, type=int, help="重存 JPEG 的质量")
    ap.add_argument("--keep-depth-16bit", action="store_true", default=True,
                    help="保留 16bit 深度（默认开启，勿关）")
    ap.add_argument("--report", default=None, help="把统计写入该 json 文件")
    args = ap.parse_args()

    train_scenes = {s.strip() for s in args.train_scenes.split(",") if s.strip()}
    only_scenes = {s.strip() for s in args.scenes.split(",")} if args.scenes else None

    lut = build_lut(PALETTE)
    suffix_re = re.compile(r"_(\d+)\.\w+$")

    subdirs = {}
    for kind in ("images", "annotations", "depth"):
        for split in ("training", "validation"):
            d = os.path.join(args.out, kind, split)
            os.makedirs(d, exist_ok=True)
            subdirs[(kind, split)] = d

    # ---- 先用 rgb tar 的成员表作为"基准清单" ----
    print("[1/4] 扫描 rgb tar，建立基准文件清单 ...")
    t0 = time.perf_counter()
    with tarfile.open(args.rgb, "r|") as tar:
        base = [(s, c, cam, f) for (s, c, cam, f, _) in
                iter_members(tar, "rgb", only_scenes, suffix_re)]
    print(f"      rgb 帧数: {len(base)}  (耗时 {time.perf_counter() - t0:.1f}s)")
    if not base:
        print("[ERROR] 没有扫描到任何 rgb 帧，请检查 tar 路径与 --scenes 参数", file=sys.stderr)
        return 1

    base_set = set(base)
    counts = Counter()
    for s, c, cam, f in base:
        counts["training" if s in train_scenes else "validation"] += 1
    print(f"      划分: training={counts['training']}  validation={counts['validation']}")

    stats = {
        "frames_total": len(base),
        "frames_training": counts["training"],
        "frames_validation": counts["validation"],
        # 用嵌套 dict（而不是 (kind, split) 元组键），否则无法 JSON 序列化
        "written": {k: {"training": 0, "validation": 0}
                    for k in ("images", "annotations", "depth")},
        "missing_in_depth": [],
        "missing_in_classseg": [],
        "unmatched_colors": defaultdict(int),
        "class_histogram": defaultdict(int),
        "depth_dtype": None,
        "depth_min": None,
        "depth_max": None,
        "errors": [],
    }
    written = defaultdict(set)          # (kind, split) -> set(frame_key)

    def frame_name(scene, cond, cam, frame, ext):
        return f"rgb_{scene}_{cond}_{cam}_{int(frame):05d}{ext}"

    # ---- 标注：彩色 RGB -> 0..14 单通道 ----
    print("\n[2/4] 转换 classSegmentation（向量化 LUT）...")
    t0 = time.perf_counter()
    n_done = n_skip = 0
    hist = np.zeros(16, dtype=np.int64)
    with tarfile.open(args.classseg, "r|") as tar:
        for scene, cond, cam, frame, m in iter_members(tar, "classSegmentation",
                                                       only_scenes, suffix_re):
            key = (scene, cond, cam, frame)
            if key not in base_set:
                n_skip += 1
                continue
            split = "training" if scene in train_scenes else "validation"
            if args.limit and stats["written"]["annotations"][split] >= args.limit:
                continue
            fh = tar.extractfile(m)
            arr = np.array(Image.open(io.BytesIO(fh.read())))
            cls = rgb_to_class(arr, lut)
            bad = cls < 0
            if bad.any():
                # 记录未收录颜色（换算回 RGB）
                b = arr[bad][:, :3]
                for row in np.unique(b.reshape(-1, 3), axis=0):
                    stats["unmatched_colors"][f"{row[0]},{row[1]},{row[2]}"] += 1
            cls = np.where(bad, 0, cls).astype(np.uint8)
            hist += np.bincount(cls.ravel(), minlength=16)[:16]
            name = frame_name(scene, cond, cam, frame, ".png")
            Image.fromarray(cls, mode="L").save(
                os.path.join(subdirs[("annotations", split)], name), optimize=True)
            written[("annotations", split)].add(name)
            stats["written"]["annotations"][split] += 1
            n_done += 1
            if n_done % 2000 == 0:
                print(f"      {n_done} 张 ... {time.perf_counter() - t0:.0f}s")
    print(f"      写出 {n_done} 张（跳过 {n_skip} 张不在 rgb 清单内）"
          f"  耗时 {time.perf_counter() - t0:.1f}s")
    stats["class_histogram"] = {int(i): int(v) for i, v in enumerate(hist) if v}

    # ---- 深度：保持 16bit 灰度 ----
    print("\n[3/4] 写出 depth（保持 16bit）...")
    t0 = time.perf_counter()
    n_done = n_skip = 0
    dmin, dmax = None, None
    with tarfile.open(args.depth, "r|") as tar:
        for scene, cond, cam, frame, m in iter_members(tar, "depth",
                                                       only_scenes, suffix_re):
            key = (scene, cond, cam, frame)
            if key not in base_set:
                n_skip += 1
                continue
            split = "training" if scene in train_scenes else "validation"
            if args.limit and stats["written"]["depth"][split] >= args.limit:
                continue
            fh = tar.extractfile(m)
            img = Image.open(io.BytesIO(fh.read()))
            if stats["depth_dtype"] is None:
                stats["depth_dtype"] = img.mode
            arr = np.array(img)
            if arr.dtype == np.uint8:
                stats["errors"].append(
                    f"depth 为 8bit（{m.name}）：16bit 深度信息已丢失，"
                    f"请确认下载的是 vkitti_2.0.3_depth.tar")
            dmin = arr.min() if dmin is None else min(dmin, int(arr.min()))
            dmax = arr.max() if dmax is None else max(dmax, int(arr.max()))
            name = frame_name(scene, cond, cam, frame, ".png")
            img.save(os.path.join(subdirs[("depth", split)], name))
            written[("depth", split)].add(name)
            stats["written"]["depth"][split] += 1
            n_done += 1
            if n_done % 2000 == 0:
                print(f"      {n_done} 张 ... {time.perf_counter() - t0:.0f}s")
    stats["depth_min"], stats["depth_max"] = dmin, dmax
    print(f"      写出 {n_done} 张（跳过 {n_skip}）  dtype={stats['depth_dtype']}  "
          f"范围 [{dmin}, {dmax}]  耗时 {time.perf_counter() - t0:.1f}s")

    # ---- RGB：原样复制字节（不重编码） ----
    print("\n[4/4] 写出 images（原样复制，不重编码）...")
    t0 = time.perf_counter()
    n_done = n_skip = 0
    with tarfile.open(args.rgb, "r|") as tar:
        for scene, cond, cam, frame, m in iter_members(tar, "rgb",
                                                       only_scenes, suffix_re):
            key = (scene, cond, cam, frame)
            if key not in base_set:
                n_skip += 1
                continue
            split = "training" if scene in train_scenes else "validation"
            if args.limit and stats["written"]["images"][split] >= args.limit:
                continue
            ext = os.path.splitext(m.name)[1].lower()
            name = frame_name(scene, cond, cam, frame, ext)
            dst = os.path.join(subdirs[("images", split)], name)
            fh = tar.extractfile(m)
            with open(dst, "wb") as out:
                while True:
                    buf = fh.read(1 << 20)
                    if not buf:
                        break
                    out.write(buf)
            written[("images", split)].add(name)
            stats["written"]["images"][split] += 1
            n_done += 1
            if n_done % 2000 == 0:
                print(f"      {n_done} 张 ... {time.perf_counter() - t0:.0f}s")
    print(f"      写出 {n_done} 张（跳过 {n_skip}）  耗时 {time.perf_counter() - t0:.1f}s")

    # ---- 一致性校验：三个目录的"主文件名"必须完全相同 ----
    # 注意 images 保留原始 .jpg，annotations/depth 为 .png，所以比较时要去掉扩展名。
    # dataset_VKITTI.py 把三个目录收成三个独立列表，只校验数量相等、不做文件名配对；
    # 一旦错配就会静默训练到错误的标签/深度，因此这里必须严格校验。
    def _stems(kind, split):
        return {os.path.splitext(n)[0] for n in written[(kind, split)]}

    print("\n=== 一致性校验（按主文件名比对）===")
    ok = True
    for split in ("training", "validation"):
        a = _stems("images", split)
        b = _stems("annotations", split)
        c = _stems("depth", split)
        same = (a == b == c)
        ok &= same
        print(f"  {split:>10}: images={len(a)} annotations={len(b)} depth={len(c)}  "
              f"{'一致' if same else '★不一致★'}")
        if not same:
            for label, x, y in (("images-annotations", a, b), ("images-depth", a, c)):
                only_x, only_y = x - y, y - x
                if only_x or only_y:
                    print(f"      {label}: 仅前者 {len(only_x)} 个, 仅后者 {len(only_y)} 个")
                    for s in list(only_x)[:3]:
                        print(f"        仅前者: {s}")
                    for s in list(only_y)[:3]:
                        print(f"        仅后者: {s}")

    print("\n=== 类别分布 ===")
    names = [k for k, _ in sorted(PALETTE.items(), key=lambda kv: kv[1])]
    total = sum(stats["class_histogram"].values()) or 1
    for i in range(15):
        cnt = stats["class_histogram"].get(i, 0)
        print(f"  类别 {i:>2} ({str(names[i]):>13}): {cnt:>12,}  {cnt / total:>6.2%}")
    print(f"  忽略类 0 占比 {stats['class_histogram'].get(0, 0) / total:.2%}"
          f"（loss 中 ignore_index=0，不参与训练）")

    if stats["unmatched_colors"]:
        ok = False
        print(f"\n★ 发现 {len(stats['unmatched_colors'])} 种未收录颜色"
              f"（已按类别 0 处理，这会造成标签错误）:")
        for c, n in list(stats["unmatched_colors"].items())[:10]:
            print(f"    RGB({c}) x{n}")
    else:
        print("\n所有像素颜色均在 15 色调色板内。")

    print(f"\n深度 dtype={stats['depth_dtype']} 范围 [{stats['depth_min']}, {stats['depth_max']}]"
          f"  （16bit 原始值，SDCHead 会自行按最大值归一化）")

    stats["written"] = dict(stats["written"])
    stats["unmatched_colors"] = dict(stats["unmatched_colors"])
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        print(f"统计已写入 {args.report}")

    print("\n" + "=" * 60)
    print("数据准备完成。" if ok else "数据准备完成，但存在上述 ★ 标记的问题，请先处理。")
    print(f"数据集根目录: {os.path.abspath(args.out)}")
    print("=" * 60)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
