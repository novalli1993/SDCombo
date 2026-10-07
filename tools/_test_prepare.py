"""验证 prepare_vkitti.py 的 LUT 正确性 + 用合成 tar 跑通端到端流程。"""
import io
import os
import subprocess
import sys
import tarfile

# 本机控制台是 cp936，子进程输出里可能含无法编码的字符；统一转成可打印形式
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import prepare_vkitti as P  # noqa: E402

FAIL = 0


def check(label, cond):
    global FAIL
    print(f"  {'OK  ' if cond else 'FAIL'} {label}")
    if not cond:
        FAIL += 1


print("=" * 66)
print("[1] LUT 与向量化转换正确性")
print("=" * 66)

lut = P.build_lut(P.PALETTE)
check("LUT 大小 = 2**24", lut.size == (1 << 24))
check("LUT 字节数约 16MiB", lut.nbytes == (1 << 24))

# (a) 每个调色板颜色都能往返
ok_rt = True
for (r, g, b), cls in P.PALETTE.items():
    arr = np.array([[[r, g, b]]], dtype=np.uint8)
    got = P.rgb_to_class(arr, lut)[0, 0]
    if got != cls:
        ok_rt = False
        print(f"    往返失败 RGB({r},{g},{b}) 期望 {cls} 实得 {got}")
check("15 个调色板颜色全部往返正确", ok_rt)

# (b) 类别号唯一（无碰撞）
check("调色板类别号互不相同",
      len(set(P.PALETTE.values())) == len(P.PALETTE))

# (c) 与逐像素参考实现逐元素比对
rng = np.random.default_rng(0)
pal = list(P.PALETTE.items())
h, w = 40, 60
img = np.zeros((h, w, 3), dtype=np.uint8)
ref = np.zeros((h, w), dtype=np.int8)
for y in range(h):
    for x in range(w):
        col, cls = pal[rng.integers(len(pal))]
        img[y, x] = col
        ref[y, x] = cls
got = P.rgb_to_class(img, lut)
check("向量化结果 == 逐像素参考实现", np.array_equal(got, ref))

# (d) 未收录颜色 -> -1
unknown = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
check("未收录颜色返回 -1", (P.rgb_to_class(unknown, lut) == -1).all())

# (e) 带 alpha 通道
rgba = np.array([[[210, 0, 200, 255]]], dtype=np.uint8)
check("RGBA 去除 alpha 后正确", P.rgb_to_class(rgba, lut)[0, 0] == 0)

# (f) 灰度输入
gray = np.zeros((1, 1, 1), dtype=np.uint8)   # 黑 -> 类别 14
check("灰度输入按 R=G=B 处理", P.rgb_to_class(gray, lut)[0, 0] == 14)

print()
print("=" * 66)
print("[2] 用合成 tar 跑端到端")
print("=" * 66)

tmp = os.path.join(REPO, "build_tmp", "prep_test")
os.makedirs(tmp, exist_ok=True)

# 构造：Scene01/15-deg-left（training）与 Scene02/fog（validation）
layout = [("Scene01", "15-deg-left", "S01"), ("Scene02", "fog", "S02")]
NFRAMES = 6
pal_items = list(P.PALETTE.items())


def make_tars():
    """生成 rgb / depth / classSegmentation 三个 tar，内容一一对应。"""
    paths = {k: os.path.join(tmp, f"{k}.tar") for k in ("rgb", "depth", "classSegmentation")}
    handles = {k: tarfile.open(p, "w") for k, p in paths.items()}
    for scene, cond, _ in layout:
        for cam in ("Camera_0", "Camera_1"):
            for f in range(NFRAMES):
                # RGB：随机噪声 jpg
                rgb = rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
                add(handles["rgb"], f"{scene}/{cond}/frames/rgb/{cam}/rgb_{f:05d}.jpg", buf.getvalue())

                # depth：16bit，值域故意超过 255 以验证没被截断
                dep = rng.integers(0, 65535, (32, 48), dtype=np.uint16)
                buf = io.BytesIO()
                Image.fromarray(dep, mode="I;16").save(buf, format="PNG")
                add(handles["depth"], f"{scene}/{cond}/frames/depth/{cam}/depth_{f:05d}.png", buf.getvalue())

                # classSegmentation：只用调色板里的颜色，保证可完全映射
                idx = rng.integers(0, len(pal_items), (32, 48))
                cols = np.array([pal_items[i][0] for i in idx.ravel()], dtype=np.uint8)
                ann = cols.reshape(32, 48, 3)
                buf = io.BytesIO()
                Image.fromarray(ann, mode="RGB").save(buf, format="PNG")
                add(handles["classSegmentation"],
                    f"{scene}/{cond}/frames/classSegmentation/{cam}/classgt_{f:05d}.png", buf.getvalue())
    for h in handles.values():
        h.close()
    return paths


def add(tar, name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


paths = make_tars()
print(f"合成 tar: rgb={os.path.getsize(paths['rgb'])/1024:.0f}KB "
      f"depth={os.path.getsize(paths['depth'])/1024:.0f}KB "
      f"classseg={os.path.getsize(paths['classSegmentation'])/1024:.0f}KB")

out = os.path.join(tmp, "out")
report = os.path.join(tmp, "report.json")
cmd = [sys.executable, os.path.join(HERE, "prepare_vkitti.py"),
       "--rgb", paths["rgb"], "--depth", paths["depth"],
       "--classseg", paths["classSegmentation"], "--out", out, "--report", report]
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print("\n".join(r.stdout.splitlines()[-22:]))
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
check("脚本退出码为 0", r.returncode == 0)

# 校验产出
import json
rep = json.load(open(report, encoding="utf-8"))
exp = 2 * 2 * NFRAMES   # 2 scene x 2 cam x 6 frames
check(f"总帧数 = {exp}", rep["frames_total"] == exp)
for split, n in (("training", 12), ("validation", 12)):
    for kind in ("images", "annotations", "depth"):
        d = os.path.join(out, kind, split)
        got = len(os.listdir(d)) if os.path.isdir(d) else 0
        check(f"{kind}/{split} 文件数 = {n}", got == n)

# 三目录"主文件名"必须完全一致（images 是 .jpg，另外两个是 .png）
for split in ("training", "validation"):
    sets = [{os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, k, split))}
            for k in ("images", "annotations", "depth")]
    check(f"{split} 三目录主文件名集合一致", sets[0] == sets[1] == sets[2])

# 标签内容：应是 0..14 单通道
ann_dir = os.path.join(out, "annotations", "training")
f0 = sorted(os.listdir(ann_dir))[0]
aimg = Image.open(os.path.join(ann_dir, f0))
check(f"标签模式为 L/8bit（实为 {aimg.mode}）", aimg.mode == "L")
avals = np.unique(np.array(aimg))
check(f"标签取值在 0..14（实为 {avals.min()}..{avals.max()}）",
      avals.min() >= 0 and avals.max() <= 14)

# 深度必须是 16bit（>255 的值存在）
dep_dir = os.path.join(out, "depth", "training")
d0 = sorted(os.listdir(dep_dir))[0]
dimg = Image.open(os.path.join(dep_dir, d0))
darr = np.array(dimg)
check(f"深度 dtype={darr.dtype}，max={darr.max()}（应 >255 说明未截断为 8bit）",
      darr.max() > 255)
check("报告记录 depth_max > 255", rep["depth_max"] > 255)
check("无未收录颜色", not rep["unmatched_colors"])

# 文件名格式
check("文件名格式 rgb_S01_15l_c0_00000.jpg",
      "rgb_S01_15l_c0_00000.jpg" in os.listdir(os.path.join(out, "images", "training")))
check("validation 用 Scene02/fog -> S02_fog",
      any(n.startswith("rgb_S02_fog_") for n in os.listdir(os.path.join(out, "images", "validation"))))

print()
print("=" * 66)
print("全部通过" if FAIL == 0 else f"有 {FAIL} 项失败")
print("=" * 66)
sys.exit(1 if FAIL else 0)
