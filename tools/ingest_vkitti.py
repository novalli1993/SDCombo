"""校验下载的 VKITTI 2 tar 包 MD5，然后把它们移动到指定的原始数据目录。

用法:
    python tools/ingest_vkitti.py --dest datasets/_raw
    python tools/ingest_vkitti.py --dest datasets/_raw --verify-only
"""
import argparse
import hashlib
import os
import shutil
import sys
import time

# 官方 vkitti_2.0.3_md5_checksums.txt
EXPECT = {
    "vkitti_2.0.3_rgb.tar": "1e00a143a397c2c53aa9720a868fe34a",
    "vkitti_2.0.3_depth.tar": "1d34a96f870fc5d33b482df87844f5e4",
    "vkitti_2.0.3_classSegmentation.tar": "e8658ab49250e61f1caa174625563263",
}
# 顺带校验（本项目不需要，若也下载了可一起搬）
OPTIONAL = {
    "vkitti_2.0.3_instanceSegmentation.tar": "c79684c6344e50663587b8da848535e7",
    "vkitti_2.0.3_textgt.tar.gz": "855f7f37746e5508f094e522c9cf41f4",
    "vkitti_2.0.3_forwardFlow.tar": "a4088e9272ad4c91da0209b99792aa22",
    "vkitti_2.0.3_backwardFlow.tar": "23ef61ef75eba690adcfe437ac71a58e",
    "vkitti_2.0.3_forwardSceneFlow.tar": "a4d88a2a770de1615dcacd21d41569a7",
    "vkitti_2.0.3_backwardSceneFlow.tar": "928826f9896e35d1d31719db1028c478",
}


def md5_of(path, chunk=1 << 22):
    h = hashlib.md5()
    done = 0
    total = os.path.getsize(path)
    t0 = time.time()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
    return h.hexdigest(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.expanduser("~"), "Downloads"))
    ap.add_argument("--dest", required=True, help="tar 归档的目标目录")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--include-optional", action="store_true",
                    help="同时搬移本项目不需要的其它 tar")
    args = ap.parse_args()

    targets = dict(EXPECT)
    if args.include_optional:
        targets.update(OPTIONAL)

    print(f"源目录  : {args.src}")
    print(f"目标目录: {os.path.abspath(args.dest)}")
    print()

    missing, failed, ok = [], [], []
    for name, exp in targets.items():
        src = os.path.join(args.src, name)
        if not os.path.isfile(src):
            if name in EXPECT:
                missing.append(name)
            continue
        size_gib = os.path.getsize(src) / 1024 ** 3
        got, dt = md5_of(src)
        if got == exp:
            print(f"OK   {name:38} {size_gib:5.2f} GiB  md5={got}  ({dt:.0f}s)")
            ok.append((name, src))
        else:
            print(f"FAIL {name:38} {size_gib:5.2f} GiB")
            print(f"       期望 {exp}")
            print(f"       实得 {got}")
            failed.append(name)

    if missing:
        print("\n★ 缺失必需文件:")
        for m in missing:
            print(f"    {m}")

    if args.verify_only or failed or missing:
        print()
        if failed or missing:
            print("校验未通过，未移动任何文件。请重新下载缺失/损坏的文件。")
            return 1
        print("校验全部通过（--verify-only，未移动文件）。")
        return 0

    os.makedirs(args.dest, exist_ok=True)
    print(f"\n=== 移动到 {os.path.abspath(args.dest)} ===")
    for name, src in ok:
        dst = os.path.join(args.dest, name)
        if os.path.abspath(src) == os.path.abspath(dst):
            print(f"  已在目标位置，跳过: {name}")
            continue
        if os.path.exists(dst):
            print(f"  目标已存在，跳过: {name}")
            continue
        t0 = time.time()
        # 同盘用 os.replace 是瞬时的；跨盘回退到 copy2+删除
        try:
            os.replace(src, dst)
        except OSError:
            shutil.copy2(src, dst)
            os.remove(src)
        print(f"  {name}  ->  {dst}  ({time.time() - t0:.0f}s)")

    print("\n完成。目标目录内容:")
    for f in sorted(os.listdir(args.dest)):
        p = os.path.join(args.dest, f)
        print(f"  {f:42} {os.path.getsize(p) / 1024 ** 3:5.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
