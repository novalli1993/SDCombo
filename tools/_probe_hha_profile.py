"""临时探针：剖析 HHA 生成单帧耗时，定位瓶颈。"""
import cProfile
import io
import os
import pstats
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "Utils", "DataPreparation", "Depth2HHA-python")))
sys.path.insert(0, HERE)
from prepare_hha_vkitti import hha_from_depth_cm  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", "datasets", "VKITTI_II"))
CAM = np.array([[725.0087, 0, 620.5], [0, 725.0087, 187.0], [0, 0, 1]])


def main():
    names = sorted(os.listdir(os.path.join(ROOT, "depth", "training")))
    # 取几个不同场景的帧，避免只测到"简单"的那一帧
    picks = [names[0], names[3000], names[9000], names[20000]]
    for name in picks:
        D = cv2.imread(os.path.join(ROOT, "depth", "training", name), cv2.IMREAD_UNCHANGED)
        t = time.time()
        hha, inv = hha_from_depth_cm(D, CAM)
        dt = time.time() - t
        print("%-32s  %5.2f s   invalid=%.1f%%" % (name, dt, inv * 100))

    # 对第一帧做一次函数级剖析
    D = cv2.imread(os.path.join(ROOT, "depth", "training", picks[0]), cv2.IMREAD_UNCHANGED)
    pr = cProfile.Profile()
    pr.enable()
    hha_from_depth_cm(D, CAM)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
    print()
    print(s.getvalue())


if __name__ == "__main__":
    main()
