import os
import shutil
import time

from_path = "T:/Dataset/VKITTIS_II"
to_path = "T:/Dataset/VKITTIS_II"

group = ["images", "annotations", "depth", "insseg"]
pipeline = ["training", "validation", "test"]
tips = ["rgb", "dep", "cls", "ins"]
Scene = ["S01", "S02", "S06", "S18", "S20"]
dirs = {"rgb": "rgb", "dep": "depth", "cls": "clsseg", "ins": "insseg"}
dir2 = {"15-deg-left": "15l", "15-deg-right": "15r", "30-deg-left": "30l", "30-deg-right": "30r", "clone": "clo",
        "fog": "fog", "morning": "mor", "overcast": "ove", "rain": "rai", "sunset": "sun"}

if os.path.exists(to_path):
    pass
else:
    os.mkdir(to_path)
for gt in group:
    group_path = to_path + "/" + gt
    if os.path.exists(group_path):
        pass
    else:
        os.mkdir(group_path)
    for pt in pipeline:
        pipe_path = group_path + "/" + pt
        if os.path.exists(pipe_path):
            pass
        else:
            os.mkdir(pipe_path)

a = b = 0
timestart = time.perf_counter()
for gf in group:
    work_dirs = from_path + "/" + gf + "/training"
    for _, _, p in os.walk(work_dirs):
        for n in p:
            image_dir = work_dirs + '/' + n
            if n[-5] == '3':
                des_dir = to_path + "/" + gf + "/test"
                shutil.move(image_dir, des_dir)
                b += 1
            a += 1
            if a % 1000 == 0:
                timeend = time.perf_counter()
                total = (timeend-timestart)/a*113580
                print(str(b) + '/' + str(a) + "/113580")
                print("{:.2f}".format(timeend-timestart),end=", eta=")
                print("{:.2f}".format(total-(timeend-timestart)))
print("Copy: ", end='')
print("{:.2%}".format(b/a))
print("Complete!")