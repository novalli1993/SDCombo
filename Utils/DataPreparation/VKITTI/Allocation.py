import os
import shutil
import time

from_path = "M:\Dataset\VKITTI"
to_path = "T:\Dataset\VKITTI_II"
if os.path.exists(to_path):
    pass
else:
    os.mkdir(to_path)
tips = ["rgb", "dep", "cls", "ins"]
Scene = ["S01", "S02", "S06", "S18", "S20"]
dirs = {"rgb": "rgb", "dep": "depth", "cls": "clsseg", "ins": "insseg"}
group = ["images", "annotations", "depth", "insseg"]
pipeline = ["training", "validation"]

if not os.path.exists(to_path + "\\" + dirs["dep"]):
    os.mkdir(to_path + "\\" + dirs["dep"])
if not os.path.exists(to_path + "\\" + dirs["ins"]):
    os.mkdir(to_path + "\\" + dirs["ins"])
for d1 in group:
    p1 = to_path + "\\" + d1
    if os.path.exists(p1):
        pass
    else:
        os.mkdir(p1)
    for d2 in pipeline:
        p2 = p1 + "\\" + d2
        if os.path.exists(p2):
            pass
        else:
            os.mkdir(p2)

a = [0, 0, 0, 0, 0, 0, 0, 0]
s = 0
time_start = time.perf_counter()
for d in dirs:
    work_path = from_path + "\\" + dirs[d]
    print(work_path)
    for _, _, p in os.walk(work_path):
        for n in p:
            if n[0:3] == "rgb":
                if n[4:7]=="S01" or n[4:7]=="S06" or n[4:7]=="S18" or n[4:7]=="S20":
                    a[0] +=1
                    # shutil.copy2(work_path + "\\" + n, to_path + '\\' + group[0] + '\\' + pipeline[0])
                else:
                    a[1] += 1
                    # shutil.copy2(work_path + "\\" + n, to_path + '\\' + group[0] + '\\' + pipeline[1])
            elif n[0:3] == "cls":
                if n[4:7]=="S01" or n[4:7]=="S06" or n[4:7]=="S18" or n[4:7]=="S20":
                    a[2] +=1
                    # shutil.copy2(work_path + "\\" + n, to_path + '\\' + group[1] + '\\' + pipeline[0])
                else:
                    a[3] += 1
                    # shutil.copy2(work_path + "\\" + n, to_path + '\\' + group[1] + '\\' + pipeline[1])
            elif n[0:3] == "dep":
                if n[4:7] == "S01" or n[4:7] == "S06" or n[4:7] == "S18" or n[4:7] == "S20":
                    a[4] += 1
                    shutil.copy2(work_path + "\\" + n, to_path + '\\' + group[2] + '\\' + pipeline[0])
                else:
                    a[5] += 1
                    shutil.copy2(work_path + "\\" + n, to_path + '\\' + group[2] + '\\' + pipeline[1])
            elif n[0:3] == "ins":
                if n[4:7]=="S01" or n[4:7]=="S06" or n[4:7]=="S18" or n[4:7]=="S20":
                    a[6] +=1
                    # shutil.copy2(work_path + "\\" + n, to_path + '\\' + group[3] + '\\' + pipeline[0])
                else:
                    a[7] += 1
                    # shutil.copy2(work_path + "\\" + n, to_path + '\\' + group[3] + '\\' + pipeline[1])
            for i in a:
                s = a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
            if s % 100 == 0:
                print(a)
                time_end = time.perf_counter()
                print(time_end-time_start)
print("Complete!")
print(a)
print(s)
