import os
import shutil
from pathlib import Path
import numpy as np

dir0 = ["vkitti_2.0.3_classSegmentation", "vkitti_2.0.3_depth", "vkitti_2.0.3_instanceSegmentation", "vkitti_2.0.3_rgb"]
dir1 = {"Scene01":"S01", "Scene02":"S02", "Scene06":"S06", "Scene18":"S18", "Scene20":"S20"}
dir2 = {"15-deg-left":"15l", "15-deg-right":"15r", "30-deg-left":"30l", "30-deg-right":"30r", "clone":"clo", "fog":"fog", "morning":"mor", "overcast":"ove", "rain":"rai", "sunset":"sun"}
dir3 = {"rgb":"rgb","depth":"dep","classsegmentation":"cls","instancesegmentation":"ins"}
dir4 = {"Camera_0":"c0","Camera_1":"c1"}
dst_path = "T:\Dataset\KITTI"
c = np.zeros((4, 5), dtype=int)
for d0 in dir0:
    for d1 in dir1:
        for d2 in dir2:
            for d3 in dir3:
                for d4 in dir4:
                    title = dir3[d3] + "_" + dir1[d1] + "_" + dir2[d2] + "_" + dir4[d4] + "_"
                    path = dst_path + "\\" + d0 + "\\" + d1 + "\\" + d2 + "\\frames\\" + d3 + "\\" + d4
                    for _,_,p in os.walk(path):
                        for n in p:
                            if d3 == "rgb":
                                i = 0
                            elif d3 == "depth":
                                i = 1
                            elif d3 == "classsegmentation":
                                i = 2
                            elif d3== "instancesegmentation":
                                i = 3
                            if d1 == "Scene01":
                                c[i][0] += 1
                            elif d1 == "Scene02":
                                c[i][1] += 1
                            elif d1 == "Scene06":
                                c[i][2] += 1
                            elif d1== "Scene18":
                                c[i][3] += 1
                            elif d1== "Scene20":
                                c[i][4] += 1
s = np.sum(c)
hline1 = "-" * (len("instancesegmentation") + len("Scene01 Scene02 Scene06 Scene18 Scene20 ") + len(str(s)) + 2)
print("\n" + hline1)
print("Count" + " "*(len("instancesegmentation")-len("Count")+2) + "Scene01 Scene02 Scene06 Scene18 Scene20 sum")
i = -1
for d3 in dir3:
    i += 1
    print(d3+' '*(len("instancesegmentation")-len(d3)+2), end='')
    for j in range(5):
        print(str(c[i][j]) + ' ' * (len("Scene01") - len(str(c[i][j])) + 1), end="")
    print(np.sum(c[i]))
print("sum" + ' ' * (len("instancesegmentation") - len("sum") + 2),end='')
for i in range(5):
    print(str(np.sum(c, axis=0)[i]) + ' ' * (len("Scene01") - len(str(np.sum(c, axis=0)[i])) + 1), end='')
print(str(np.sum(c)))
print(hline1)
r=np.zeros(c.shape)
for i in range(len(c)):
    r[i] = c[i] / s
hline2 = "-" * (len("instancesegmentation") + len("Scene01 Scene02 Scene06 Scene18 Scene20 ") + len(str('{:.0%}'.format(np.sum(r)))) + 2)
print("\n" + hline2)
print("Rate" + " "*(len("instancesegmentation")-len("Rate")+2) + "Scene01 Scene02 Scene06 Scene18 Scene20 sum")
i = -1
for d3 in dir3:
    i += 1
    print(d3+' '*(len("instancesegmentation")-len(d3)+2), end='')
    for j in range(5):
        print('{:.2%}'.format(r[i][j])+' '*(len("Scene01")-len('{:.2%}'.format(r[i][j]))+1), end="")
    print('{:.0%}'.format(np.sum(r[i])))
print("sum" + ' ' * (len("instancesegmentation") - len("sum") + 2),end='')
for i in range(5):
    print('{:.2%}'.format(np.sum(r, axis=0)[i]) + ' '*(len("Scene01")-len('{:.2%}'.format(np.sum(r, axis=0)[i]))+1), end='')
print(str('{:.0%}'.format(np.sum(r))))
print(hline2)
