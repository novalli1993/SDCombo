import os
import shutil

dir0 = {"vkitti_2.0.3_classSegmentation", "vkitti_2.0.3_depth", "vkitti_2.0.3_instanceSegmentation", "vkitti_2.0.3_rgb"}
dir1 = {"Scene01": "S01", "Scene02": "S02", "Scene06": "S06", "Scene18": "S18", "Scene20": "S20"}
dir2 = {"15-deg-left": "15l", "15-deg-right": "15r", "30-deg-left": "30l", "30-deg-right": "30r", "clone": "clo",
        "fog": "fog", "morning": "mor", "overcast": "ove", "rain": "rai", "sunset": "sun"}
dir3 = {"rgb": "rgb", "depth": "dep", "classsegmentation": "cls", "instancesegmentation": "ins"}
dir4 = {"Camera_0": "c0", "Camera_1": "c1"}
dst_path = "T:\Dataset\KITTI"
des_path = "M:\Dataset\VKITTI"
dp_rgb = des_path + "\\rgb"
dp_depth = des_path + "\\depth"
dp_clsseg = des_path + "\\clsseg"
dp_insseg = des_path + "\\insseg"
dp = {"rgb": dp_rgb, "depth": dp_depth, "classsegmentation": dp_clsseg, "instancesegmentation": dp_insseg}
if os.path.exists(des_path):
    pass
else:
    os.mkdir(des_path)
for i in dp:
    if os.path.exists(dp[i]):
        pass
    else:
        os.mkdir(dp[i])
a = [0, 0, 0, 0]
for d0 in dir0:
    for d1 in dir1:
        for d2 in dir2:
            for d3 in dir3:
                for d4 in dir4:
                    title = dir3[d3] + "_" + dir1[d1] + "_" + dir2[d2] + "_" + dir4[d4] + "_"
                    path = dst_path + "\\" + d0 + "\\" + d1 + "\\" + d2 + "\\frames\\" + d3 + "\\" + d4
                    for _, _, p in os.walk(path):
                        for n in p:
                            name = title + n[-7:]
                            file = path + "\\" + n
                            new_file = dp[d3] + "\\" + name
                            shutil.copy(file, new_file)
                            if d3 == "rgb":
                                a[0] += 1
                            elif d3 == "depth":
                                a[1] += 1
                            elif d3 == "classsegmentation":
                                a[2] += 1
                            elif d3 == "instancesegmentation":
                                a[3] += 1
                            if (a[0] + a[1] + a[2] + a[3]) % 100 == 0:
                                print(a)
print("Complete!")
