import os

from_path = "T:\\Dataset\\KITTI\\vkitti_2.0.3_textgt"
to_path = "T:\\Dataset\\VKITTI_II"

dir1 = {"Scene01": "S01", "Scene02": "S02", "Scene06": "S06", "Scene18": "S18", "Scene20": "S20"}
dir2 = {"15-deg-left": "15l", "15-deg-right": "15r", "30-deg-left": "30l", "30-deg-right": "30r", "clone": "clo",
        "fog": "fog", "morning": "mor", "overcast": "ove", "rain": "rai", "sunset": "sun"}
classes = [["", "", ""]]

# for d1 in dir1:
#     for d2 in dir2:
#         work_path = from_path + "\\" + d1 + "\\" + d2
#         f = open(work_path + "\\colors.txt")
#         line = f.readline()
#         while line:
#             palette = line.split(" ")
#             palette[3] = palette[3][0:len(palette[3]) - 1]
#             if classes.count(palette) == 1:
#                 pass
#             else:
#                 classes.append(palette)
#             line = f.readline()
# classes.remove(['', '', ''])
# classes.remove(['Category', 'r', 'g', 'b'])
# classes.sort()

work_path = from_path + "\\Scene01\\15-deg-left"
f = open(work_path + "\\colors.txt")
line = f.readline()
while line:
    palette = line.split(" ")
    palette[3] = palette[3][0:len(palette[3]) - 1]
    if classes.count(palette) == 1:
        pass
    else:
        classes.append(palette)
    line = f.readline()
classes.remove(['', '', ''])
classes.remove(['Category', 'r', 'g', 'b'])


c = []
p = []
for i in classes:
    p.append([0, 0, 0])
    for j in range(4):
        if j == 0:
            c.append(i[j])
        elif j == 1 or j == 2:
            p[len(p) - 1][j - 1] = i[j]
        elif j == 3:
            p[len(p) - 1][j - 1] = i[j]
for l in range(len(c)):
    print('\'' + c[l] + '\',', end=" ")
print('')
for m in range(len(p)):
    print('[' + str(p[m][0])+","+ str(p[m][1])+","+ str(p[m][2])+'],', end=' ')

# print('')
# g = []
# for m in range(len(p)):
#     g.append(int(p[m][0]) * 0.299 + int(p[m][1]) * 0.587 + int(p[m][2]) * 0.114)
#     print(g[m], end=', ')
