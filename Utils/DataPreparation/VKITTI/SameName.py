import os

work_path = "T:\\Dataset\\VKITTI_II\\depth"
dirs = ["training", "validation"]

for d in dirs:
    file_path = work_path + "\\" + d
    for _,_,p in os.walk(file_path):
        for n in p:
            new = "rgb" + n[3:]
            print(new)
            os.renames(file_path + "\\" + n, file_path + "\\" + new)