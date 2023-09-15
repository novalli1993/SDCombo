import os
path = "work_dir/logger"
for _,_,p in os.walk(path):
    for file in p:
        print(os.path.split(file))