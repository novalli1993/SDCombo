import os

os.system("python ./evaluation_HHA.py -weight=\"work_dir/model/model_20230831_220805_38.pth\"")
os.system("python ./evaluation_HHA.py -weight=\"work_dir/model/model_20230831_220805_37.pth\"")
os.system("python ./demo_Depth.py")