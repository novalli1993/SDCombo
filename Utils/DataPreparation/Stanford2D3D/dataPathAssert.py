import os

dataset = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
types = ["rgb", "semantic", "depth", "predict"]

weight = "J:/Model/SDCombo/work_dir/model/model_20230811_153904_8.pth"

select = {'area': 0, 'pic_num': 0}
file_name_prefix = os.listdir(os.path.join(dataset, area[select['area']], types[0]))[select['pic_num']][:-7]
file_name_suffix = '.png'
rgb = os.path.join(dataset, area[select['area']], types[0], file_name_prefix + types[0] + file_name_suffix)
semantic = os.path.join(dataset, area[select['area']], types[1], file_name_prefix + types[1] + file_name_suffix)
depth = os.path.join(dataset, area[select['area']], types[2], file_name_prefix + types[2] + file_name_suffix)

name = os.path.split(rgb)[1][:-7]
path = os.path.split(os.path.split(rgb)[0])[0]
image = os.path.join(path, "rgb",name + "rgb.png")

print(name)
print(path)
print(image)