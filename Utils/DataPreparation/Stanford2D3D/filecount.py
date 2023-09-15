import os

from_path = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
for a in area:
    images_path = os.path.join(from_path, a, 'semantic')
    print(len(os.listdir(images_path)))