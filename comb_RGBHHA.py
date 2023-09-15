import datetime
import os
from PIL import Image, ImageDraw, ImageFont

data_path = "J:/Dataset/Stanford2D3D"
areas = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]

compare_path = "Demo/Compare"
if os.path.exists(compare_path):
    pass
else:
    os.mkdir(compare_path)

mark = "RGBHHA"
save_dir = os.path.join(compare_path, mark)
if os.path.exists(save_dir):
    pass
else:
    os.mkdir(save_dir)

for area in areas:
    save_path = os.path.join(save_dir, area)
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)
    file_list = os.listdir(os.path.join(data_path, area, "RGB"))
    for file in file_list[:100]:
        file_prefix = file[:-7]
        file_RGB = os.path.join(data_path, area, "RGB", file)
        file_HHA = os.path.join(data_path, area, "HHA", file_prefix + "HHA.png")

        image_RGB = Image.open(file_RGB)
        image_HHA = Image.open(file_HHA)

        image = Image.new("RGB",(image_RGB.size[0],(image_RGB.size[1]+128)*2))
        image.paste(image_RGB, (0,0))
        image.paste(image_HHA, (0, image_RGB.size[1]+128))

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('C:/Users/noval/AppData/Local/Microsoft/Windows/Fonts/Microsoft_YaHei_UI.TTF', 100)
        draw.text((0, image_RGB.size[1]), "RGB", (255, 255, 255), font=font)
        draw.text((0, image_RGB.size[1] * 2 + 128), "HHA", (255, 255, 255), font=font)

        image.save(os.path.join(save_path, file_prefix + "RGBHHA.png"))
