import datetime
import os
from PIL import Image, ImageDraw, ImageFont

compare_path = "Demo/Compare"
if os.path.exists(compare_path):
    pass
else:
    os.mkdir(compare_path)
groundtruth = "Demo/Ground Truth"
demo_HHA = "Demo/20230830_151212_8"
demo_512 = "Demo/20230901_131623_9"
demo_40 = "Demo/20230831_220805_34"
demo = {"HHA":demo_HHA, "Depth":demo_512, "DL":demo_40}
areas = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]

mark = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
    file_list = os.listdir(os.path.join(groundtruth, area))
    for file in file_list:
        file_prefix = file[:-6]
        file_gt = os.path.join(area,file_prefix + "gt.png")
        file_demo = os.path.join(area,file_prefix + "demo.png")

        image_gt = Image.open(os.path.join(groundtruth, file_gt))
        image_HHA = Image.open(os.path.join(demo_HHA, file_demo))
        image_Depth = Image.open(os.path.join(demo_512, file_demo))
        image_DL = Image.open(os.path.join(demo_40, file_demo))

        image = Image.new("RGB",(image_gt.size[0]*2+2,image_gt.size[1]*2+128*2))
        image.paste(image_gt, (0,0))
        image.paste(image_HHA, (0, image_gt.size[1]+128))
        image.paste(image_Depth, (image_gt.size[0]+2, 0))
        image.paste(image_DL, (image_gt.size[0]+2, image_gt.size[1]+128))

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('C:/Users/noval/AppData/Local/Microsoft/Windows/Fonts/Microsoft_YaHei_UI.TTF', 100)
        draw.text((0, (image_gt.size[0] * 2 + 2) // 2), "Ground Truth", (255, 255, 255), font=font)
        draw.text(((image_gt.size[0] * 2 + 2) // 2, (image_gt.size[0] * 2 + 2) // 2), "SDCombo size=512",
                  (255, 255, 255),
                  font=font)
        draw.text((0, image_gt.size[1] * 2 + 128), "SDCombo", (255, 255, 255), font=font)
        draw.text(((image_gt.size[0] * 2 + 2) // 2, image_gt.size[1] * 2 + 128), "SDCombo 40 epochs", (255, 255, 255),
                  font=font)

        image.save(os.path.join(save_path, file_prefix + "com.png"))
