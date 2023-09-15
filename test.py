import os
from PIL import Image, ImageDraw, ImageFont

compare_path = "Demo/Compare"
if os.path.exists(compare_path):
    pass
else:
    os.mkdir(compare_path)
groundtruth = "Demo/Ground Truth"
demo_HHA = "Demo/20230830_211613_9"
demo_Depth = "Demo/20230830_151212_8"
demo_DL = "Demo/DL_20230831_070701_18"
demo = {"HHA":demo_HHA, "Depth":demo_Depth, "DL":demo_DL}
areas = ["area_1"]

mark = "test"
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
    file = file_list[0]
    file_prefix = file[:-6]
    file_gt = os.path.join(area,file_prefix + "gt.png")
    file_demo = os.path.join(area,file_prefix + "demo.png")

    image_gt = Image.open(os.path.join(groundtruth, file_gt))
    image_HHA = Image.open(os.path.join(demo_HHA, file_demo))
    image_Depth = Image.open(os.path.join(demo_Depth, file_demo))
    image_DL = Image.open(os.path.join(demo_DL, file_demo))

    image = Image.new("RGB",(image_gt.size[0]*2+2,image_gt.size[1]*2+128*2))
    image.paste(image_gt, (0,0))
    image.paste(image_HHA, (0, image_gt.size[1]+128))
    image.paste(image_Depth, (image_gt.size[0]+2, 0))
    image.paste(image_DL, (image_gt.size[0]+2, image_gt.size[1]+128))

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Users/noval/AppData/Local/Microsoft/Windows/Fonts/Microsoft_YaHei_UI.TTF', 100)
    draw.text((0, (image_gt.size[0]*2+2)//2), "Ground Truth", (255, 255, 255), font=font)
    draw.text(((image_gt.size[0] * 2 + 2) // 2, (image_gt.size[0] * 2 + 2) // 2), "SDCombo with HHA", (255, 255, 255),
              font=font)
    draw.text((0, image_gt.size[1]*2+128), "SDCombo with Depth", (255, 255, 255), font=font)
    draw.text(((image_gt.size[0] * 2 + 2) // 2, image_gt.size[1] * 2 + 128), "DeepLabV3", (255, 255, 255),
              font=font)

    image.save(os.path.join(save_path, file_prefix + "com.png"))
