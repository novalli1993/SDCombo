import os
import zipfile
import shutil
import time

data_dir = "T:/Dataset/KITTI/rawdata"
zip_file = data_dir +'/2011_09_26_drive_0001_sync.zip'
save_dir = data_dir +'/2011_09_26_drive_0001_sync.zip/2011_09_26/2011_09_26_drive_0001_sync/image_02/data'
if os.path.exists(save_dir):
    pass
else:
    os.mkdir(save_dir)
file=zipfile.ZipFile(zip_file)
file.extract("2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png",save_dir)