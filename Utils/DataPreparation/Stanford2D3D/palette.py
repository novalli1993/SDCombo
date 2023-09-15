import torch

from Backup.dataset_Stanford2D3D import *

device = 'cuda'

# file from dataset
dataset = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
annotation_path = os.path.join(dataset, area[0], "semantic")
file_list = os.listdir(annotation_path)
# annotation_file = os.path.join(annotation_path, file_list[0])
annotation_file = os.path.join(annotation_path, "camera_00d10d86db1e435081a837ced388375f_office_24_frame_0_domain_semantic.png")
print(annotation_file)

# file from other place
# annotation_file = ""

annotation = torch.from_numpy(np.array(Image.open(annotation_file)).astype(np.int8)).to(device)
print(annotation)
palette = torch.zeros(annotation.size()[0], annotation.size()[1], 3).type(torch.int).to(device)
for i in range(len(CLASSES)):
    palette[:, :, 0] += (annotation[:, :] == i) * PALETTE[i][0]
    palette[:, :, 1] += (annotation[:, :] == i) * PALETTE[i][1]
    palette[:, :, 2] += (annotation[:, :] == i) * PALETTE[i][2]

palette = (palette.cpu().numpy()).astype(np.uint8)
print(palette)
palette = Image.fromarray(palette)
palette.show()