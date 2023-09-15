import os
import torch
from torchviz import make_dot
from PIL import Image

dataset = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
types = ["rgb", "semantic", "depth", "predict"]

device = 'cuda'

select = {'area': 0, 'pic_num': 0}
file_name_prefix = os.listdir(os.path.join(dataset, area[select['area']], types[0]))[select['pic_num']][:-7]
file_name_suffix = '.png'
rgb = os.path.join(dataset, area[select['area']], types[0], file_name_prefix + types[0] + file_name_suffix)
semantic = os.path.join(dataset, area[select['area']], types[1], file_name_prefix + types[1] + file_name_suffix)
depth = os.path.join(dataset, area[select['area']], types[2], file_name_prefix + types[2] + file_name_suffix)

from train_HHA import PipelineEval

rgb = Image.open(rgb)
semantic = Image.open(semantic)
depth = Image.open(depth)
data_transform = PipelineEval(128, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
image_tensor, sematic_tensor, depth_tensor = data_transform(rgb, semantic, depth)
image_tensor = torch.unsqueeze(image_tensor, dim=0)
sematic_tensor = torch.unsqueeze(sematic_tensor, dim=0)
depth_tensor = torch.unsqueeze(depth_tensor, dim=0)
image_tensor = image_tensor.to(device)
sematic_tensor = sematic_tensor.to(device)
depth_tensor = depth_tensor.to(device)

from Joint.model_DL4sDL import _SDCombo as SDCombo

model = SDCombo(True, 14)
model.to(device)

model.eval()
vis_graph = make_dot(model(image_tensor, depth_tensor)['out'], params=dict(model.named_parameters()))
vis_graph.view()

print(model)
