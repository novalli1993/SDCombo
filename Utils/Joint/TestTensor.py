import numpy as np

import torch.utils.data

from Dataset.dataset_VKITTI import *
from Backbone.mmseg_custom.models.backbones.intern_image import *

device = torch.device("cuda")
intern_image = InternImage(core_op='DCNv3', channels=160, depths=[5, 5, 22, 5], groups=[10, 20, 40, 80], act_layer='GELU', norm_layer='LN').to(device)

image_dir = 'J:/Dataset/VKITTIS/images/training/rgb_S01_15l_c0_000.jpg'
image = Image.open(image_dir)
input_tensor = torch.tensor(np.array(image,dtype=int))
input_tensor=torch.unsqueeze(input_tensor, 0)
input_tensor=input_tensor.permute(0,3,1,2).to(device)
# print(input_tensor.shape)
# print(input_tensor)

output_tensors = intern_image(input_tensor.float())

for output_tensor in output_tensors:
    print(output_tensor.shape)
    print(output_tensor.dtype)