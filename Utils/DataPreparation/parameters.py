import torch
import collections

import torch.optim

import Dataset.transforms as T
from Dataset.dataset_VKITTI import *
from train_val import *
from Segmentation.Models.model import IIP


def create_model(pretrained=None):
    model = IIP(pretrained)
    missing_keys = []
    if pretrained is not None:
        weights_dict = torch.load(pretrained, map_location='cpu')['model']
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
    return model, missing_keys


state_dict = collections.OrderedDict()
module_trained = ['upernet', 'fcnhead']

model, missing_keys = create_model("../save_weights/model_20230725_202828_9.pth")
# model, missing_keys, unexpected_keys = create_model()

for n, p in model.named_parameters():
    print(n)

keys_to_optimize = []
for n, p in model.named_parameters():
    if n in missing_keys:
        p.requires_grad = True
    elif n.split('.')[0] in module_trained and n not in missing_keys:
        keys_to_optimize.append(n)
        p.requires_grad = True
    else:
        p.requires_grad = False

params_to_optimize = [p for p in model.parameters() if p.requires_grad]
print("Parameters to optimize:")
for i in [n for n, p in model.named_parameters() if p.requires_grad]:
    print(i)

print("keys_to_optimize:")
for i in [n for n, p in model.named_parameters() if p in keys_to_optimize]:
    print(i)
