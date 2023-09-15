import torch
import collections

state_dict = collections.OrderedDict()

weight = '../../work_dir/model/model_20230830_211613_9.pth'

weight_cls_model = torch.load(weight)
for key, value in weight_cls_model["model"].items():
    if key.split('.')[0] == "backbone":
        key = "features"+key[8:]
        state_dict[key] = value
    elif key.split('.')[0] == "aux_classifier":
        pass
for i in state_dict.keys():
    print(i)

torch.save(state_dict, '../../work_dir/model/DL_model_DL_20230827_185154_8.pth')
