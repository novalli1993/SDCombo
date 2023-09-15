import torch
import collections

state_dict = collections.OrderedDict()

# mobilenet_v3_large-8738ca79.pth deeplabv3_mobilenet_v3_large-fc3c493d.pth
pthFile = torch.load('../../work_dir/model/model_20230819_153749_9.pth')
# pthFile_a = torch.load('../../work_dir/model/model_20230819_020022_4.pth')

zeros = {}
for i in pthFile["model"].keys():
    for j in pthFile["model"][i].reshape(-1):
        zeros[i] = 0
        if j.item() == 0:
            zeros[i] += 1
    if zeros[i] !=0:
        print(i, zeros[i])
