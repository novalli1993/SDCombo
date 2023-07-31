import torch
import collections

state_dict = collections.OrderedDict()

pthFile = torch.load('../../work_dir/model/upernet_internimage_t_512_160k_ade20k.pth')


for i in pthFile.keys():
    print("|-",i,type(pthFile[i]))
    if isinstance(pthFile[i],dict):
        for j in pthFile[i].keys():
            print(j)
            # print(pthFile[i][j])
    else:
        print(pthFile[i])
