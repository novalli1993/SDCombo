import torch
import collections

state_dict = collections.OrderedDict()

# mobilenet_v3_large-8738ca79.pth deeplabv3_mobilenet_v3_large-fc3c493d.pth
pthFile = torch.load('../../work_dir/model/model_DL_20230831_070701_18.pth')
# pthFile_a = torch.load('../../work_dir/model/model_20230819_020022_4.pth') deeplabv3_mobilenet_v3_large-fc3c493d.pth


# for i in pthFile.keys():
#     print("-",i)
#     if isinstance(pthFile[i],dict):
#         for j in pthFile[i].keys():
#             print("  -",j)
#             # print(torch.equal(pthFile[i][j],pthFile_a[i][j]))
#     else:
#         print(pthFile[i])

print(pthFile["model"].keys())
