import math

from train_HHA import *
from Backup.dataset_Stanford2D3D import *


def cam_mask(mask, palette, n):
    seg_img = np.zeros((np.shape(mask)[0], np.shape(mask)[1], 3))
    for c in range(n):
        seg_img[:, :, 0] += ((mask[:, :] == c) * (palette[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((mask[:, :] == c) * (palette[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((mask[:, :] == c) * (palette[c][2])).astype('uint8')
    colorized_mask = Image.fromarray(np.uint8(seg_img))
    return colorized_mask


def confusion_matrix(label_true, label_pred, n_class=15):
    result = torch.zeros((n_class, n_class), dtype=torch.int64, device=label_true.device)
    k = (label_true >= 0) & (label_true < n_class)
    inds = n_class * label_true[k].to(torch.int64) + label_pred[k]
    result += torch.bincount(inds, minlength=n_class ** 2).reshape(n_class, n_class)
    return result


def iou(confusion_matrix):
    iu = 100 * torch.diag(confusion_matrix) / (
            confusion_matrix.sum(1) + confusion_matrix.sum(0) - torch.diag(confusion_matrix))
    return iu


def miou(iu):
    result = []
    for i in iu:
        if not math.isnan(i):
            result.append(i)
    result = sum(result) / len(result)
    return result


device = 'cuda'

dataset = "J:/Dataset/Stanford2D3D"
area = ["area_1", "area_2", "area_3", "area_4", "area_5", "area_6"]
types = ["rgb", "semantic", "depth", "predict"]

weight = "J:/Model/SDCombo/work_dir/model/model_20230811_153904_8.pth"

select = {'area': 0, 'pic_num': 0}
file_name_prefix = os.listdir(os.path.join(dataset, area[select['area']], types[0]))[select['pic_num']][:-7]
file_name_suffix = '.png'
rgb = os.path.join(dataset, area[select['area']], types[0], file_name_prefix + types[0] + file_name_suffix)
semantic = os.path.join(dataset, area[select['area']], types[1], file_name_prefix + types[1] + file_name_suffix)
depth = os.path.join(dataset, area[select['area']], types[2], file_name_prefix + types[2] + file_name_suffix)

# image format
rgb = Image.open(rgb)
semantic = Image.open(semantic)
depth = Image.open(depth)
data_transform = PipelineEval(1080, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
image_tensor, sematic_tensor, depth_tensor = data_transform(rgb, semantic, depth)
image_tensor = torch.unsqueeze(image_tensor, dim=0)
sematic_tensor = torch.unsqueeze(sematic_tensor, dim=0)
depth_tensor = torch.unsqueeze(depth_tensor, dim=0)
image_tensor = image_tensor.to(device)
sematic_tensor = sematic_tensor.to(device)
depth_tensor = depth_tensor.to(device)

# creat model
model, _, _ = create_model(weight, 14)
model.to(device)

# inference
model.eval()
with torch.no_grad():
    output = model(image_tensor, depth_tensor)
    prediction = output.argmax(1).squeeze(0)
    mask = prediction.to("cpu").numpy().astype(np.uint8)

# save the prediction
save_path = "Demo"
if not os.path.exists(save_path):
    os.mkdir(save_path)
cam_mask(mask, PALETTE, 14).save(os.path.join(save_path, file_name_prefix + types[3] + file_name_suffix))

# confusion matrix
label_true = sematic_tensor.flatten()
label_pred = prediction.flatten()
con_mat = confusion_matrix(label_true, label_pred, 14)
print("Confusion Matrix: \n", con_mat)
iu = iou(con_mat)
print("IoU: ", iu)
print("mIoU: ", miou(iu))

# print the matrix
# con_mat = con_mat.cpu().numpy()
# plt.matshow(con_mat, cmap=plt.cm.Reds)
# plt.colorbar()
#
# for i in range(len(con_mat)):
#     for j in range(len(con_mat)):
#         plt.annotate(con_mat[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
#
# # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
#
# # plt.ylabel('True label')
# # plt.xlabel('Predicted label')
#
# plt.ylabel('True label'
#            , fontdict={'family': 'Times New Roman', 'size': 20}
#            ) # 设置字体大小。
# plt.xlabel('Predicted label'
#            , fontdict={'family': 'Times New Roman', 'size': 20}
#            )
# plt.xticks(range(len(con_mat)), labels=CLASSES)
# plt.yticks(range(len(con_mat)), labels=CLASSES)
# plt.show()
