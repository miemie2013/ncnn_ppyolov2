
import torch
import cv2
import numpy as np
import ncnn_utils as ncnn_utils
from my_tests.mmdet_nets import LRELU
from my_tests.mmdet_trans import ResizeImage, NormalizeImage, Permute
from my_tests.mmdet_pre import PPYOLOEValTransform

model = LRELU()
model.eval()
torch.save(model.state_dict(), "07.pth")

bp = open('07_pncnn.bin', 'wb')
pp = ''
layer_id = 0
tensor_id = 0
pp += 'Input\tlayer_%.8d\t0 1 tensor_%.8d\n' % (layer_id, tensor_id)
layer_id += 1
tensor_id += 1

ncnn_data = {}
ncnn_data['bp'] = bp
ncnn_data['pp'] = pp
ncnn_data['layer_id'] = layer_id
ncnn_data['tensor_id'] = tensor_id
bottom_names = ncnn_utils.newest_bottom_names(ncnn_data)
bottom_names = model.export_ncnn(ncnn_data, bottom_names)

# 如果1个张量作为了n(n>1)个层的输入张量，应该用Split层将它复制n份，每1层用掉1个。
bottom_names = ncnn_utils.split_input_tensor(ncnn_data, bottom_names)
pp = ncnn_data['pp']
layer_id = ncnn_data['layer_id']
tensor_id = ncnn_data['tensor_id']
pp = pp.replace('tensor_%.8d' % (0,), 'images')
pp = pp.replace(bottom_names[-1], 'output')
pp = '7767517\n%d %d\n'%(layer_id, tensor_id) + pp
with open('07_pncnn.param', 'w', encoding='utf-8') as f:
    f.write(pp)
    f.close()



# 预测时的数据预处理
context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
to_rgb = True
target_size = 6

# NormalizeImage
normalizeImage2 = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    is_scale=True,
    is_channel_first=False,
)
# Permute
permute2 = dict(
    to_bgr=False,
    channel_first=True,
)
# ResizeImage
resizeImage2 = dict(
    target_size=640,
    interp=2,
)


resizeImage = ResizeImage(target_size=target_size, interp=resizeImage2['interp'])
normalizeImage = NormalizeImage(**normalizeImage2)
permute = Permute(**permute2)
preproc = PPYOLOEValTransform(context, to_rgb, resizeImage, normalizeImage, permute)

img = cv2.imread('my_test.jpg')

img_info = {"id": 0}
height, width = img.shape[:2]
img_info["height"] = height
img_info["width"] = width
img_info["raw_img"] = img

img, scale_factor = preproc(img)

x = torch.from_numpy(img)
x = x.to(torch.float32)
x.requires_grad_(False)

y = model(x)


dic = {}
dic['x'] = x.cpu().detach().numpy()
dic['y'] = y.cpu().detach().numpy()


np.savez('07', **dic)
print()
