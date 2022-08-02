
import torch
import cv2
import numpy as np
import ncnn_utils as ncnn_utils
from my_tests.mmdet_nets import CSPResNet, load_ckpt, PPYOLOE, CustomCSPPAN, PPYOLOEHead
from my_tests.mmdet_trans import ResizeImage, NormalizeImage, Permute
from my_tests.mmdet_pre import PPYOLOEValTransform

depth_mult = 0.33
width_mult = 0.50
backbone = dict(
    layers=[3, 6, 6, 3],
    channels=[64, 128, 256, 512, 1024],
    # return_idx=[3],
    return_idx=[1, 2, 3],
    use_large_stem=True,
    depth_mult=depth_mult,
    width_mult=width_mult,
)
fpn2 = dict(
    in_channels=[int(256 * width_mult), int(512 * width_mult), int(1024 * width_mult)],
    out_channels=[768, 384, 192],
    stage_num=1,
    block_num=3,
    act='swish',
    spp=True,
    depth_mult=depth_mult,
    width_mult=width_mult,
)
head2 = dict(
    in_channels=[int(768 * width_mult), int(384 * width_mult), int(192 * width_mult)],
    fpn_strides=[32, 16, 8],
    grid_cell_scale=5.0,
    grid_cell_offset=0.5,
    static_assigner_epoch=100,
    use_varifocal_loss=True,
    num_classes=80,
    loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5, },
    eval_size=(640, 640),
)
bb = CSPResNet(**backbone)
fpn = CustomCSPPAN(**fpn2)
head = PPYOLOEHead(static_assigner=None, assigner=None, nms_cfg=None, **head2)
model = PPYOLOE(bb, fpn, head)
# model = PPYOLOE(bb, fpn, None)
# model = PPYOLOE(bb, None, None)
ckpt = torch.load('ppyoloe_crn_s_300e_coco.pth', map_location="cpu")
model = load_ckpt(model, ckpt["model"])


model.eval()
torch.save(model.state_dict(), "06.pth")

bp = open('06_pncnn.bin', 'wb')
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

# Debug
# pp = pp.replace(bottom_names[-1], 'output')

pp = pp.replace(bottom_names[0], 'cls_score')
pp = pp.replace(bottom_names[1], 'reg_dist')

pp = '7767517\n%d %d\n'%(layer_id, tensor_id) + pp
with open('06_pncnn.param', 'w', encoding='utf-8') as f:
    f.write(pp)
    f.close()



# 预测时的数据预处理
context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
to_rgb = True
target_size = 640

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

img = cv2.imread('000000000019.jpg')

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
dic['y'] = y[-1].cpu().detach().numpy()
# dic['y2'] = y[-2].cpu().detach().numpy()
# dic['y3'] = y[-3].cpu().detach().numpy()


np.savez('06', **dic)
print()
