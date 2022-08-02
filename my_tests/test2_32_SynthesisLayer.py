
import torch
import cv2
import numpy as np
import ncnn_utils as ncnn_utils
from my_tests.mmdet_nets import load_ckpt
from my_tests.mmgan_styleganv2ada import normalize_2nd_moment, normalize_2nd_moment2ncnn
from my_tests.mmgan_styleganv2ada import SynthesisLayer2, StyleGANv2ADA_SynthesisNetwork

w_dim = 512
z_dim = 512
c_dim = 0
# img_resolution = 512
img_resolution = 4
img_channels = 3
channel_base = 32768
channel_max = 512
num_fp16_res = 4
conv_clamp = 256



x_shape = [1, 512, 4, 4]
w_shape = [1, 512]
in_channels = 512
out_channels = 512
w_dim = 512
resolution = 4
kernel_size = 3
up = 1
use_noise = True
activation = 'lrelu'
resample_filter = [1, 3, 3, 1]
conv_clamp = 256
channels_last = False
fused_modconv = True
gain = 1

# x_shape = [1, 512, 4, 4]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 8
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1

# x_shape = [1, 512, 8, 8]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 8
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 8, 8]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 16
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 16, 16]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 16
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 16, 16]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 32
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 32, 32]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 32
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 32, 32]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 64
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 64, 64]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 512
# w_dim = 512
# resolution = 64
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 512, 64, 64]
# w_shape = [1, 512]
# in_channels = 512
# out_channels = 256
# w_dim = 512
# resolution = 128
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 256, 128, 128]
# w_shape = [1, 512]
# in_channels = 256
# out_channels = 256
# w_dim = 512
# resolution = 128
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 256, 128, 128]
# w_shape = [1, 512]
# in_channels = 256
# out_channels = 128
# w_dim = 512
# resolution = 256
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 128, 256, 256]
# w_shape = [1, 512]
# in_channels = 128
# out_channels = 128
# w_dim = 512
# resolution = 256
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 128, 256, 256]
# w_shape = [1, 512]
# in_channels = 128
# out_channels = 64
# w_dim = 512
# resolution = 512
# kernel_size = 3
# up = 2
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1
#
# x_shape = [1, 64, 512, 512]
# w_shape = [1, 512]
# in_channels = 64
# out_channels = 64
# w_dim = 512
# resolution = 512
# kernel_size = 3
# up = 1
# use_noise = True
# activation = 'lrelu'
# resample_filter = [1, 3, 3, 1]
# conv_clamp = 256
# channels_last = False
# fused_modconv = True
# gain = 1


use_fp16 = True
# in_channels = 8
# out_channels = 2

from my_tests.mmgan_styleganv2ada import modulated_conv2d2ncnn, conv2d_resample2ncnn, upfirdn2d2ncnn, _conv2d_wrapper2ncnn
from my_tests.mmgan_styleganv2ada import modulated_conv2d, conv2d_resample, upfirdn2d, _conv2d_wrapper, upsample2d, downsample2d
from my_tests.mmgan_styleganv2ada import SynthesisBlock, SynthesisLayer, ToRGBLayer, FullyConnectedLayer
from my_tests.mmgan_styleganv2ada import StyleGANv2ADA_MappingNetwork, StyleGANv2ADA_SynthesisNetwork
use_noise = True
model = SynthesisLayer2(in_channels, out_channels, w_dim, resolution,
                       kernel_size, up, use_noise, activation, resample_filter, conv_clamp, channels_last, use_fp16=use_fp16)
model.eval()
torch.save(model.state_dict(), "32.pth")

bp = open('32_pncnn.bin', 'wb')
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
bottom_names = model.export_ncnn(ncnn_data, bottom_names, use_fp16, fused_modconv=fused_modconv, gain=gain)


# 如果1个张量作为了n(n>1)个层的输入张量，应该用Split层将它复制n份，每1层用掉1个。
bottom_names = ncnn_utils.split_input_tensor(ncnn_data, bottom_names)
pp = ncnn_data['pp']
layer_id = ncnn_data['layer_id']
tensor_id = ncnn_data['tensor_id']
pp = pp.replace('tensor_%.8d' % (0,), 'images')
pp = pp.replace(bottom_names[-1], 'output')
pp = '7767517\n%d %d\n'%(layer_id, tensor_id) + pp
with open('32_pncnn.param', 'w', encoding='utf-8') as f:
    f.write(pp)
    f.close()




dic2 = np.load('30.npz')
ws = dic2['y']
ws = torch.from_numpy(ws)
ws = ws.float()
ws = ws.squeeze(0)
ws.requires_grad_(False)
ws = ws.cuda()
model = model.cuda()

noise_mode = 'const'
y = model(None, ws, noise_mode=noise_mode, fused_modconv=fused_modconv, gain=gain)


dic = {}
dic['ws'] = ws.cpu().detach().numpy()
dic['y'] = y.cpu().detach().numpy()

np.savez('32', **dic)
print()
