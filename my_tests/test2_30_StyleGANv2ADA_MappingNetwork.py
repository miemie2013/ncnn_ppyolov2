
import torch
import cv2
import numpy as np
import ncnn_utils as ncnn_utils
from my_tests.mmdet_nets import load_ckpt
from my_tests.mmgan_styleganv2ada import normalize_2nd_moment, normalize_2nd_moment2ncnn
from my_tests.mmgan_styleganv2ada import FullyConnectedLayer, StyleGANv2ADA_MappingNetwork
from my_tests.mmgan_styleganv3 import FullyConnectedLayer, StyleGANv3_MappingNetwork



# in_channels = 512
in_channels = 256
w_dim = 512
lr = 0.1
activation = 'linear'
activation = 'lrelu'
activation = 'relu'
activation = 'tanh'
# activation = 'sigmoid'
# activation = 'elu'
# activation = 'selu'
# activation = 'softplus'
# activation = 'swish'


# mapping_ema = FullyConnectedLayer(w_dim, in_channels, activation=activation, bias_init=1)

# mapping = dict(
#     z_dim=512,
#     c_dim=0,
#     w_dim=512,
#     num_layers=8,
# )
# mapping['num_ws'] = 1
# mapping_ema = StyleGANv2ADA_MappingNetwork(**mapping)


mapping = dict(
    z_dim=512,
    c_dim=0,
    w_dim=512,
    num_layers=2,
)
mapping['num_ws'] = 1
mapping_ema = StyleGANv3_MappingNetwork(**mapping)
mapping_ema.eval()

# ckpt = torch.load('styleganv2ada_512_afhqcat.pth', map_location="cpu")
# ckpt = torch.load('65.pth', map_location="cpu")
# ckpt = torch.load('stylegan3_r_afhqv2_512.pth', map_location="cpu")
ckpt = torch.load('77.pth', map_location="cpu")
mapping_ema = load_ckpt(mapping_ema, ckpt["mapping_ema"])

torch.save(mapping_ema.state_dict(), "30.pth")

bp = open('30_pncnn.bin', 'wb')
pp = ''
layer_id = 0
tensor_id = 0
pp += 'Input\tlayer_%.8d\t0 2 tensor_%.8d tensor_%.8d\n' % (layer_id, tensor_id, tensor_id + 1)
layer_id += 1
tensor_id += 2

ncnn_data = {}
ncnn_data['bp'] = bp
ncnn_data['pp'] = pp
ncnn_data['layer_id'] = layer_id
ncnn_data['tensor_id'] = tensor_id
bottom_names = ['tensor_%.8d' % (tensor_id - 2,), 'tensor_%.8d' % (tensor_id - 1,)]
bottom_names = mapping_ema.export_ncnn(ncnn_data, bottom_names)


# 如果1个张量作为了n(n>1)个层的输入张量，应该用Split层将它复制n份，每1层用掉1个。
bottom_names = ncnn_utils.split_input_tensor(ncnn_data, bottom_names)
pp = ncnn_data['pp']
layer_id = ncnn_data['layer_id']
tensor_id = ncnn_data['tensor_id']
pp = pp.replace('tensor_%.8d' % (0,), 'z')
pp = pp.replace('tensor_%.8d' % (1,), 'coeff')
pp = pp.replace(bottom_names[-1], 'output')
pp = '7767517\n%d %d\n'%(layer_id, tensor_id) + pp
with open('30_pncnn.param', 'w', encoding='utf-8') as f:
    f.write(pp)
    f.close()



seed = 75
z_dim = 512
z = np.random.RandomState(seed).randn(1, z_dim)
z = torch.from_numpy(z)
z = z.float()
z.requires_grad_(False)

y = mapping_ema(z, None)
# y = mapping_ema(z)


dic = {}
dic['z'] = z.cpu().detach().numpy()
dic['y'] = y.cpu().detach().numpy()

import struct
seed_bin = open('seed_75.bin', 'wb')
s = struct.pack('i', 0)
seed_bin.write(s)
for i1 in range(z_dim):
    s = struct.pack('f', z[0][i1])
    seed_bin.write(s)

ws_bin = open('ws_75.bin', 'wb')
s = struct.pack('i', 0)
ws_bin.write(s)
for i1 in range(512):
    s = struct.pack('f', y[0][0][i1])
    ws_bin.write(s)

np.savez('30', **dic)
print()
