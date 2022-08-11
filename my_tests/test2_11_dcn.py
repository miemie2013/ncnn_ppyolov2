
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import collections
from itertools import repeat
import cv2
import numpy as np
import struct
import ncnn_utils as ncnn_utils


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

class MyDCNv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(MyDCNv2, self).__init__()
        assert groups == 1
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1]))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels, ))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, offset, mask):
        in_C = self.in_channels
        out_C = self.out_channels
        stride_h = self.stride[0]
        stride_w = self.stride[1]
        padding_h = self.padding[0]
        padding_w = self.padding[1]
        dilation_h = self.dilation[0]
        dilation_w = self.dilation[1]
        groups = self.groups
        N, _, H, W = x.shape
        _, w_in, kH, kW = self.weight.shape

        kernel_extent_w = dilation_w * (kW - 1) + 1
        kernel_extent_h = dilation_h * (kH - 1) + 1
        out_W = (W + padding_w * 2 - kernel_extent_w) / stride_w + 1
        out_H = (H + padding_h * 2 - kernel_extent_h) / stride_h + 1
        out_W = int(out_W)
        out_H = int(out_H)

        # ================== 1.先对图片x填充得到填充后的图片pad_x ==================
        pad_x_H = H + padding_h * 2
        pad_x_W = W + padding_w * 2
        pad_x = torch.zeros((N, in_C, pad_x_H, pad_x_W), dtype=torch.float32, device=x.device)
        pad_x[:, :, padding_h:padding_h + H, padding_w:padding_w + W] = x

        # ================== 2.求所有采样点的坐标 ==================
        # 卷积核左上角在pad_x中的位置
        y_outer, x_outer = torch.meshgrid([torch.arange(out_H, device=x.device), torch.arange(out_W, device=x.device)])
        y_outer = y_outer * stride_h
        x_outer = x_outer * stride_w
        start_pos_yx = torch.stack((y_outer, x_outer), 2).float()       # [out_H, out_W, 2]         仅仅是卷积核左上角在pad_x中的位置
        start_pos_yx = start_pos_yx.unsqueeze(0).unsqueeze(3)           # [1, out_H, out_W, 1, 2]   仅仅是卷积核左上角在pad_x中的位置
        start_pos_yx = torch.tile(start_pos_yx, [N, 1, 1, kH * kW, 1])  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核左上角在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核左上角在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核左上角在pad_x中的位置
        start_pos_y.requires_grad = False
        start_pos_x.requires_grad = False

        # 卷积核内部的偏移
        y_inner, x_inner = torch.meshgrid([torch.arange(kH, device=x.device), torch.arange(kW, device=x.device)])
        y_inner = y_inner * dilation_h
        x_inner = x_inner * dilation_w
        filter_inner_offset_yx = torch.stack((y_inner, x_inner), 2).float()                    # [kH, kW, 2]       卷积核内部的偏移
        filter_inner_offset_yx = torch.reshape(filter_inner_offset_yx, (1, 1, 1, kH * kW, 2))  # [1, 1, 1, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_yx = torch.tile(filter_inner_offset_yx, [N, out_H, out_W, 1, 1])  # [N, out_H, out_W, kH*kW, 2]   卷积核内部的偏移
        filter_inner_offset_y = filter_inner_offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_x = filter_inner_offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   卷积核内部的偏移
        filter_inner_offset_y.requires_grad = False
        filter_inner_offset_x.requires_grad = False

        # 预测的偏移
        offset = offset.permute(0, 2, 3, 1)   # [N, out_H, out_W, kH*kW*2]
        offset_yx = torch.reshape(offset, (N, out_H, out_W, kH * kW, 2))  # [N, out_H, out_W, kH*kW, 2]
        offset_y = offset_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]
        offset_x = offset_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]

        # 最终采样位置。
        pos_y = start_pos_y + filter_inner_offset_y + offset_y  # [N, out_H, out_W, kH*kW, 1]
        pos_x = start_pos_x + filter_inner_offset_x + offset_x  # [N, out_H, out_W, kH*kW, 1]
        # 限制采样点位置。如果调用F.grid_sample()，已经在F.grid_sample()里限制了，就注释掉。
        # pos_y = torch.clamp(pos_y, -1.0, pad_x_H)  # 最终采样位置限制在-1到pad_x_H之间。（看DCN C++源码得知）
        # pos_x = torch.clamp(pos_x, -1.0, pad_x_W)  # 最终采样位置限制在-1到pad_x_H之间。（看DCN C++源码得知）

        # ================== 3.采样。用F.grid_sample()双线性插值采样。 ==================
        pos_x = pos_x / (pad_x_W - 1) * 2.0 - 1.0
        pos_y = pos_y / (pad_x_H - 1) * 2.0 - 1.0
        xtyt = torch.cat([pos_x, pos_y], -1)  # [N, out_H, out_W, kH*kW, 2]
        xtyt = torch.reshape(xtyt, (N, out_H, out_W * kH * kW, 2))  # [N, out_H, out_W*kH*kW, 2]
        value = F.grid_sample(pad_x, xtyt, mode='bilinear', padding_mode='zeros', align_corners=True)  # [N, in_C, out_H, out_W*kH*kW]
        value = torch.reshape(value, (N, in_C, out_H, out_W, kH * kW))    # [N, in_C, out_H, out_W, kH * kW]
        value = value.permute(0, 1, 4, 2, 3)                              # [N, in_C, kH * kW, out_H, out_W]

        # ================== 4.乘以重要程度 ==================
        # 乘以重要程度
        mask = mask.unsqueeze(1)            # [N,    1, kH * kW, out_H, out_W]
        value = value * mask                # [N, in_C, kH * kW, out_H, out_W]
        new_x = torch.reshape(value, (N, in_C * kH * kW, out_H, out_W))  # [N, in_C * kH * kW, out_H, out_W]

        # ================== 5.乘以本层的权重，加上偏置 ==================
        # 1x1卷积
        rw = torch.reshape(self.weight, (out_C, w_in * kH * kW, 1, 1))  # [out_C, w_in, kH, kW] -> [out_C, w_in*kH*kW, 1, 1]  变成1x1卷积核
        out = F.conv2d(new_x, rw, bias=self.bias, stride=1, groups=groups)  # [N, out_C, out_H, out_W]
        return out



class ConvNormLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 dcn_v2=False):
        super(ConvNormLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.dcn_v2 = dcn_v2

        if not self.dcn_v2:
            self.conv = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=False)
            self.conv_w_lr = lr
            # 初始化权重
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)
        else:
            self.offset_channel = 2 * filter_size ** 2
            self.mask_channel = filter_size ** 2

            self.conv_offset = nn.Conv2d(
                in_channels=ch_in,
                out_channels=3 * filter_size ** 2,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                bias=True)
            # 初始化权重
            torch.nn.init.constant_(self.conv_offset.weight, 0.0)
            torch.nn.init.constant_(self.conv_offset.bias, 0.0)

            # 自实现的DCNv2
            # self.conv = MyDCNv2(
            #     in_channels=ch_in,
            #     out_channels=ch_out,
            #     kernel_size=filter_size,
            #     stride=stride,
            #     padding=(filter_size - 1) // 2,
            #     dilation=1,
            #     groups=groups,
            #     bias=True)
            # 官方DCN
            self.conv = torchvision.ops.DeformConv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                dilation=1,
                groups=groups,
                bias=True)

            self.dcn_w_lr = lr
            # 初始化权重
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)

        self.freeze_norm = freeze_norm
        norm_lr = 0. if freeze_norm else lr
        self.norm_lr = norm_lr
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset = offset_mask[:, :self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel:, :, :]
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)
        # return offset, mask, out
        return out

    def export_ncnn(self, ncnn_data, bottom_names):
        if not self.dcn_v2:
            pass
        else:
            offset_mask = ncnn_utils.conv2d(ncnn_data, bottom_names, self.conv_offset)
            offset = ncnn_utils.crop(ncnn_data, offset_mask, starts='1,%d' % (0,), ends='1,%d' % (self.offset_channel,), axes='1,0')
            mask = ncnn_utils.crop(ncnn_data, offset_mask, starts='1,%d' % (self.offset_channel,), ends='1,%d' % (self.offset_channel + self.mask_channel,), axes='1,0')
            mask = ncnn_utils.activation(ncnn_data, mask, act_name='sigmoid')
            out = ncnn_utils.deformable_conv2d(ncnn_data, [bottom_names[0], offset[0], mask[0]], self.conv)
        return out
        # return mask



class ConvNormLayer2(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 stride,
                 groups=1,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=True,
                 lr=1.0,
                 dcn_v2=False):
        super(ConvNormLayer2, self).__init__()
        assert norm_type in ['bn', 'sync_bn']
        self.norm_type = norm_type
        self.dcn_v2 = dcn_v2

        if not self.dcn_v2:
            self.conv = nn.Conv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                groups=groups,
                bias=False)
            self.conv_w_lr = lr
            # 初始化权重
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)
        else:
            self.offset_channel = 2 * filter_size ** 2
            self.mask_channel = filter_size ** 2

            self.conv_offset = nn.Conv2d(
                in_channels=ch_in,
                out_channels=3 * filter_size ** 2,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                bias=True)
            # 初始化权重
            torch.nn.init.constant_(self.conv_offset.weight, 0.0)
            torch.nn.init.constant_(self.conv_offset.bias, 0.0)

            # 自实现的DCNv2
            self.conv = MyDCNv2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=filter_size,
                stride=stride,
                padding=(filter_size - 1) // 2,
                dilation=1,
                groups=groups,
                bias=True)
            # 官方DCN
            # self.conv = torchvision.ops.DeformConv2d(
            #     in_channels=ch_in,
            #     out_channels=ch_out,
            #     kernel_size=filter_size,
            #     stride=stride,
            #     padding=(filter_size - 1) // 2,
            #     dilation=1,
            #     groups=groups,
            #     bias=True)

            self.dcn_w_lr = lr
            # 初始化权重
            torch.nn.init.xavier_normal_(self.conv.weight, gain=1.)

        self.freeze_norm = freeze_norm
        norm_lr = 0. if freeze_norm else lr
        self.norm_lr = norm_lr
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset = offset_mask[:, :self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel:, :, :]
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)
        return out


img_h = 2
img_w = 2

img_h = 8
img_w = 8





# ch_in = 3
# ch_out = 64

ch_in = 1
ch_out = 1

# ch_in = 1
# ch_out = 4

# ch_in = 1
# ch_out = 8

# ch_in = 1
# ch_out = 16

# ch_in = 4
# ch_out = 1

# ch_in = 4
# ch_out = 4

ch_in = 4
ch_out = 8

# ch_in = 4
# ch_out = 16

# ch_in = 8
# ch_out = 1

# ch_in = 8
# ch_out = 4

# ch_in = 8
# ch_out = 8

# ch_in = 8
# ch_out = 16
#
# ch_in = 16
# ch_out = 1
#
# ch_in = 16
# ch_out = 4
#
# ch_in = 16
# ch_out = 8
#
# ch_in = 16
# ch_out = 16



ch_in *= 3
ch_out *= 3


filter_size = 1
stride = 1

# filter_size = 1
# stride = 2

# filter_size = 2
# stride = 1

# filter_size = 2
# stride = 2

# filter_size = 3
# stride = 1

filter_size = 3
stride = 2

# filter_size = 4
# stride = 1

# filter_size = 4
# stride = 2

# filter_size = 5
# stride = 1

# filter_size = 5
# stride = 5


'''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
即为DCN前向代码。


gn:

https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/group_norm_op.h

class GroupNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
即为gn前向代码。



https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/group_norm_op.cu

https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/group_norm_op.cc

'''
model = ConvNormLayer(ch_in, ch_out, filter_size=filter_size, stride=stride, dcn_v2=True)
torch.nn.init.normal_(model.conv_offset.weight)
torch.nn.init.normal_(model.conv_offset.bias)
torch.nn.init.normal_(model.conv.weight)
torch.nn.init.normal_(model.conv.bias)
# torch.nn.init.normal_(model.norm.weight)
# torch.nn.init.normal_(model.norm.bias)
# torch.nn.init.normal_(model.norm.running_mean)
# torch.nn.init.constant_(model.norm.running_var, 2.3)
model.eval()
state_dict = model.state_dict()
torch.save(state_dict, "11.pth")

model2 = ConvNormLayer2(ch_in, ch_out, filter_size=filter_size, stride=stride, dcn_v2=True)
model2.eval()
model2.load_state_dict(state_dict)

dic = {}


bp = open('11_pncnn.bin', 'wb')
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
with open('11_pncnn.param', 'w', encoding='utf-8') as f:
    f.write(pp)
    f.close()



# aaaaaaaaa = cv2.imread('my_test32.jpg')
# aaaaaaaaa = cv2.imread('my_test9_7.jpg')
# aaaaaaaaa = cv2.imread('my_test5_3.jpg')
aaaaaaaaa = cv2.imread('my_test2.jpg')
aaaaaaaaa = aaaaaaaaa.astype(np.float32)

mean = [117.3, 126.5, 130.2]
std = [108.4, 117.3, 127.6]
mean = np.array(mean)[np.newaxis, np.newaxis, :]
std = np.array(std)[np.newaxis, np.newaxis, :]
aaaaaaaaa -= mean
aaaaaaaaa /= std


x = torch.from_numpy(aaaaaaaaa)
x = x.to(torch.float32)
x = x.permute((2, 0, 1))
x = torch.unsqueeze(x, 0)
x.requires_grad_(False)


dummy_input = torch.randn(1, ch_in, img_h, img_w)
x = dummy_input

# offset, mask, y = model(x)
y = model(x)
y2 = model2(x)
dic['x'] = x.cpu().detach().numpy()
dic['y'] = y.cpu().detach().numpy()
# dic['offset'] = offset.cpu().detach().numpy()
# dic['mask'] = mask.cpu().detach().numpy()


aaa = dic['x']
print(aaa)
aaa = np.reshape(aaa, (-1, ))
print(aaa)

seed_bin = open('../build/examples/in0.bin', 'wb')
s = struct.pack('i', 0)
seed_bin.write(s)
for i1 in range(aaa.shape[0]):
    s = struct.pack('f', aaa[i1])
    seed_bin.write(s)


yyy1 = y.cpu().detach().numpy()
yyy2 = y2.cpu().detach().numpy()
ddd = np.sum((yyy1 - yyy2) ** 2)
print('ddd=%.9f' % ddd)


np.savez('11', **dic)

mod = torch.jit.trace(model, x)
mod.save("11.pt")


print()
