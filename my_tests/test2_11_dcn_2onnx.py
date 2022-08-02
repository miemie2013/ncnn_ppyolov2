
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np


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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        self.groups = groups

        filter_shape = [out_channels, in_channels // groups, kernel_size, kernel_size]

        self.weight = torch.nn.Parameter(torch.randn(filter_shape))
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels, ))

    def forward(self, x, offset, mask):
        in_C = self.in_channels
        out_C = self.out_channels
        stride = self.stride
        padding = self.padding
        # dilation = self.dilation
        groups = self.groups
        N, _, H, W = x.shape
        _, w_in, kH, kW = self.weight.shape
        out_W = (W + 2 * padding - (kW - 1)) // stride
        out_H = (H + 2 * padding - (kH - 1)) // stride

        # ================== 1.先对图片x填充得到填充后的图片pad_x ==================
        pad_x_H = H + padding * 2 + 1
        pad_x_W = W + padding * 2 + 1
        pad_x = torch.zeros((N, in_C, pad_x_H, pad_x_W), dtype=torch.float32, device=x.device)
        pad_x[:, :, padding:padding + H, padding:padding + W] = x

        # ================== 2.求所有采样点的坐标 ==================
        # 卷积核中心点在pad_x中的位置
        y_outer, x_outer = torch.meshgrid([torch.arange(out_H, device=x.device), torch.arange(out_W, device=x.device)])
        y_outer = y_outer * stride + padding
        x_outer = x_outer * stride + padding
        start_pos_yx = torch.stack((y_outer, x_outer), 2).float()       # [out_H, out_W, 2]         仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = start_pos_yx.unsqueeze(0).unsqueeze(3)           # [1, out_H, out_W, 1, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_yx = torch.tile(start_pos_yx, [N, 1, 1, kH * kW, 1])  # [N, out_H, out_W, kH*kW, 2]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y = start_pos_yx[:, :, :, :, :1]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_x = start_pos_yx[:, :, :, :, 1:]  # [N, out_H, out_W, kH*kW, 1]   仅仅是卷积核中心点在pad_x中的位置
        start_pos_y.requires_grad = False
        start_pos_x.requires_grad = False

        # 卷积核内部的偏移
        half_W = (kW - 1) // 2
        half_H = (kH - 1) // 2
        y_inner2, x_inner2 = torch.meshgrid([torch.arange(kH, device=x.device), torch.arange(kW, device=x.device)])
        y_inner = y_inner2 - half_H
        x_inner = x_inner2 - half_W
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
        pos_y = torch.clamp(pos_y, 0.0, H + padding * 2 - 1.0)  # 最终采样位置限制在pad_x内
        pos_x = torch.clamp(pos_x, 0.0, W + padding * 2 - 1.0)  # 最终采样位置限制在pad_x内

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

        global_stats = True if freeze_norm else None
        if norm_type in ['sync_bn', 'bn']:
            # ppdet中freeze_norm == True时，use_global_stats = global_stats = True， bn的均值和方差是不会变的！！！，
            # 而且训练时前向传播用的是之前统计均值和方差，而不是当前批次的均值和方差！（即训练时的前向传播就是预测时的前向传播）
            # 所以这里设置momentum = 0.0 让bn的均值和方差不会改变。并且model.train()之后要马上调用model.fix_bn()（让训练bn时的前向传播就是预测时bn的前向传播）
            momentum = 0.0 if freeze_norm else 0.1
            self.norm = nn.BatchNorm2d(ch_out, affine=True, momentum=momentum)
        norm_params = self.norm.parameters()

        if freeze_norm:
            for param in norm_params:
                param.requires_grad_(False)
        self.act = nn.LeakyReLU(0.33)

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset = offset_mask[:, :self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel:, :, :]
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)

        # if self.norm_type in ['bn', 'sync_bn']:
        #     out = self.norm(out)
        # out = self.act(out)
        return out



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

        global_stats = True if freeze_norm else None
        if norm_type in ['sync_bn', 'bn']:
            # ppdet中freeze_norm == True时，use_global_stats = global_stats = True， bn的均值和方差是不会变的！！！，
            # 而且训练时前向传播用的是之前统计均值和方差，而不是当前批次的均值和方差！（即训练时的前向传播就是预测时的前向传播）
            # 所以这里设置momentum = 0.0 让bn的均值和方差不会改变。并且model.train()之后要马上调用model.fix_bn()（让训练bn时的前向传播就是预测时bn的前向传播）
            momentum = 0.0 if freeze_norm else 0.1
            self.norm = nn.BatchNorm2d(ch_out, affine=True, momentum=momentum)
        norm_params = self.norm.parameters()

        if freeze_norm:
            for param in norm_params:
                param.requires_grad_(False)
        self.act = nn.LeakyReLU(0.33)

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset = offset_mask[:, :self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel:, :, :]
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)

        # if self.norm_type in ['bn', 'sync_bn']:
        #     out = self.norm(out)
        # out = self.act(out)
        return out


ch_in = 3
ch_out = 2

filter_size = 1
stride = 1

filter_size = 1
stride = 2

filter_size = 2
stride = 1

filter_size = 2
stride = 2

filter_size = 3
stride = 1

# filter_size = 3
# stride = 2
#
# filter_size = 4
# stride = 1
#
# filter_size = 4
# stride = 2
#
# filter_size = 5
# stride = 1
#
# filter_size = 5
# stride = 2


model = ConvNormLayer(ch_in, ch_out, filter_size=filter_size, stride=stride, dcn_v2=True)
# model = ConvNormLayer2(ch_in, ch_out, filter_size=filter_size, stride=stride, dcn_v2=True)
model.eval()
state_dict = torch.load('11.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

output_name = '11.onnx'
input = "images"
output = "output"
dynamic = True
opset = 11

dummy_input = torch.randn(1, 3, 32, 32)

# torch.onnx._export(
#     model,
#     dummy_input,
#     output_name,
#     input_names=[input],
#     output_names=[output],
#     dynamic_axes={input: {0: 'batch'},
#                   output: {0: 'batch'}} if dynamic else None,
#     opset_version=opset,
# )
mod = torch.jit.trace(model, dummy_input)
mod.save("11.pt")

print()
