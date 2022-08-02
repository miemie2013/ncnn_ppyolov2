

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import struct
import numpy as np
import ncnn_utils as ncnn_utils
from loguru import logger








def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            logger.warning(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            logger.warning(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)
    return model


class DropBlock(torch.nn.Module):
    def __init__(self, block_size, keep_prob, name, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = torch.rand(x.shape, device=x.device)
            matrix = (matrix < gamma).float()
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y
        # return x




def identity(x):
    return x

def mish(x):
    return F.mish(x) if hasattr(F, mish) else x * F.tanh(F.softplus(x))


def swish(x):
    return x * torch.sigmoid(x)

TRT_ACT_SPEC = {'swish': swish}

ACT_SPEC = {'mish': mish, 'swish': swish}


def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return identity

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if trt and name in TRT_ACT_SPEC:
        fn = TRT_ACT_SPEC[name]
    elif name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)



class LRELU(nn.Module):
    def __init__(self):
        super(LRELU, self).__init__()
        # self.fc = nn.Linear(3, 2)
        lr_multiplier = 1.0
        out_features = 2
        in_features = 3
        bias_init = 2.5
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        # self.act = nn.LeakyReLU(0.95)
        # self.act_name = 'leaky_relu'

    def forward(self, x):
        # x = x.mean((2, 3), keepdim=False)
        #
        # w = self.weight.to(x.dtype) * self.weight_gain
        # b = self.bias
        # if b is not None:
        #     b = b.to(x.dtype)
        #     if self.bias_gain != 1:
        #         b = b * self.bias_gain
        #
        # x = torch.addmm(b.unsqueeze(0), x, w.t())
        # x = self.fc(x)

        # x = self.act(x)
        # x = F.softmax(x, dim=1)



        # x = F.interpolate(x, scale_factor=0.5, mode='bicubic')
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = x.permute((0, 2, 3, 1))
        x = x.permute((1, 0, 2, 3))
        # x = x.reshape((1, -1, 3))
        # x = x.sum([3, ])
        # x = x.permute((0, 2, 1))
        # batch_size = x.shape[0]
        # output_size = x.shape[2]
        # scale_x_y = 1.05
        # stride = 8


        # rows = torch.arange(0, output_size, dtype=torch.float32, device=x.device)
        # cols = torch.arange(0, output_size, dtype=torch.float32, device=x.device)
        # rows = rows[np.newaxis, np.newaxis, :, np.newaxis].repeat((1, output_size, 1, 1))
        # cols = cols[np.newaxis, :, np.newaxis, np.newaxis].repeat((1, 1, output_size, 1))
        # offset = torch.cat([rows, cols], dim=3)
        # offset = offset.repeat((batch_size, 1, 1, 1))
        # conv_raw_dxdy = x[:, :, :, :2]
        # pred_xy = (scale_x_y * torch.sigmoid(conv_raw_dxdy) + offset - (scale_x_y - 1.0) * 0.5) * stride
        return x

    def export_ncnn(self, ncnn_data, bottom_names):
        # bottom_names = ncnn_utils.activation(ncnn_data, bottom_names, self.act_name, args={'negative_slope': 0.95,})
        # bottom_names = ncnn_utils.softmax(ncnn_data, bottom_names, dim=1)
        # bottom_names = ncnn_utils.interpolate(ncnn_data, bottom_names, scale_factor=0.5, mode='bicubic')
        bottom_names = ncnn_utils.permute(ncnn_data, bottom_names, perm='(1, 0, 2, 3)')
        # bottom_names = ncnn_utils.reshape(ncnn_data, bottom_names, (1, -1, 3))
        return bottom_names



class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None,
                 act_name=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(ch_out)
        self.act_name = act_name
        if act is None or isinstance(act, (str, dict)):
            self.act = get_act_fn(act)
        else:
            self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

    def export_ncnn(self, ncnn_data, bottom_names):
        bottom_names = ncnn_utils.fuse_conv_bn(ncnn_data, bottom_names, self.conv, self.bn)
        bottom_names = ncnn_utils.activation(ncnn_data, bottom_names, self.act_name)
        return bottom_names


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', act_name='relu'):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act_name = act_name
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        if hasattr(self, 'conv'):
            raise NotImplementedError("not implemented.")
        else:
            # 看conv1分支，是卷积操作
            add_0 = self.conv1.export_ncnn(ncnn_data, bottom_names)

            # 看conv2分支，是卷积操作
            add_1 = self.conv2.export_ncnn(ncnn_data, bottom_names)

            # 最后是逐元素相加
            bottom_names = add_0 + add_1
            bottom_names = ncnn_utils.binaryOp(ncnn_data, bottom_names, op='Add')

        # 最后是激活
        bottom_names = ncnn_utils.activation(ncnn_data, bottom_names, self.act_name)
        return bottom_names


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', act_name='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act, act_name=act_name)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act, act_name=act_name)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y

    def export_ncnn(self, ncnn_data, bottom_names):
        if self.shortcut:
            add_0 = bottom_names

            # 看conv1层，是卷积操作
            y = self.conv1.export_ncnn(ncnn_data, bottom_names)
            # 看conv2层，是卷积操作
            y = self.conv2.export_ncnn(ncnn_data, y)

            # 最后是逐元素相加
            bottom_names = add_0 + y
            bottom_names = ncnn_utils.binaryOp(ncnn_data, bottom_names, op='Add')
        else:
            # 看conv1层，是卷积操作
            bottom_names = self.conv1.export_ncnn(ncnn_data, bottom_names)
            # 看conv2层，是卷积操作
            bottom_names = self.conv2.export_ncnn(ncnn_data, bottom_names)
        return bottom_names

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid', act_name='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act_name = act_name
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

    def export_ncnn(self, ncnn_data, bottom_names):
        # 看x_se分支，首先是mean操作，对应ncnn里的Reduction层
        x_se = ncnn_utils.reduction(ncnn_data, bottom_names, op='ReduceMean', input_dims=4, dims=(2, 3), keepdim=True)

        # 看x_se分支，然后是卷积操作
        x_se = ncnn_utils.conv2d(ncnn_data, x_se, self.fc)

        # 看x_se分支，然后是激活操作
        x_se = ncnn_utils.activation(ncnn_data, x_se, act_name=self.act_name)

        # 最后是逐元素相乘
        bottom_names = [bottom_names[0], x_se[0]]
        bottom_names = ncnn_utils.binaryOp(ncnn_data, bottom_names, op='Mul')
        return bottom_names


    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if isinstance(self.fc, torch.nn.Conv2d):
            if self.fc.weight.requires_grad:
                param_group_conv_weight = {'params': [self.fc.weight]}
                param_group_conv_weight['lr'] = base_lr * 1.0
                param_group_conv_weight['base_lr'] = base_lr * 1.0
                param_group_conv_weight['weight_decay'] = base_wd
                param_group_conv_weight['need_clip'] = need_clip
                param_group_conv_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight)
            if self.fc.bias.requires_grad:
                param_group_conv_bias = {'params': [self.fc.bias]}
                param_group_conv_bias['lr'] = base_lr * 1.0
                param_group_conv_bias['base_lr'] = base_lr * 1.0
                param_group_conv_bias['weight_decay'] = base_wd
                param_group_conv_bias['need_clip'] = need_clip
                param_group_conv_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias)


class CSPResStage2(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 act_name=None,
                 attn='eca'):
        super(CSPResStage2, self).__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act, act_name=act_name)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act, act_name=act_name)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act, act_name=act_name)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, ch_mid // 2, act=act, act_name=act_name, shortcut=True)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid', act_name='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act, act_name=act_name)

    def forward(self, x):
        x = x[:, :2, :4, :4]
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], 1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        # 切片操作Slice对应Crop
        bottom_names = ncnn_utils.crop(ncnn_data, bottom_names, starts='3,0,0,0', ends='3,2,4,4', axes='3,0,1,2')

        if self.conv_down is not None:
            bottom_names = self.conv_down.export_ncnn(ncnn_data, bottom_names)
        # 看conv1层，是卷积操作
        y1 = self.conv1.export_ncnn(ncnn_data, bottom_names)

        # 看conv2层，是卷积操作
        temp = self.conv2.export_ncnn(ncnn_data, bottom_names)
        for layer in self.blocks:
            temp = layer.export_ncnn(ncnn_data, temp)
        y2 = temp

        # concat
        bottom_names = y1 + y2
        bottom_names = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)

        if self.attn is not None:
            bottom_names = self.attn.export_ncnn(ncnn_data, bottom_names)
        bottom_names = self.conv3.export_ncnn(ncnn_data, bottom_names)
        return bottom_names

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if self.conv_down is not None:
            self.conv_down.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for layer in self.blocks:
            layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.attn is not None:
            self.attn.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv3.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class CSPResStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 act_name=None,
                 attn='eca'):
        super(CSPResStage, self).__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act, act_name=act_name)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act, act_name=act_name)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act, act_name=act_name)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, ch_mid // 2, act=act, act_name=act_name, shortcut=True)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid', act_name='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act, act_name=act_name)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], 1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        if self.conv_down is not None:
            bottom_names = self.conv_down.export_ncnn(ncnn_data, bottom_names)
        # 看conv1层，是卷积操作
        y1 = self.conv1.export_ncnn(ncnn_data, bottom_names)

        # 看conv2层，是卷积操作
        temp = self.conv2.export_ncnn(ncnn_data, bottom_names)
        for layer in self.blocks:
            temp = layer.export_ncnn(ncnn_data, temp)
        y2 = temp

        # concat
        bottom_names = y1 + y2
        bottom_names = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)

        if self.attn is not None:
            bottom_names = self.attn.export_ncnn(ncnn_data, bottom_names)
        bottom_names = self.conv3.export_ncnn(ncnn_data, bottom_names)
        return bottom_names

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if self.conv_down is not None:
            self.conv_down.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for layer in self.blocks:
            layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.attn is not None:
            self.attn.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv3.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class CSPResNet(nn.Module):
    __shared__ = ['width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='swish',
                 return_idx=[0, 1, 2, 3, 4],
                 depth_wise=False,
                 use_large_stem=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 freeze_at=-1,
                 trt=False):
        super(CSPResNet, self).__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        act_name = act
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act

        if use_large_stem:
            self.stem = nn.Sequential()
            self.stem.add_module('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act, act_name=act_name))
            self.stem.add_module('conv2', ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=act, act_name=act_name))
            self.stem.add_module('conv3', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act, act_name=act_name))
        else:
            self.stem = nn.Sequential()
            self.stem.add_module('conv1', ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=act, act_name=act_name))
            self.stem.add_module('conv2', ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=act, act_name=act_name))

        n = len(channels) - 1
        self.stages = nn.Sequential()
        for i in range(n):
            self.stages.add_module(str(i), CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=act, act_name=act_name))

        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at + 1, n)):
                self._freeze_parameters(self.stages[i])

    def _freeze_parameters(self, m):
        for p in m.parameters():
            p.requires_grad_(False)

    def forward(self, inputs):
        x = self.stem(inputs)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs

    def export_ncnn(self, ncnn_data, bottom_names):
        for layer in self.stem:
            bottom_names = layer.export_ncnn(ncnn_data, bottom_names)
        out_names = []
        for idx, stage in enumerate(self.stages):
            bottom_names = stage.export_ncnn(ncnn_data, bottom_names)
            if idx in self.return_idx:
                out_names.append(bottom_names[0])
        return out_names


class SPP(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 act='swish',
                 act_name='swish',
                 data_format='NCHW'):
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            name = 'pool{}'.format(i)
            pool = nn.MaxPool2d(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                    ceil_mode=False)
            self.add_module(name, pool)
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act, act_name=act_name)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == 'NCHW':
            y = torch.cat(outs, 1)
        else:
            y = torch.cat(outs, -1)

        y = self.conv(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        concat_input = [bottom_names[0]]
        for pool in self.pool:
            pool_out = ncnn_utils.pooling(ncnn_data, bottom_names, op='MaxPool', pool=pool)
            concat_input.append(pool_out[0])

        # concat
        if self.data_format == 'NCHW':
            bottom_names = ncnn_utils.concat(ncnn_data, concat_input, dim=1)
        else:
            bottom_names = ncnn_utils.concat(ncnn_data, concat_input, dim=3)
        bottom_names = self.conv.export_ncnn(ncnn_data, bottom_names)
        return bottom_names

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class CSPStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', act_name='swish', spp=False):
        super(CSPStage, self).__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act, act_name=act_name)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act, act_name=act_name)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_module(
                str(i),
                eval(block_fn)(next_ch_in, ch_mid, act=act, act_name=act_name, shortcut=False))
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act, act_name=act_name))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act, act_name=act_name)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], 1)
        y = self.conv3(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        # 看conv1分支，是卷积操作
        y1 = self.conv1.export_ncnn(ncnn_data, bottom_names)

        # 看conv2分支，是卷积操作
        y2 = self.conv2.export_ncnn(ncnn_data, bottom_names)
        for layer in self.convs:
            y2 = layer.export_ncnn(ncnn_data, y2)

        # concat
        bottom_names = y1 + y2
        bottom_names = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)

        bottom_names = self.conv3.export_ncnn(ncnn_data, bottom_names)
        return bottom_names

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for layer in self.convs:
            layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv3.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class CustomCSPPAN(nn.Module):
    __shared__ = ['norm_type', 'data_format', 'width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 norm_type='bn',
                 act='leaky',
                 stage_fn='CSPStage',
                 block_fn='BasicBlock',
                 stage_num=1,
                 block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 data_format='NCHW',
                 width_mult=1.0,
                 depth_mult=1.0,
                 trt=False):

        super(CustomCSPPAN, self).__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act_name = act
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        self.num_blocks = len(in_channels)
        self.data_format = data_format
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        fpn_stages = []
        fpn_routes = []
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   act_name=act_name,
                                   spp=(spp and i == 0)))

            if drop_block:
                stage.add_module('drop', DropBlock(block_size, keep_prob))

            fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act,
                        act_name=act_name))

            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act,
                    act_name=act_name))

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   act_name=act_name,
                                   spp=False))
            if drop_block:
                stage.add_module('drop', DropBlock(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def forward(self, blocks, for_mot=False):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], 1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], 1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]

    def export_ncnn(self, ncnn_data, bottom_names):
        blocks = bottom_names[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                bottom_names = route + [block, ]
                block = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)
                block = block[0]
            route = [block, ]
            for layer in self.fpn_stages[i]:
                route = layer.export_ncnn(ncnn_data, route)
            fpn_feats.append(route[0])

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i].export_ncnn(ncnn_data, route)
                route = ncnn_utils.interpolate(ncnn_data, route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i].export_ncnn(ncnn_data, [route, ])
            bottom_names = route + [block, ]
            block = ncnn_utils.concat(ncnn_data, bottom_names, dim=1)
            route = block
            for layer in self.pan_stages[i]:
                route = layer.export_ncnn(ncnn_data, route)
            route = route[0]
            pan_feats.append(route)

        return pan_feats[::-1]

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for i in range(self.num_blocks):
            for layer in self.fpn_stages[i]:
                layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            if i < self.num_blocks - 1:
                self.fpn_routes[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for i in reversed(range(self.num_blocks - 1)):
            self.pan_routes[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            for layer in self.pan_stages[i]:
                layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)



class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish', act_name='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act, act_name=act_name)

        self._init_weights()

    def _init_weights(self):
        pass
        # normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)

    def export_ncnn(self, ncnn_data, bottom_names):
        feat = bottom_names[0]
        avg_feat = bottom_names[1]

        branch_0 = ncnn_utils.conv2d(ncnn_data, [avg_feat, ], self.fc)
        weight = ncnn_utils.activation(ncnn_data, branch_0, 'sigmoid')

        # 然后是逐元素相乘
        bottom_names = ncnn_utils.binaryOp(ncnn_data, [feat, weight[0]], op='Mul')
        bottom_names = self.conv.export_ncnn(ncnn_data, bottom_names)
        return bottom_names



class PPYOLOEHead(nn.Module):
    __shared__ = ['num_classes', 'eval_size', 'trt', 'exclude_nms']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 nms_cfg=None,
                 exclude_nms=False):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.iou_loss = None
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.nms = nms
        # if isinstance(self.nms, MultiClassNMS) and trt:
        #     self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.nms_cfg = nms_cfg
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act_name = act
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act, act_name=act_name))
            self.stem_reg.append(ESEAttn(in_c, act=act, act_name=act_name))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for i in range(len(self.in_channels)):
            self.stem_cls[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            self.stem_reg[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            if self.pred_cls[i].weight.requires_grad:
                param_group_conv_weight = {'params': [self.pred_cls[i].weight]}
                param_group_conv_weight['lr'] = base_lr * 1.0
                param_group_conv_weight['base_lr'] = base_lr * 1.0
                param_group_conv_weight['weight_decay'] = base_wd
                param_group_conv_weight['need_clip'] = need_clip
                param_group_conv_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight)
            if self.pred_cls[i].bias.requires_grad:
                param_group_conv_bias = {'params': [self.pred_cls[i].bias]}
                param_group_conv_bias['lr'] = base_lr * 1.0
                param_group_conv_bias['base_lr'] = base_lr * 1.0
                param_group_conv_bias['weight_decay'] = base_wd
                param_group_conv_bias['need_clip'] = need_clip
                param_group_conv_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias)
            if self.pred_reg[i].weight.requires_grad:
                param_group_conv_weight2 = {'params': [self.pred_reg[i].weight]}
                param_group_conv_weight2['lr'] = base_lr * 1.0
                param_group_conv_weight2['base_lr'] = base_lr * 1.0
                param_group_conv_weight2['weight_decay'] = base_wd
                param_group_conv_weight2['need_clip'] = need_clip
                param_group_conv_weight2['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight2)
            if self.pred_reg[i].bias.requires_grad:
                param_group_conv_bias2 = {'params': [self.pred_reg[i].bias]}
                param_group_conv_bias2['lr'] = base_lr * 1.0
                param_group_conv_bias2['base_lr'] = base_lr * 1.0
                param_group_conv_bias2['weight_decay'] = base_wd
                param_group_conv_bias2['need_clip'] = need_clip
                param_group_conv_bias2['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        # bias_cls = bias_init_with_prob(0.01)
        # for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
        #     constant_(cls_.weight)
        #     constant_(cls_.bias, bias_cls)
        #     constant_(reg_.weight)
        #     constant_(reg_.bias, 1.0)

        self.proj = torch.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj.requires_grad = False
        self.proj_conv.weight.requires_grad_(False)
        self.proj_conv.weight.copy_(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_distri.flatten(2).permute((0, 2, 1)))
        cls_score_list = torch.cat(cls_score_list, 1)
        reg_distri_list = torch.cat(reg_distri_list, 1)

        # import numpy as np
        # dic = np.load('../aaa.npz')
        # cls_score_list = torch.Tensor(dic['cls_score_list'])
        # reg_distri_list = torch.Tensor(dic['reg_distri_list'])
        # anchors = torch.Tensor(dic['anchors'])
        # anchor_points = torch.Tensor(dic['anchor_points'])
        # stride_tensor = torch.Tensor(dic['stride_tensor'])
        # gt_class = torch.Tensor(dic['gt_class'])
        # gt_bbox = torch.Tensor(dic['gt_bbox'])
        # pad_gt_mask = torch.Tensor(dic['pad_gt_mask'])
        # targets['gt_class'] = gt_class
        # targets['gt_bbox'] = gt_bbox
        # targets['pad_gt_mask'] = pad_gt_mask
        #
        # loss = torch.Tensor(dic['loss'])
        # loss_cls = torch.Tensor(dic['loss_cls'])
        # loss_iou = torch.Tensor(dic['loss_iou'])
        # loss_dfl = torch.Tensor(dic['loss_dfl'])
        # loss_l1 = torch.Tensor(dic['loss_l1'])

        losses = self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)
        return losses

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            # shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], -1).to(torch.float32)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full(
                    [h * w, 1], stride, dtype=torch.float32))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval2(self, feats):
        # if self.eval_size:
        #     anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        # else:
        #     anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l])
            reg_dist = reg_dist.permute((0, 2, 1, 3))
            reg_dist = F.softmax(reg_dist, dim=1)
            reg_dist = self.proj_conv(reg_dist)
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = torch.cat(cls_score_list, -1)  # [N, 80, A]
        reg_dist_list = torch.cat(reg_dist_list, -1)    # [N,  4, A]

        # return cls_score_list, reg_dist_list, anchor_points, stride_tensor
        out = torch.cat([reg_dist_list, cls_score_list], 1)    # [N,  4+80, A]
        return [out,]

    def forward_eval(self, feats):
        # if self.eval_size:
        #     anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        # else:
        #     anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            # b, _, h, w = feat.shape
            # l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l])
            reg_dist = reg_dist.reshape([1, 4, self.reg_max + 1, -1])
            reg_dist = reg_dist.permute((0, 2, 1, 3))
            reg_dist = F.softmax(reg_dist, dim=1)
            reg_dist = self.proj_conv(reg_dist)
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([1, self.num_classes, -1]))
            reg_dist_list.append(reg_dist.reshape([1, 4, -1]))

        cls_score_list = torch.cat(cls_score_list, -1)  # [N, 80, A]
        reg_dist_list = torch.cat(reg_dist_list, -1)    # [N,  4, A]

        # return cls_score_list, reg_dist_list, anchor_points, stride_tensor
        out = torch.cat([reg_dist_list, cls_score_list], 1)    # [N,  4+80, A]
        return [out,]

    def export_ncnn(self, ncnn_data, bottom_names):
        feats = bottom_names
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = ncnn_utils.adaptive_avg_pool2d(ncnn_data, [feat, ], output_size='(1, 1)')

            x = self.stem_cls[i].export_ncnn(ncnn_data, [feat, avg_feat[0]])
            # 逐元素相加
            bottom_names = x + [feat, ]
            bottom_names = ncnn_utils.binaryOp(ncnn_data, bottom_names, op='Add')

            # 然后是卷积操作
            cls_logit = ncnn_utils.conv2d(ncnn_data, bottom_names, self.pred_cls[i])

            x = self.stem_reg[i].export_ncnn(ncnn_data, [feat, avg_feat[0]])
            # 然后是卷积操作
            reg_dist = ncnn_utils.conv2d(ncnn_data, x, self.pred_reg[i])
            reg_dist = ncnn_utils.reshape(ncnn_data, reg_dist, (1, 4, self.reg_max + 1, -1))
            reg_dist = ncnn_utils.permute(ncnn_data, reg_dist, '(0, 2, 1, 3)')
            reg_dist = ncnn_utils.softmax(ncnn_data, reg_dist, dim=1)
            reg_dist = ncnn_utils.conv2d(ncnn_data, reg_dist, self.proj_conv)

            cls_score = ncnn_utils.activation(ncnn_data, cls_logit, act_name='sigmoid')

            cls_score = ncnn_utils.reshape(ncnn_data, cls_score, (1, self.num_classes, -1))
            reg_dist = ncnn_utils.reshape(ncnn_data, reg_dist, (1, 4, -1))

            cls_score_list.append(cls_score[0])
            reg_dist_list.append(reg_dist[0])
        cls_score_list = ncnn_utils.concat(ncnn_data, cls_score_list, dim=2)  # [N, 80, A]
        reg_dist_list = ncnn_utils.concat(ncnn_data, reg_dist_list, dim=2)    # [N,  4, A]

        # 转置一下，让ncnn更好处理
        cls_score_list = ncnn_utils.permute(ncnn_data, cls_score_list, '(0, 2, 1)')  # [N, A, 80]
        reg_dist_list = ncnn_utils.permute(ncnn_data, reg_dist_list, '(0, 2, 1)')    # [N, A,  4]
        return cls_score_list + reg_dist_list

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t

        # loss = F.binary_cross_entropy(
        #     score, label, weight=weight, reduction='sum')

        score = score.to(torch.float32)
        eps = 1e-9
        loss = label * (0 - torch.log(score + eps)) + \
               (1.0 - label) * (0 - torch.log(1.0 - score + eps))
        loss *= weight
        loss = loss.sum()
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label

        # loss = F.binary_cross_entropy(
        #     pred_score, gt_score, weight=weight, reduction='sum')

        # pytorch的F.binary_cross_entropy()的weight不能向前传播梯度，但是
        # paddle的F.binary_cross_entropy()的weight可以向前传播梯度（给pred_score），
        # 所以这里手动实现F.binary_cross_entropy()
        # 使用混合精度训练时，pred_score类型是torch.float16，需要转成torch.float32避免log(0)=nan
        pred_score = pred_score.to(torch.float32)
        eps = 1e-9
        loss = gt_score * (0 - torch.log(pred_score + eps)) + \
               (1.0 - gt_score) * (0 - torch.log(1.0 - pred_score + eps))
        loss *= weight
        loss = loss.sum()
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        b, l, _ = get_static_shape(pred_dist)
        device = pred_dist.device
        pred_dist = pred_dist.reshape([b, l, 4, self.reg_max + 1])
        pred_dist = F.softmax(pred_dist, dim=-1)
        pred_dist = pred_dist.matmul(self.proj.to(device))
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clamp(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.int64)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float32) - target
        weight_right = 1 - weight_left

        eps = 1e-9
        # 使用混合精度训练时，pred_dist类型是torch.float16，pred_dist_act类型是torch.float32
        pred_dist_act = F.softmax(pred_dist, dim=-1)
        target_left_onehot = F.one_hot(target_left, pred_dist_act.shape[-1])
        target_right_onehot = F.one_hot(target_right, pred_dist_act.shape[-1])
        loss_left = target_left_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_right = target_right_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_left = loss_left.sum(-1) * weight_left
        loss_right = loss_right.sum(-1) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).repeat(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([]).to(pred_dist.device)
            loss_iou = torch.zeros([]).to(pred_dist.device)
            loss_dfl = pred_dist.sum() * 0.
            # loss_l1 = None
            # loss_iou = None
            # loss_dfl = None
        return loss_l1, loss_iou, loss_dfl




class PPYOLOE(torch.nn.Module):
    def __init__(self, backbone, neck, yolo_head):
        super(PPYOLOE, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head

    def forward(self, x, scale_factor=None, targets=None):
        '''
        获得损失（训练）、推理 都要放在forward()中进行，否则DDP会计算错误结果。
        '''
        body_feats = self.backbone(x)
        fpn_feats = self.neck(body_feats)
        out = self.yolo_head(fpn_feats, targets)
        # if self.training:
        #     return out
        # else:
        #     out = self.yolo_head.post_process(out, scale_factor)
        #     return out
        return out

    def export_ncnn(self, ncnn_data, bottom_names):
        body_feats_names = self.backbone.export_ncnn(ncnn_data, bottom_names)
        fpn_feats_names = self.neck.export_ncnn(ncnn_data, body_feats_names)
        outputs = self.yolo_head.export_ncnn(ncnn_data, fpn_feats_names)
        # print(ncnn_data['pp'])
        # print(body_feats_names)
        # print(fpn_feats_names)
        return outputs



def add_coord(x, data_format):
    b = x.shape[0]
    if data_format == 'NCHW':
        h, w = x.shape[2], x.shape[3]
    else:
        h, w = x.shape[1], x.shape[2]

    gx = torch.arange(0, w, dtype=x.dtype, device=x.device) / (w - 1.) * 2.0 - 1.
    gy = torch.arange(0, h, dtype=x.dtype, device=x.device) / (h - 1.) * 2.0 - 1.

    if data_format == 'NCHW':
        gx = gx.reshape([1, 1, 1, w]).expand([b, 1, h, w])
        gy = gy.reshape([1, 1, h, 1]).expand([b, 1, h, w])
    else:
        gx = gx.reshape([1, 1, w, 1]).expand([b, h, w, 1])
        gy = gy.reshape([1, h, 1, 1]).expand([b, h, w, 1])

    gx.requires_grad = False
    gy.requires_grad = False
    return gx, gy


class CoordConv(torch.nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size,
                 padding,
                 data_format='NCHW'):
        super(CoordConv, self).__init__()
        self.conv = ConvBNLayer(
            ch_in + 2, ch_out, filter_size, padding=padding, act=None)
        self.data_format = data_format

    def forward(self, x):
        gx, gy = add_coord(x, self.data_format)
        if self.data_format == 'NCHW':
            y = torch.cat([x, gx, gy], 1)
        else:
            y = torch.cat([x, gx, gy], -1)
        y = self.conv(y)
        return y

    def export_ncnn(self, ncnn_data, bottom_names):
        bottom_names = ncnn_utils.coordconcat(ncnn_data, bottom_names)
        bottom_names = self.conv.export_ncnn(ncnn_data, bottom_names)
        return bottom_names











