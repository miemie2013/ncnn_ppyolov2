// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Part of the following code in this file refs to
// https://github.com/msracver/Deformable-ConvNets/blob/master/faster_rcnn/operator_cxx/deformable_convolution.cu
//
// Copyright (c) 2017 Microsoft
// Licensed under The Apache-2.0 License [see LICENSE for details]
// \file deformable_psroi_pooling.cu
// \brief
// \author Yi Li, Guodong Zhang, Jifeng Dai

#pragma once
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/deformable_conv_func.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using CPUDeviceContext = platform::CPUDeviceContext;


template <typename T>
HOSTDEVICE T DmcnIm2colBilinear(const T* bottom_data, const int data_width,
                                const int height, const int width, T h, T w) {
/*
bottom_data: bottom_data是输入特征图[in_C, H, W]每个通道开始的位置。比如[0, 0, 0]、[1, 0, 0]、[2, 0, 0]、...
data_width: W   输入特征图的宽
height: H   输入特征图的高
width:  W   输入特征图的宽
h: 最终采样位置的y坐标
w: 最终采样位置的x坐标
*/
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh;
  T hw = 1 - lw;

  T v1 =
      (h_low >= 0 && w_low >= 0) ? bottom_data[h_low * data_width + w_low] : 0;
  T v2 = (h_low >= 0 && w_high <= width - 1)
             ? bottom_data[h_low * data_width + w_high]
             : 0;
  T v3 = (h_high <= height - 1 && w_low >= 0)
             ? bottom_data[h_high * data_width + w_low]
             : 0;
  T v4 = (h_high <= height - 1 && w_high <= width - 1)
             ? bottom_data[h_high * data_width + w_high]
             : 0;

  T w1 = hh * hw;
  T w2 = hh * lw;
  T w3 = lh * hw;
  T w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}


template <typename T>
void ModulatedDeformableIm2colCPUKernel(
    const int num_kernels, const T* data_im, const T* data_offset,
    const T* data_mask, const int height, const int width, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int num_channels, const int deformable_group, const int height_col,
    const int width_col, T* data_col) {
/*
num_kernels: num_kernels = in_C * im2col_step * out_H * out_W
data_im: 输入特征图(形状[N, in_C, H, W])的指针，外部循环每次指针的地址依次是x[0, 0, 0, 0]、x[1, 0, 0, 0]、x[2, 0, 0, 0]、...即当前图片开始的地址。
data_offset: offset(形状[N, kH*kW*2, out_H, out_W])的指针，外部循环每次指针的地址依次是offset[0, 0, 0, 0]、offset[1, 0, 0, 0]、offset[2, 0, 0, 0]、...即当前图片开始的地址。
data_mask: mask(形状[N, kH*kW, out_H, out_W])的指针，外部循环每次指针的地址依次是mask[0, 0, 0, 0]、mask[1, 0, 0, 0]、mask[2, 0, 0, 0]、...即当前图片开始的地址。
height: H
width: W
kernel_h: kH
kernel_w: kW
pad_h: 1
pad_w: 1
stride_h: 1
stride_w: 1
dilation_h: 1
dilation_w: 1
channel_per_deformable_group: channel_per_deformable_group = in_C / deformable_groups
batch_size: 注意，batch_size是im2col_step==1
num_channels: in_C
deformable_group: 1
height_col: out_H
width_col: out_W
data_col: 输入图片im2col的结果 shape=[groups, K, N]=[groups, in_C * kH * kW / groups, out_H * out_W]
*/

  // 这个for循环表示了卷积窗滑动，并且多遍历了in_C这一维，即输入特征图的通道数。（im2col_step==1暂时不管）
  for (int i = 0; i < num_kernels; i++) {  // num_kernels = in_C * im2col_step * out_H * out_W
    // i是1个1D的坐标，把i变成4D的坐标，即某个形为[in_C * im2col_step * out_H * out_W, ]的张量 reshape 成[in_C, im2col_step, out_H, out_W]张量时坐标的相应变换。
    // i==0对应坐标[0, 0, 0, 0]
    // 推广到i对应坐标[c_im, b_col, h_col, w_col]，现在开始求[c_im, b_col, h_col, w_col]
    const int w_col = i % width_col;  // “个位”，输出特征图所有像素x坐标
    const int h_col = (i / width_col) % height_col;  // “十位”，输出特征图所有像素y坐标
    const int b_col = (i / width_col) / height_col % batch_size;  // “百位”，？
    const int c_im = (i / width_col / height_col) / batch_size;  // “千位”，因为i不可能大于num_kernels，所以不需要%in_C
    const int c_col = c_im * kernel_h * kernel_w;  // 某个形为[in_C*kH*kW, ...]的张量 对应c_im==某个值时 第0维开始的坐标。miemie2013: 这里是全场最佳解读！！！

    const int deformable_group_index = c_im / channel_per_deformable_group;  // “千位”c_im是in_C维的坐标，c_im是第几个组的。

    const int h_in = h_col * stride_h - pad_h;  // 卷积窗(左上角)现在的位置在原图中的y坐标
    const int w_in = w_col * stride_w - pad_w;  // 卷积窗(左上角)现在的位置在原图中的x坐标

    // 准备1个指针data_col_ptr，方便写入im2col的结果。+后面的表达式是把4D的坐标变成1个1D的坐标，
    // 即某个形为[in_C*kH*kW, im2col_step, out_H, out_W]的张量 reshape 成[in_C*kH*kW * im2col_step * out_H * out_W, ]张量时坐标的相应变换。
    // miemie2013: 这里是全场最佳解读！！！因为0维坐标是c_col，推导出0维长度是in_C*kH*kW
    // 坐标[c_col, b_col, h_col, w_col] 变成 xxx
    T* data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;

    // 输入特征图。data_im外部循环传入的是每张图片，形状是[in_C, H, W]
    // 因为im2col_step==1，所以b_col恒==0，所以data_im_ptr是每个通道开始的位置。比如[0, 0, 0]、[1, 0, 0]、[2, 0, 0]、...
    const T* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;

    // offset。data_im外部循环传入的是每张图片，形状是[kH*kW*2, out_H, out_W]
    // 因为im2col_step==1，所以b_col恒==0，当deformable_group==1时，deformable_group_index恒==0，data_offset_ptr恒指向[0, 0, 0]
    const T* data_offset_ptr =
        data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;

    // mask。data_im外部循环传入的是每张图片，形状是[kH*kW, out_H, out_W]
    // 因为im2col_step==1，所以b_col恒==0，当deformable_group==1时，deformable_group_index恒==0，data_offset_ptr恒指向[0, 0, 0]
    const T* data_mask_ptr =
        data_mask +
        (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;

    // 遍历卷积核每一行每一列
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        // data_offset_ptr形状是[kH*kW*2, out_H, out_W], 第0维是yxyxyx...这样的顺序
        // reshape data_offset_ptr形状为[kH, kW, 2, out_H, out_W]
        // 这里是把 5D坐标[i, j, 0, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的y偏移
        const int data_offset_h_ptr =
            (((i * kernel_w + j) * 2) * height_col + h_col) * width_col + w_col;

        // 这里是把 5D坐标[i, j, 1, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的x偏移
        const int data_offset_w_ptr =
            (((i * kernel_w + j) * 2 + 1) * height_col + h_col) * width_col + w_col;

        // data_mask_ptr形状是[kH*kW, out_H, out_W],
        // reshape data_mask_ptr形状为[kH, kW, out_H, out_W]
        // 这里是把 4D坐标[i, j, h_col, w_col] 变成 1D坐标data_mask_hw_ptr ，取出预测的mask重要程度
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;

        const T offset_h = data_offset_ptr[data_offset_h_ptr];  // 预测的y偏移
        const T offset_w = data_offset_ptr[data_offset_w_ptr];  // 预测的x偏移
        const T mask = data_mask_ptr[data_mask_hw_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;  // 最终采样位置的y坐标
        const T w_im = w_in + j * dilation_w + offset_w;  // 最终采样位置的x坐标
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val =
              DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);  // 双线性插值采样。
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T>
static inline void ModulatedDeformableIm2colCPU(
    const platform::CPUDeviceContext& ctx, const T* data_im,
    const T* data_offset, const T* data_mask,
    const std::vector<int64_t> im_shape, const std::vector<int64_t> col_shape,
    const std::vector<int64_t> filter_shape, const std::vector<int> paddings,
    const std::vector<int> strides, const std::vector<int> dilations,
    const int deformable_groups, T* data_col) {
/*
data_im: 输入特征图(形状[N, in_C, H, W])的指针，外部循环每次指针的地址依次是x[0, 0, 0, 0]、x[1, 0, 0, 0]、x[2, 0, 0, 0]、...即当前图片开始的地址。
data_offset: offset(形状[N, kH*kW*2, out_H, out_W])的指针，外部循环每次指针的地址依次是offset[0, 0, 0, 0]、offset[1, 0, 0, 0]、offset[2, 0, 0, 0]、...即当前图片开始的地址。
data_mask: mask(形状[N, kH*kW, out_H, out_W])的指针，外部循环每次指针的地址依次是mask[0, 0, 0, 0]、mask[1, 0, 0, 0]、mask[2, 0, 0, 0]、...即当前图片开始的地址。
im_shape: 值是[in_C, H, W]
col_shape: 值是[in_C * kH * kW, im2col_step, out_H, out_W]
filter_shape: 值是[out_C, in_C, kH, kW]
paddings: 1
strides: 1
dilations: 1
deformable_groups: 1
data_col: 输入图片im2col的结果 shape=[groups, K, N]=[groups, in_C * kH * kW / groups, out_H * out_W]
*/
  int channel_per_deformable_group = im_shape[0] / deformable_groups;  // channel_per_deformable_group = in_C / deformable_groups
  int num_kernels = im_shape[0] * col_shape[1] * col_shape[2] * col_shape[3];  // num_kernels = in_C * im2col_step * out_H * out_W

  // get outputs of im2col with offset by bilinear interpolation
  ModulatedDeformableIm2colCPUKernel(
      num_kernels, data_im, data_offset, data_mask, im_shape[1], im_shape[2],
      filter_shape[2], filter_shape[3], paddings[0], paddings[1], strides[0],
      strides[1], dilations[0], dilations[1], channel_per_deformable_group,
      col_shape[1], im_shape[0], deformable_groups, col_shape[2], col_shape[3],
      data_col);
/*
num_kernels: num_kernels = in_C * im2col_step * out_H * out_W
data_im: 输入特征图(形状[N, in_C, H, W])的指针，外部循环每次指针的地址依次是x[0, 0, 0, 0]、x[1, 0, 0, 0]、x[2, 0, 0, 0]、...即当前图片开始的地址。
data_offset: offset(形状[N, kH*kW*2, out_H, out_W])的指针，外部循环每次指针的地址依次是offset[0, 0, 0, 0]、offset[1, 0, 0, 0]、offset[2, 0, 0, 0]、...即当前图片开始的地址。
data_mask: mask(形状[N, kH*kW, out_H, out_W])的指针，外部循环每次指针的地址依次是mask[0, 0, 0, 0]、mask[1, 0, 0, 0]、mask[2, 0, 0, 0]、...即当前图片开始的地址。
height: H
width: W
kernel_h: kH
kernel_w: kW
pad_h: 1
pad_w: 1
stride_h: 1
stride_w: 1
dilation_h: 1
dilation_w: 1
channel_per_deformable_group: channel_per_deformable_group = in_C / deformable_groups
batch_size: 注意，batch_size是im2col_step==1
num_channels: in_C
deformable_group: 1
height_col: out_H
width_col: out_W
data_col: 输入图片im2col的结果 shape=[groups, K, N]=[groups, in_C * kH * kW / groups, out_H * out_W]
*/
}

template <typename T>
class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* offset = ctx.Input<Tensor>("Offset");
    auto* mask = ctx.Input<Tensor>("Mask");
    Tensor filter = *ctx.Input<Tensor>("Filter");
    Tensor* output = ctx.Output<Tensor>("Output");
    output->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<CPUDeviceContext>();

    const int groups = ctx.Attr<int>("groups");
    const int deformable_groups = ctx.Attr<int>("deformable_groups");
    const int im2col_step = ctx.Attr<int>("im2col_step");
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    const std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    const std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");

    const int batch_size = static_cast<int>(input->dims()[0]);

    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));

    // col_shape_vec: {in_C * kH * kW, im2col_step, out_H, out_W}
    std::vector<int64_t> col_buffer_shape_vec(filter_shape_vec.size());
    col_buffer_shape_vec[0] =
        input->dims()[1] * filter.dims()[2] * filter.dims()[3];
    col_buffer_shape_vec[1] = im2col_step;
    for (size_t j = 0; j < filter_shape_vec.size() - 2; ++j) {
      col_buffer_shape_vec[j + 2] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_buffer_shape_vec));  // 值是[in_C * kH * kW, im2col_step, out_H, out_W]  std::vector<int64_t>转framework::DDim
    std::vector<int64_t> output_buffer_shape_vec(1);
    output_buffer_shape_vec[0] = batch_size * output_shape_vec[1] *
                                 output_shape_vec[2] * output_shape_vec[3];
    framework::DDim output_shape(framework::make_ddim(output_buffer_shape_vec));  // 值是[N * out_C * out_H * out_W, ]  std::vector<int64_t>转framework::DDim
    Tensor col_buffer;
    Tensor output_buffer;
    col_buffer = ctx.AllocateTmpTensor<T, CPUDeviceContext>(col_shape, dev_ctx);
    output_buffer =
        ctx.AllocateTmpTensor<T, CPUDeviceContext>(output_shape, dev_ctx);
    int64_t M = output_shape_vec[1] / groups;   // M = out_C / groups  是输出特征图每组的通道数
    int64_t N = im2col_step * output_shape_vec[2] * output_shape_vec[3];   // N = 1 * out_H * out_W  是输出特征图 高*宽
    int64_t K =
        input->dims()[1] * filter_shape_vec[2] * filter_shape_vec[3] / groups;   // K = in_C * kH * kW / groups  是输入特征图每组的通道数 * kH * kW，

    Tensor weight_3d;
    weight_3d.ShareDataWith(filter).Resize(
        framework::make_ddim({groups, M, K}));   // 卷积层的权重 shape=[groups, M, K]=[groups, out_C / groups, in_C * kH * kW / groups]

    Tensor col_buffer_3d;
    col_buffer_3d.ShareDataWith(col_buffer)
        .Resize(framework::make_ddim({groups, K, N}));   // 输入图片im2col的结果 shape=[groups, K, N]=[groups, in_C * kH * kW / groups, out_H * out_W]

    // 可变形卷积最后的结果 shape=[batch_size, groups, M, N]=[batch_size, groups, out_C / groups, out_H * out_W]
    // 即卷积层权重 矩阵乘 im2col的结果 = [groups, out_C / groups, in_C * kH * kW / groups] * [groups, in_C * kH * kW / groups, out_H * out_W]
    //                              = [groups, out_C / groups, out_H * out_W]
    Tensor output_4d;
    output_4d.ShareDataWith(output_buffer)
        .Resize(framework::make_ddim({batch_size / im2col_step, groups, M, N}));
    output_4d.mutable_data<T>(ctx.GetPlace());

    framework::DDim input_shape =
        framework::slice_ddim(input->dims(), 1, input->dims().size());  // 值是[in_C, H, W]  slice_ddim表示截取shape从1到最后，即input.shape[1:]
    std::vector<int64_t> input_shape_vec = framework::vectorize(input_shape);  // 值是[in_C, H, W]  framework::DDim转std::vector<int64_t>
    int input_dim = input->numel() / input->dims()[0];           // input_dim = in_C*H*W
    int input_offset_dim = offset->numel() / offset->dims()[0];  // input_offset_dim = kH * kW * 2 * out_H * out_W
    int input_mask_dim = mask->numel() / mask->dims()[0];        // input_mask_dim = kH * kW * out_H * out_W
    auto blas = math::GetBlas<CPUDeviceContext, T>(dev_ctx);     // 调用MatMul的工具人
    const T* input_ptr = input->data<T>();
    const T* offset_ptr = offset->data<T>();
    const T* mask_ptr = mask->data<T>();
    col_buffer.mutable_data<T>(ctx.GetPlace());
    T* col_buffer_ptr = col_buffer.data<T>();   // 输入图片im2col的结果 shape=[groups, K, N]=[groups, in_C * kH * kW / groups, out_H * out_W]
    for (int i = 0; i < batch_size / im2col_step; ++i) {  // 遍历每张图片
      ModulatedDeformableIm2colCPU(
          dev_ctx, input_ptr + i * im2col_step * input_dim,
          offset_ptr + i * im2col_step * input_offset_dim,
          mask_ptr + i * im2col_step * input_mask_dim, input_shape_vec,
          col_buffer_shape_vec, filter_shape_vec, paddings, strides, dilations,
          deformable_groups, col_buffer_ptr);
/*
data_im: 输入特征图(形状[N, in_C, H, W])的指针，外部循环每次指针的地址依次是x[0, 0, 0, 0]、x[1, 0, 0, 0]、x[2, 0, 0, 0]、...即当前图片开始的地址。
data_offset: offset(形状[N, kH*kW*2, out_H, out_W])的指针，外部循环每次指针的地址依次是offset[0, 0, 0, 0]、offset[1, 0, 0, 0]、offset[2, 0, 0, 0]、...即当前图片开始的地址。
data_mask: mask(形状[N, kH*kW, out_H, out_W])的指针，外部循环每次指针的地址依次是mask[0, 0, 0, 0]、mask[1, 0, 0, 0]、mask[2, 0, 0, 0]、...即当前图片开始的地址。
im_shape: 值是[in_C, H, W]
col_shape: 值是[in_C * kH * kW, im2col_step, out_H, out_W]
filter_shape: 值是[out_C, in_C, kH, kW]
paddings: 1
strides: 1
dilations: 1
deformable_groups: 1
data_col: 输入图片im2col的结果 shape=[groups, K, N]=[groups, in_C * kH * kW / groups, out_H * out_W]
*/
      Tensor output_3d = output_4d.Slice(i, i + 1).Resize(
          framework::slice_ddim(output_4d.dims(), 1, output_4d.dims().size()));
      // get the product of pixel and weight
      for (int g = 0; g < groups; ++g) {
        Tensor weight_3d_slice =
            weight_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                weight_3d.dims(), 1, weight_3d.dims().size()));
        Tensor col_buffer_3d_slice =
            col_buffer_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                col_buffer_3d.dims(), 1, col_buffer_3d.dims().size()));
        Tensor output_3d_slice =
            output_3d.Slice(g, g + 1).Resize(framework::slice_ddim(
                output_3d.dims(), 1, output_3d.dims().size()));
        blas.MatMul(weight_3d_slice, false, col_buffer_3d_slice, false, T(1.0),
                    &output_3d_slice, T(0.0));
      }
    }
    output->ShareDataWith(output_buffer)
        .Resize(framework::make_ddim(output_shape_vec));
  }
};


}  // namespace operators
}  // namespace paddle
Footer
