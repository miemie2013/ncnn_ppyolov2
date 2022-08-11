
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np


def DmcnIm2colBilinear(bottom_data, data_im_ptr, data_width, height, width, h, w):
    h_low = np.floor(h)
    w_low = np.floor(w)
    h_high = h_low + 1
    w_high = w_low + 1

    lh = h - h_low
    lw = w - w_low
    hh = 1 - lh
    hw = 1 - lw

    v1 = 0.
    if h_low >= 0 and w_low >= 0:
        v1 = bottom_data[int(data_im_ptr + h_low * data_width + w_low)]
    v2 = 0.
    if h_low >= 0 and w_high <= width - 1:
        v2 = bottom_data[int(data_im_ptr + h_low * data_width + w_high)]
    v3 = 0.
    if h_high <= height - 1 and w_low >= 0:
        v3 = bottom_data[int(data_im_ptr + h_high * data_width + w_low)]
    v4 = 0.
    if h_high <= height - 1 and w_high <= width - 1:
        v4 = bottom_data[int(data_im_ptr + h_high * data_width + w_high)]

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4



def DmcnIm2colBilinear2(input, data_im_ptr, W, H, W2, h_im, w_im):
    h_low = np.floor(h_im)
    w_low = np.floor(w_im)
    h_high = h_low + 1
    w_high = w_low + 1

    lh = h_im - h_low
    lw = w_im - w_low
    hh = 1 - lh
    hw = 1 - lw

    v1 = 0.
    if h_low >= 0 and w_low >= 0:
        v1 = input[int(data_im_ptr + h_low * W + w_low)]
    v2 = 0.
    if h_low >= 0 and w_high <= W - 1:
        v2 = input[int(data_im_ptr + h_low * W + w_high)]
    v3 = 0.
    if h_high <= H - 1 and w_low >= 0:
        v3 = input[int(data_im_ptr + h_high * W + w_low)]
    v4 = 0.
    if h_high <= H - 1 and w_high <= W - 1:
        v4 = input[int(data_im_ptr + h_high * W + w_high)]

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


class DeformableConvCPUKernel(object):
    def __init__(self):
        super(DeformableConvCPUKernel, self).__init__()

    def __call__(self, input, offset, mask, weight, bias, stride):
        '''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
即为DCN前向代码。
        '''
        num_output = -99
        kernel_w = -99
        kernel_h = -99
        dilation_w = 1
        dilation_h = 1
        stride_w = stride
        stride_h = stride
        bias_term = True
        weight_data_size = -99

        offset = np.reshape(offset, (-1,))
        mask = np.reshape(mask, (-1,))

        in_C, H, W = input.shape
        input = np.reshape(input, (-1,))
        out_C, in_C, kH, kW = weight.shape
        num_output = out_C
        kernel_h = kH
        kernel_w = kW
        weight_data_size = out_C * in_C * kH * kW
        filter_size = kH
        paddings = (filter_size - 1) // 2
        pad_left = paddings
        pad_right = paddings
        pad_top = paddings
        pad_bottom = paddings

        kernel_extent_h = dilation_h * (kernel_h - 1) + 1
        kernel_extent_w = dilation_w * (kernel_w - 1) + 1
        out_H = (H + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1
        out_W = (W + pad_left + pad_right - kernel_extent_w) / stride_w + 1
        out_H = int(out_H)
        out_W = int(out_W)

        # im2col算法：output = weight * im2col
        # weight形状reshape成[out_C, in_C * kH * kW]，im2col形状是[in_C * kH * kW, out_H * out_W]
        # 矩阵相乘，结果的形状是[out_C, out_H * out_W]，就是卷积运算的结果了！

        # im2col，最后会变成[in_C * kH * kW, out_H * out_W]的形状, 这里模拟C++里的指针。
        im2col = np.zeros((in_C * kH * kW * out_H * out_W, ), dtype=np.float32)
        im2col_ptr = 0
        data_im = 0
        data_offset = 0
        data_mask = 0

        # im2col_ncnn，最后会变成[out_H * out_W, in_C * kH * kW]的形状。
        # ncnn的矩阵相乘A*B=C中，A形状是[m, n]，B形状是[k, n]，C形状是[m, k]，即B矩阵是以转置的方式传入的。
        im2col_ncnn = np.zeros((out_H * out_W * in_C * kH * kW, ), dtype=np.float32)
        im2col_ncnn_ptr = 0

        # im2col
        # 卷积窗滑动，并且多遍历了in_C这一维。
        for c_im in range(in_C):
            for h_col in range(out_H):
                for w_col in range(out_W):
                    c_col = c_im * kH * kW   # 某个形为[in_C*kH*kW, ...]的张量 对应c_im==某个值时 第0维开始的坐标。miemie2013: 这里是全场最佳解读！！！

                    h_in = h_col * stride_h - pad_top    # 卷积窗(左上角)现在的位置在原图中的y坐标
                    w_in = w_col * stride_w - pad_left   # 卷积窗(左上角)现在的位置在原图中的x坐标

                    '''
            // 准备1个指针data_col_ptr，方便写入im2col的结果。+后面的表达式是把3D的坐标变成1个1D的坐标，
            // 即某个形为[in_C*kH*kW, out_H, out_W]的张量 reshape 成[in_C*kH*kW * out_H * out_W, ]张量时坐标的相应变换。
            // miemie2013: 这里是全场最佳解读！！！因为0维坐标是c_col，推导出0维长度是in_C*kH*kW
            // 坐标[c_col, h_col, w_col] 变成 xxx
                    '''
                    data_col_ptr = im2col_ptr + (c_col * out_H + h_col) * out_W + w_col
                    data_col_ptr2 = im2col_ncnn_ptr + (h_col * out_W + w_col) * in_C * kH * kW + c_col

                    data_im_ptr = data_im + c_im * H * W
                    data_offset_ptr = data_offset
                    data_mask_ptr = data_mask

                    # 遍历卷积核每一行每一列
                    for i in range(kH):
                        for j in range(kW):
                            '''
                // data_offset_ptr形状是[kH*kW*2, out_H, out_W], 第0维是yxyxyx...这样的顺序
                // reshape data_offset_ptr形状为[kH, kW, 2, out_H, out_W]
                // 这里是把 5D坐标[i, j, 0, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的y偏移
                            '''
                            data_offset_h_ptr = (((i * kW + j) * 2) * out_H + h_col) * out_W + w_col
                            '''
                // 这里是把 5D坐标[i, j, 1, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的x偏移
                            '''
                            data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_H + h_col) * out_W + w_col
                            '''
                // data_mask_ptr形状是[kH*kW, out_H, out_W],
                // reshape data_mask_ptr形状为[kH, kW, out_H, out_W]
                // 这里是把 4D坐标[i, j, h_col, w_col] 变成 1D坐标data_mask_hw_ptr ，取出预测的mask重要程度
                            '''
                            data_mask_hw_ptr = ((i * kernel_w + j) * out_H + h_col) * out_W + w_col

                            offset_h = offset[data_offset_ptr + data_offset_h_ptr]
                            offset_w = offset[data_offset_ptr + data_offset_w_ptr]
                            mask_ = mask[data_mask_ptr + data_mask_hw_ptr]
                            val = 0.
                            h_im = h_in + i * dilation_h + offset_h
                            w_im = w_in + j * dilation_w + offset_w
                            if h_im > -1 and w_im > -1 and h_im < H and w_im < W:
                                val = DmcnIm2colBilinear(input, data_im_ptr, W, H, W, h_im, w_im)
                            im2col[data_col_ptr] = val * mask_
                            im2col_ncnn[data_col_ptr2] = val * mask_
                            data_col_ptr += out_H * out_W
                            data_col_ptr2 += 1
        # 把im2col填写完毕。
        im2col = np.reshape(im2col, (in_C * kH * kW, out_H * out_W))   # [in_C * kH * kW, out_H * out_W]
        im2col_ncnn = np.reshape(im2col_ncnn, (out_H * out_W, in_C * kH * kW))
        aaaaaaaaa1 = im2col.T
        aaaaaaaaa2 = im2col_ncnn
        ddd1 = np.sum((aaaaaaaaa1 - aaaaaaaaa2) ** 2)
        print('ddd=%.9f' % ddd1)



        # ncnn_output = '../build/examples/output.txt'
        # with open(ncnn_output, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip()
        # line = line[:-1]
        # ss = line.split(',')
        # y = []
        # for s in ss:
        #     y.append(float(s))
        # y = np.array(y).astype(np.float32)
        # y = np.reshape(y, aaaaaaaaa2.shape)
        #
        # yy1 = y
        # yy2 = aaaaaaaaa2
        # ddd = np.sum((yy1 - yy2) ** 2)
        # print('ddd=%.9f' % ddd)


        weight = np.reshape(weight, (out_C, in_C * kH * kW))   # [out_C, in_C * kH * kW]
        output = np.matmul(weight, im2col)   # [out_C, out_H * out_W]
        if bias_term:
            bias = np.reshape(bias, (out_C, 1))   # [out_C, 1]
            output += bias
        output = np.reshape(output, (1, out_C, out_H, out_W))   # [1, out_C, out_H, out_W]
        return output

class DeformableConvCPUKernelv2(object):
    def __init__(self):
        super(DeformableConvCPUKernelv2, self).__init__()

    def __call__(self, input, offset, mask, weight, bias, stride):
        '''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
即为DCN前向代码。
        '''
        num_output = -99
        kernel_w = -99
        kernel_h = -99
        dilation_w = 1
        dilation_h = 1
        stride_w = stride
        stride_h = stride
        bias_term = True
        weight_data_size = -99

        offset = np.reshape(offset, (-1,))
        mask = np.reshape(mask, (-1,))

        in_C, H, W = input.shape
        input = np.reshape(input, (-1,))
        out_C, in_C, kH, kW = weight.shape
        num_output = out_C
        kernel_h = kH
        kernel_w = kW
        weight_data_size = out_C * in_C * kH * kW
        filter_size = kH
        paddings = (filter_size - 1) // 2
        pad_left = paddings
        pad_right = paddings
        pad_top = paddings
        pad_bottom = paddings

        kernel_extent_h = dilation_h * (kernel_h - 1) + 1
        kernel_extent_w = dilation_w * (kernel_w - 1) + 1
        out_H = (H + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1
        out_W = (W + pad_left + pad_right - kernel_extent_w) / stride_w + 1
        out_H = int(out_H)
        out_W = int(out_W)

        # im2col算法：output = im2col * weight_t
        # im2col形状是[out_H * out_W, in_C * kH * kW]
        # weight形状reshape成[out_C, in_C * kH * kW]，再转置成weight_t.shape=[in_C * kH * kW, out_C]
        # 矩阵相乘，结果的形状是[out_H * out_W, out_C]，就是卷积运算的结果了！

        # im2col，最后会变成[out_H * out_W, in_C * kH * kW]的形状, 这里模拟C++里的指针。
        im2col = np.zeros((out_H * out_W * in_C * kH * kW, ), dtype=np.float32)
        im2col_ptr = 0
        data_im = 0
        data_offset = 0
        data_mask = 0

        # im2col
        # 卷积窗滑动，并且多遍历了in_C这一维。
        for c_im in range(in_C):
            for h_col in range(out_H):
                for w_col in range(out_W):
                    c_col = c_im * kH * kW   # 某个形为[in_C*kH*kW, ...]的张量 对应c_im==某个值时 第0维开始的坐标。miemie2013: 这里是全场最佳解读！！！

                    h_in = h_col * stride_h - pad_top    # 卷积窗(左上角)现在的位置在原图中的y坐标
                    w_in = w_col * stride_w - pad_left   # 卷积窗(左上角)现在的位置在原图中的x坐标

                    '''
            // 准备1个指针data_col_ptr，方便写入im2col的结果。+后面的表达式是把3D的坐标变成1个1D的坐标，
            // 即某个形为[in_C*kH*kW, out_H, out_W]的张量 reshape 成[in_C*kH*kW * out_H * out_W, ]张量时坐标的相应变换。
            // miemie2013: 这里是全场最佳解读！！！因为0维坐标是c_col，推导出0维长度是in_C*kH*kW
            // 坐标[c_col, h_col, w_col] 变成 xxx
                    '''
                    data_col_ptr = im2col_ptr + (h_col * out_W + w_col) * in_C * kH * kW + c_col

                    data_im_ptr = data_im + c_im * H * W
                    data_offset_ptr = data_offset
                    data_mask_ptr = data_mask

                    # 遍历卷积核每一行每一列
                    for i in range(kH):
                        for j in range(kW):
                            '''
                // data_offset_ptr形状是[kH*kW*2, out_H, out_W], 第0维是yxyxyx...这样的顺序
                // reshape data_offset_ptr形状为[kH, kW, 2, out_H, out_W]
                // 这里是把 5D坐标[i, j, 0, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的y偏移
                            '''
                            data_offset_h_ptr = (((i * kW + j) * 2) * out_H + h_col) * out_W + w_col
                            '''
                // 这里是把 5D坐标[i, j, 1, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的x偏移
                            '''
                            data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_H + h_col) * out_W + w_col
                            '''
                // data_mask_ptr形状是[kH*kW, out_H, out_W],
                // reshape data_mask_ptr形状为[kH, kW, out_H, out_W]
                // 这里是把 4D坐标[i, j, h_col, w_col] 变成 1D坐标data_mask_hw_ptr ，取出预测的mask重要程度
                            '''
                            data_mask_hw_ptr = ((i * kernel_w + j) * out_H + h_col) * out_W + w_col

                            offset_h = offset[data_offset_ptr + data_offset_h_ptr]
                            offset_w = offset[data_offset_ptr + data_offset_w_ptr]
                            mask_ = mask[data_mask_ptr + data_mask_hw_ptr]
                            val = 0.
                            h_im = h_in + i * dilation_h + offset_h
                            w_im = w_in + j * dilation_w + offset_w
                            if h_im > -1 and w_im > -1 and h_im < H and w_im < W:
                                val = DmcnIm2colBilinear(input, data_im_ptr, W, H, W, h_im, w_im)
                            im2col[data_col_ptr] = val * mask_
                            data_col_ptr += 1
        # 把im2col填写完毕。
        im2col = np.reshape(im2col, (out_H * out_W, in_C * kH * kW))   # [out_H * out_W, in_C * kH * kW]



        # ncnn_output = '../build/examples/output.txt'
        # with open(ncnn_output, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip()
        # line = line[:-1]
        # ss = line.split(',')
        # y = []
        # for s in ss:
        #     y.append(float(s))
        # y = np.array(y).astype(np.float32)
        # y = np.reshape(y, aaaaaaaaa2.shape)
        #
        # yy1 = y
        # yy2 = aaaaaaaaa2
        # ddd = np.sum((yy1 - yy2) ** 2)
        # print('ddd=%.9f' % ddd)


        weight = np.reshape(weight, (out_C, in_C * kH * kW))   # [out_C, in_C * kH * kW]
        weight_t = weight.T     # [in_C * kH * kW, out_C]
        output = np.matmul(im2col, weight_t)   # [out_H * out_W, out_C]
        if bias_term:
            bias = np.reshape(bias, (1, out_C))   # [1, out_C]
            output += bias
        output = output.T   # [out_C, out_H * out_W]
        output = np.reshape(output, (1, out_C, out_H, out_W))   # [1, out_C, out_H, out_W]
        return output

class DeformableConvCPUKernelv3(object):
    def __init__(self):
        super(DeformableConvCPUKernelv3, self).__init__()

    def __call__(self, input, offset, mask, weight, bias, stride):
        '''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
即为DCN前向代码。
        '''
        num_output = -99
        kernel_w = -99
        kernel_h = -99
        dilation_w = 1
        dilation_h = 1
        stride_w = stride
        stride_h = stride
        bias_term = True
        weight_data_size = -99

        offset = np.reshape(offset, (-1,))
        mask = np.reshape(mask, (-1,))

        in_C, H, W = input.shape
        input = np.reshape(input, (-1,))
        out_C, in_C, kH, kW = weight.shape
        num_output = out_C
        kernel_h = kH
        kernel_w = kW
        weight_data_size = out_C * in_C * kH * kW
        filter_size = kH
        paddings = (filter_size - 1) // 2
        pad_left = paddings
        pad_right = paddings
        pad_top = paddings
        pad_bottom = paddings

        kernel_extent_h = dilation_h * (kernel_h - 1) + 1
        kernel_extent_w = dilation_w * (kernel_w - 1) + 1
        out_H = (H + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1
        out_W = (W + pad_left + pad_right - kernel_extent_w) / stride_w + 1
        out_H = int(out_H)
        out_W = int(out_W)

        # im2col算法：output = im2col * weight2
        # im2col形状是[out_H * out_W, kH * kW * in_C]
        # weight形状reshape成[out_C, in_C, kH * kW]，再转置成[out_C, kH * kW, in_C]，再reshape成[out_C, kH * kW * in_C]，再转置成[kH * kW * in_C, out_C]
        # 矩阵相乘，结果的形状是[out_H * out_W, out_C]，就是卷积运算的结果了！

        # im2col，最后会变成[out_H * out_W, in_C * kH * kW]的形状, 这里模拟C++里的指针。
        im2col = np.zeros((out_H * out_W * kH * kW * in_C, ), dtype=np.float32)
        im2col_ptr = 0
        data_im = 0
        data_offset = 0
        data_mask = 0

        # im2col
        # 卷积窗滑动，并且多遍历了in_C这一维。
        for h_col in range(out_H):
            for w_col in range(out_W):
                h_in = h_col * stride_h - pad_top    # 卷积窗(左上角)现在的位置在原图中的y坐标
                w_in = w_col * stride_w - pad_left   # 卷积窗(左上角)现在的位置在原图中的x坐标

                # im2col的形状是[out_H, out_W, kH, kW, in_C]，求内部坐标[h_col, w_col, 0, 0, 0]对应的1D坐标
                data_col_ptr = im2col_ptr + (h_col * out_W + w_col) * kH * kW * in_C

                data_offset_ptr = data_offset
                data_mask_ptr = data_mask

                # 遍历卷积核每一行每一列
                for i in range(kH):
                    for j in range(kW):
                        '''
            // data_offset_ptr形状是[kH*kW*2, out_H, out_W], 第0维是yxyxyx...这样的顺序
            // reshape data_offset_ptr形状为[kH, kW, 2, out_H, out_W]
            // 这里是把 5D坐标[i, j, 0, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的y偏移
                        '''
                        data_offset_h_ptr = (((i * kW + j) * 2) * out_H + h_col) * out_W + w_col
                        '''
            // 这里是把 5D坐标[i, j, 1, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的x偏移
                        '''
                        data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_H + h_col) * out_W + w_col
                        '''
            // data_mask_ptr形状是[kH*kW, out_H, out_W],
            // reshape data_mask_ptr形状为[kH, kW, out_H, out_W]
            // 这里是把 4D坐标[i, j, h_col, w_col] 变成 1D坐标data_mask_hw_ptr ，取出预测的mask重要程度
                        '''
                        data_mask_hw_ptr = ((i * kernel_w + j) * out_H + h_col) * out_W + w_col

                        offset_h = offset[data_offset_ptr + data_offset_h_ptr]
                        offset_w = offset[data_offset_ptr + data_offset_w_ptr]
                        mask_ = mask[data_mask_ptr + data_mask_hw_ptr]
                        h_im = h_in + i * dilation_h + offset_h
                        w_im = w_in + j * dilation_w + offset_w
                        cond = h_im > -1 and w_im > -1 and h_im < H and w_im < W
                        if cond:
                            # Bilinear
                            h_low = np.floor(h_im)
                            w_low = np.floor(w_im)
                            h_high = h_low + 1
                            w_high = w_low + 1

                            lh = h_im - h_low
                            lw = w_im - w_low
                            hh = 1 - lh
                            hw = 1 - lw

                            v1_cond = h_low >= 0 and w_low >= 0
                            v2_cond = h_low >= 0 and w_high <= W - 1
                            v3_cond = h_high <= H - 1 and w_low >= 0
                            v4_cond = h_high <= H - 1 and w_high <= W - 1

                            if v1_cond:
                                v1_pos = h_low * W + w_low
                            if v2_cond:
                                v2_pos = h_low * W + w_high
                            if v3_cond:
                                v3_pos = h_high * W + w_low
                            if v4_cond:
                                v4_pos = h_high * W + w_high

                            w1 = hh * hw
                            w2 = hh * lw
                            w3 = lh * hw
                            w4 = lh * lw

                        # 输入特征图的形状是[in_C, H, W]
                        data_im_ptr = data_im
                        for c_im in range(in_C):
                            val = 0.
                            if cond:
                                # Bilinear
                                v1 = 0.
                                if v1_cond:
                                    v1 = input[int(data_im_ptr + v1_pos)]
                                v2 = 0.
                                if v2_cond:
                                    v2 = input[int(data_im_ptr + v2_pos)]
                                v3 = 0.
                                if v3_cond:
                                    v3 = input[int(data_im_ptr + v3_pos)]
                                v4 = 0.
                                if v4_cond:
                                    v4 = input[int(data_im_ptr + v4_pos)]
                                val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                            im2col[data_col_ptr] = val * mask_
                            data_col_ptr += 1
                            data_im_ptr += H*W
        # 把im2col填写完毕。
        im2col = np.reshape(im2col, (out_H * out_W, kH * kW * in_C))   # [out_H * out_W, kH * kW * in_C]



        # ncnn_output = '../build/examples/output.txt'
        # with open(ncnn_output, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.strip()
        # line = line[:-1]
        # ss = line.split(',')
        # y = []
        # for s in ss:
        #     y.append(float(s))
        # y = np.array(y).astype(np.float32)
        # y = np.reshape(y, aaaaaaaaa2.shape)
        #
        # yy1 = y
        # yy2 = aaaaaaaaa2
        # ddd = np.sum((yy1 - yy2) ** 2)
        # print('ddd=%.9f' % ddd)


        # weight形状reshape成[out_C, in_C, kH * kW]，再转置成[out_C, kH * kW, in_C]，再reshape成[out_C, kH * kW * in_C]，再转置成[kH * kW * in_C, out_C]
        weight2 = np.reshape(weight, (out_C, in_C, kH * kW))
        weight2 = weight2.transpose(0, 2, 1)
        weight2 = np.reshape(weight2, (out_C, kH * kW * in_C))   # [out_C, kH * kW * in_C]
        weight2 = weight2.T     # [kH * kW * in_C, out_C]
        output = np.matmul(im2col, weight2)   # [out_H * out_W, out_C]
        if bias_term:
            bias = np.reshape(bias, (1, out_C))   # [1, out_C]
            output += bias
        output = output.T   # [out_C, out_H * out_W]
        output = np.reshape(output, (1, out_C, out_H, out_W))   # [1, out_C, out_H, out_W]
        return output

class DeformableConvCPUKernel_naive(object):
    def __init__(self):
        super(DeformableConvCPUKernel_naive, self).__init__()

    def __call__(self, input, offset, mask, weight, bias, stride):
        '''
https://github.com/PaddlePaddle/Paddle/blob/release/2.0/paddle/fluid/operators/deformable_conv_op.h

class DeformableConvCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
即为DCN前向代码。
        '''
        num_output = -99
        kernel_w = -99
        kernel_h = -99
        dilation_w = 1
        dilation_h = 1
        stride_w = stride
        stride_h = stride
        bias_term = True
        weight_data_size = -99

        offset = np.reshape(offset, (-1,))
        mask = np.reshape(mask, (-1,))

        in_C, H, W = input.shape
        input = np.reshape(input, (-1,))
        out_C, in_C, kH, kW = weight.shape
        num_output = out_C
        kernel_h = kH
        kernel_w = kW
        weight_data_size = out_C * in_C * kH * kW
        filter_size = kH
        paddings = (filter_size - 1) // 2
        pad_left = paddings
        pad_right = paddings
        pad_top = paddings
        pad_bottom = paddings

        kernel_extent_h = dilation_h * (kernel_h - 1) + 1
        kernel_extent_w = dilation_w * (kernel_w - 1) + 1
        out_H = (H + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1
        out_W = (W + pad_left + pad_right - kernel_extent_w) / stride_w + 1
        out_H = int(out_H)
        out_W = int(out_W)

        # output.shape is [num_output, out_h, out_w] (in python).
        output = np.zeros((out_W * out_H * num_output, ), dtype=np.float32)
        output_ptr = 0
        data_im = 0
        data_offset = 0
        data_mask = 0

        # deformable conv
        # 卷积窗滑动，并且多遍历了in_C这一维。
        for h_col in range(out_H):
            for w_col in range(out_W):
                h_in = h_col * stride_h - pad_top    # 卷积窗(左上角)现在的位置在原图中的y坐标
                w_in = w_col * stride_w - pad_left   # 卷积窗(左上角)现在的位置在原图中的x坐标

                output_hw_ptr = output_ptr + (h_col * out_W + w_col)
                for oc in range(out_C):
                    sum = 0
                    if bias_term:
                        sum = bias[oc]
                    data_offset_ptr = data_offset
                    data_mask_ptr = data_mask

                    # 遍历卷积核每一行每一列
                    for i in range(kH):
                        for j in range(kW):
                            '''
                // data_offset_ptr形状是[kH*kW*2, out_H, out_W], 第0维是yxyxyx...这样的顺序
                // reshape data_offset_ptr形状为[kH, kW, 2, out_H, out_W]
                // 这里是把 5D坐标[i, j, 0, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的y偏移
                            '''
                            data_offset_h_ptr = (((i * kW + j) * 2) * out_H + h_col) * out_W + w_col
                            '''
                // 这里是把 5D坐标[i, j, 1, h_col, w_col] 变成 1D坐标data_offset_h_ptr ，取出预测的x偏移
                            '''
                            data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_H + h_col) * out_W + w_col
                            '''
                // data_mask_ptr形状是[kH*kW, out_H, out_W],
                // reshape data_mask_ptr形状为[kH, kW, out_H, out_W]
                // 这里是把 4D坐标[i, j, h_col, w_col] 变成 1D坐标data_mask_hw_ptr ，取出预测的mask重要程度
                            '''
                            data_mask_hw_ptr = ((i * kernel_w + j) * out_H + h_col) * out_W + w_col

                            offset_h = offset[data_offset_ptr + data_offset_h_ptr]
                            offset_w = offset[data_offset_ptr + data_offset_w_ptr]
                            mask_ = mask[data_mask_ptr + data_mask_hw_ptr]
                            h_im = h_in + i * dilation_h + offset_h
                            w_im = w_in + j * dilation_w + offset_w
                            if oc == 0:
                                print(offset_h)
                                print(offset_w)
                                print(mask_)
                                print()
                            cond = h_im > -1 and w_im > -1 and h_im < H and w_im < W
                            if cond:
                                # Bilinear
                                h_low = np.floor(h_im)
                                w_low = np.floor(w_im)
                                h_high = h_low + 1
                                w_high = w_low + 1

                                lh = h_im - h_low
                                lw = w_im - w_low
                                hh = 1 - lh
                                hw = 1 - lw

                                v1_cond = h_low >= 0 and w_low >= 0
                                v2_cond = h_low >= 0 and w_high <= W - 1
                                v3_cond = h_high <= H - 1 and w_low >= 0
                                v4_cond = h_high <= H - 1 and w_high <= W - 1

                                if v1_cond:
                                    v1_pos = h_low * W + w_low
                                if v2_cond:
                                    v2_pos = h_low * W + w_high
                                if v3_cond:
                                    v3_pos = h_high * W + w_low
                                if v4_cond:
                                    v4_pos = h_high * W + w_high

                                w1 = hh * hw
                                w2 = hh * lw
                                w3 = lh * hw
                                w4 = lh * lw

                            # 输入特征图的形状是[in_C, H, W]
                            data_im_ptr = data_im
                            for c_im in range(in_C):
                                val = 0.
                                if cond:
                                    # Bilinear
                                    v1 = 0.
                                    if v1_cond:
                                        v1 = input[int(data_im_ptr + v1_pos)]
                                    v2 = 0.
                                    if v2_cond:
                                        v2 = input[int(data_im_ptr + v2_pos)]
                                    v3 = 0.
                                    if v3_cond:
                                        v3 = input[int(data_im_ptr + v3_pos)]
                                    v4 = 0.
                                    if v4_cond:
                                        v4 = input[int(data_im_ptr + v4_pos)]
                                    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                                aaaaaaa = val * mask_
                                # if h_col == 1 and w_col == 0 and oc == 0 and c_im == 0:
                                #     print(aaaaaaa)
                                #     print(weight[oc, c_im, i, j])
                                #     print()
                                sum += val * mask_ * weight[oc, c_im, i, j]
                                data_im_ptr += H*W
                    output[output_hw_ptr] = sum
                    output_hw_ptr += out_H * out_W
        output = np.reshape(output, (1, out_C, out_H, out_W))   # [1, out_C, out_H, out_W]
        return output


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

    def forward(self, inputs):
        if not self.dcn_v2:
            out = self.conv(inputs)
        else:
            offset_mask = self.conv_offset(inputs)
            offset = offset_mask[:, :self.offset_channel, :, :]
            mask = offset_mask[:, self.offset_channel:, :, :]
            mask = torch.sigmoid(mask)
            out = self.conv(inputs, offset, mask=mask)
        return offset, mask, out


img_h = 2
img_w = 2

# img_h = 8
# img_w = 8





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

# ch_in = 4
# ch_out = 8

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

filter_size = 2
stride = 1

# filter_size = 2
# stride = 2

# filter_size = 3
# stride = 1

# filter_size = 3
# stride = 2

# filter_size = 4
# stride = 1

# filter_size = 4
# stride = 2

# filter_size = 5
# stride = 1

# filter_size = 5
# stride = 5


model = ConvNormLayer2(ch_in, ch_out, filter_size=filter_size, stride=stride, dcn_v2=True)
model.eval()
state_dict = torch.load('11.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


dic2 = np.load('11.npz')
x = dic2['x']
x = torch.from_numpy(x)
x = torch.reshape(x, (1, ch_in, img_h, img_w))

offset, mask, y = model(x)

xx = x.cpu().detach().numpy()[0]
offset = offset.cpu().detach().numpy()
mask = mask.cpu().detach().numpy()

dcn_w = model.conv.weight.cpu().detach().numpy()
dcn_b = model.conv.bias.cpu().detach().numpy()


deformableConvCPUKernel222 = DeformableConvCPUKernel_naive()

y2 = deformableConvCPUKernel222(xx, offset, mask, dcn_w, dcn_b, stride=stride)


# yy1 = y.cpu().detach().numpy()
yy1 = dic2['y']
yy2 = y2
ddd = np.sum((yy1 - yy2) ** 2)
print('ddd=%.9f' % ddd)


print()
