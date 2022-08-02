// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "net.h"
#include "datareader.h"
#include "layer_type.h"
#include "modelbin.h"
#include "paramdict.h"

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <stdio.h>
#include <vector>

void pretty_print(const ncnn::Mat& m)
{
    int w = m.w;
    int h = m.h;
    int d = m.d;
    int channels = m.c;
    int size = w * h * d;

    for (int q = 0; q < channels; q++)
    {
        const float* ptr = m.channel(q);
        for (int i = 0; i < size; i++)
        {
            float x = ptr[i];
            printf("%f ", x);
        }
        printf("------------------------\n");
    }
}

void save_data(const ncnn::Mat& m, char* name)
{
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    FILE* fp = fopen(name, "wb");
    int size = W * H * D;
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int i = 0; i < size; i++)
        {
            fprintf(fp, "%e,", ptr[i]);
        }
    }
}

void print_shape(const ncnn::Mat& m, const char* name)
{
    int dims = m.dims;
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    printf("%s shape dims=%d\n", name, dims);
    printf("C=%d\n", C);
    printf("D=%d\n", D);
    printf("H=%d\n", H);
    printf("W=%d\n", W);
}




static float DmcnIm2colBilinear(const float* bottom_data, const int data_width, const int height, const int width, float h, float w) {
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh;
    float hw = 1 - lw;

    float v1 = (h_low >= 0 && w_low >= 0) ? bottom_data[h_low * data_width + w_low] : 0;
    float v2 = (h_low >= 0 && w_high <= width - 1) ? bottom_data[h_low * data_width + w_high] : 0;
    float v3 = (h_high <= height - 1 && w_low >= 0) ? bottom_data[h_high * data_width + w_low] : 0;
    float v4 = (h_high <= height - 1 && w_high <= width - 1) ? bottom_data[h_high * data_width + w_high] : 0;

    float w1 = hh * hw;
    float w2 = hh * lw;
    float w3 = lh * hw;
    float w4 = lh * lw;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}


class DeformableConv2d : public ncnn::Layer
{
public:
    DeformableConv2d()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        num_output = pd.get(0, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        dilation_w = pd.get(2, 1);
        dilation_h = pd.get(12, dilation_w);
        stride_w = pd.get(3, 1);
        stride_h = pd.get(13, stride_w);
        pad_left = pd.get(4, 0);
        pad_right = pd.get(15, pad_left);
        pad_top = pd.get(14, pad_left);
        pad_bottom = pd.get(16, pad_top);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, ncnn::Mat());
        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb)
    {
        weight_data = mb.load(weight_data_size, 0);
        if (weight_data.empty())
            return -100;

        if (bias_term)
        {
            bias_data = mb.load(num_output, 1);
            if (bias_data.empty())
                return -100;
        }

        const int in_c = weight_data_size / (num_output * kernel_h * kernel_w);
        const int M = in_c * kernel_h * kernel_w;
        weight_data = weight_data.reshape(M, num_output);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& offset = bottom_blobs[1];
        const ncnn::Mat& mask = bottom_blobs[2];

        const int w = bottom_blob.w;
        const int h = bottom_blob.h;
        const int in_c = bottom_blob.c;
        const size_t elemsize = bottom_blob.elemsize;

        const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
        const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

        const int out_w = (w + pad_left + pad_right - kernel_extent_w) / stride_w + 1;
        const int out_h = (h + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;

        // output = im2col matmul weight, im2col.shape is [out_h * out_w, in_c * kernel_h * kernel_w] (in python),
        // weight.shape is [num_output, in_c * kernel_h * kernel_w] (in python),
        // output.shape is [out_h * out_w, num_output] (in python).
        ncnn::Mat im2col;
        im2col.create(out_h * out_w * in_c * kernel_h * kernel_w, elemsize, opt.blob_allocator);
        if (im2col.empty())
            return -100;

        ncnn::Mat& output = top_blobs[0];
        output.create(num_output, out_h * out_w, elemsize, opt.blob_allocator);
        if (output.empty())
            return -100;

        ncnn::Mat bottom_blob_flatten = bottom_blob.reshape(w * h * in_c);
        ncnn::Mat offset_flatten = offset.reshape(offset.w * offset.h * offset.c);
        ncnn::Mat mask_flatten = mask.reshape(mask.w * mask.h * mask.c);
        const float* data_im_ptr = bottom_blob_flatten;
        const float* data_offset_ptr = offset_flatten;
        const float* data_mask_ptr = mask_flatten;
        float* im2col_ptr = im2col;

        // im2col
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int c_im = 0; c_im < in_c; c_im++)
        {
            for (int h_col = 0; h_col < out_h; h_col++)
            {
                for (int w_col = 0; w_col < out_w; w_col++)
                {
                    int c_col = c_im * kernel_h * kernel_w;
                    int h_in = h_col * stride_h - pad_top;
                    int w_in = w_col * stride_w - pad_left;
                    float* data_col_ptr = im2col_ptr + (h_col * out_w + w_col) * in_c * kernel_h * kernel_w + c_col;
                    const float* data_im_channel_ptr = data_im_ptr + c_im * h * w;
                    for (int i = 0; i < kernel_h; i++)
                    {
                        for (int j = 0; j < kernel_w; j++)
                        {
                            const int data_offset_h_ptr = (((i * kernel_w + j) * 2) * out_h + h_col) * out_w + w_col;
                            const int data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_h + h_col) * out_w + w_col;
                            const int data_mask_hw_ptr = ((i * kernel_w + j) * out_h + h_col) * out_w + w_col;

                            const float offset_h = data_offset_ptr[data_offset_h_ptr];
                            const float offset_w = data_offset_ptr[data_offset_w_ptr];
                            const float mask_ = data_mask_ptr[data_mask_hw_ptr];
                            float val = 0.f;
                            const float h_im = h_in + i * dilation_h + offset_h;
                            const float w_im = w_in + j * dilation_w + offset_w;
                            if (h_im > -1 && w_im > -1 && h_im < h && w_im < w) {
                                val = DmcnIm2colBilinear(data_im_channel_ptr, w, h, w, h_im, w_im);
                            }
                            *data_col_ptr = val * mask_;
                            data_col_ptr += 1;
                        }
                    }
                }
            }
        }
        im2col = im2col.reshape(in_c * kernel_h * kernel_w, out_h * out_w);

        // call InnerProduct
        ncnn::Layer* innerProduct = ncnn::create_layer(ncnn::LayerType::InnerProduct);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output);
        pd.set(1, bias_term);
        pd.set(2, weight_data_size);
        pd.set(9, activation_type);
        pd.set(10, activation_params);
        innerProduct->load_param(pd);

        // set weights
        ncnn::Mat weights[2];
        weights[0] = weight_data;
        if (bias_term)
        {
            weights[1] = bias_data;
        }
        innerProduct->load_model(ncnn::ModelBinFromMatArray(weights));
        innerProduct->create_pipeline(opt);

        // forward
        innerProduct->forward(im2col, output, opt);
        innerProduct->destroy_pipeline(opt);
        delete innerProduct;

        ncnn::Mat output_t;
        // call Permute
        ncnn::Layer* permute = ncnn::create_layer(ncnn::LayerType::Permute);

        // set param
        ncnn::ParamDict permute_pd;
        permute_pd.set(0, 1);
        permute->load_param(permute_pd);
        permute->create_pipeline(opt);
        // forward
        permute->forward(output, output_t, opt);
        permute->destroy_pipeline(opt);
        delete permute;
        output_t = output_t.reshape(out_w, out_h, num_output);
        top_blobs[0] = output_t;
        return 0;
    }
public:
    // param
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_left; // -233=SAME_UPPER -234=SAME_LOWER
    int pad_right;
    int pad_top;
    int pad_bottom;
    int bias_term;

    int weight_data_size;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    ncnn::Mat activation_params;

    // model
    ncnn::Mat weight_data;
    ncnn::Mat bias_data;
};

DEFINE_LAYER_CREATOR(DeformableConv2d)


static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores, const char* param_path, const char* bin_path)
{
    ncnn::Net model;

    model.opt.use_vulkan_compute = true;

//    model.register_custom_layer("DeformableConv2d", DeformableConv2d_layer_creator);

    model.load_param(param_path);
    model.load_model(bin_path);

//    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows);
//    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_RGB2BGR, bgr.cols, bgr.rows);
//    pretty_print(in);

//    mean = [117.3, 126.5, 130.2]
//    std = [108.4, 117.3, 127.6]
    const float mean_vals[3] = {117.3f, 126.5f, 130.2f};
    const float norm_vals[3] = {1.0f/108.4f, 1.0f/117.3f, 1.0f/127.6f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = model.create_extractor();

    ex.input("images", in);
    print_shape(in, "images");

    ncnn::Mat out;
    ex.extract("output", out);
    print_shape(out, "output");
    save_data(out, "output.txt");

    return 0;
}


int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* param_path = argv[2];
    const char* bin_path = argv[3];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores, param_path, bin_path);

    return 0;
}
