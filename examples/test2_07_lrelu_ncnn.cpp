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

void save_data(const ncnn::Mat& m)
{
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    FILE* fp = fopen("output.txt", "wb");
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
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    printf("%s shape C=%d\n", name, C);
    printf("D=%d\n", D);
    printf("H=%d\n", H);
    printf("W=%d\n", W);
}



class CoordConcat : public ncnn::Layer
{
public:
    CoordConcat()
    {
        one_blob_only = true;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        axis = pd.get(0, 0);
        return 0;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int dims = bottom_blob.dims;   // miemie2013: input tensor dims
        size_t elemsize = bottom_blob.elemsize;  // miemie2013: for example, one float32 element use 4 Bytes.
        int positive_axis = axis < 0 ? dims + axis : axis;

        if (dims == 3 && positive_axis == 0)
        {
            // miemie2013: like python, concat tensors whose shapes are [N, c1, H, W], [N, c2, H, W] with axis == 1 .  don't use #pragma omp parallel
            // concat dim
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int c = bottom_blob.c;

            // miemie2013: y = torch.cat([x, gx, gy], 1)   the last 2 channel is x and y
            int out_C = c + 2;

            top_blob.create(w, h, out_C, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            // miemie2013: the ori tensor is in the front. I use concat's code.
            size_t size = bottom_blob.cstep * c;
            const unsigned char* ptr = bottom_blob;
            unsigned char* outptr = top_blob.channel(0);
            memcpy(outptr, ptr, size * elemsize);


            float* x_ptr = top_blob.channel(out_C - 2);
            float* y_ptr = top_blob.channel(out_C - 1);
            // miemie2013: x and y is in the back. Use  #pragma omp parallel
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float y = (float)i / ((float)h - 1.f) * 2.f - 1.f;
                for (int j = 0; j < w; j++)
                {
                    float x = (float)j / ((float)w - 1.f) * 2.f - 1.f;
                    x_ptr[i * w + j] = x;
                    y_ptr[i * w + j] = y;
                }
            }

            return 0;
        }

        return 0;
    }
public:
    int axis;
};

DEFINE_LAYER_CREATOR(CoordConcat)


// refer to Convolution layer.
class Shell : public ncnn::Layer
{
public:
    Shell()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        C = pd.get(2, 1);
        D = pd.get(11, 1);
        H = pd.get(1, 1);
        W = pd.get(0, 1);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);
        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb)
    {
        printf("weight_data_size=%d\n", weight_data_size);
        weight_data = mb.load(weight_data_size, 0);
        if (weight_data.empty())
            return -100;

        if (bias_term)
        {
            bias_data = mb.load(C, 1);
            if (bias_data.empty())
                return -100;
        }
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        // refer to Split layer.
        printf("ccccccccccccccccccccccccccccccc \n");
        printf("%d \n", bottom_blobs.size());
        printf("%d \n", top_blobs.size());
        printf("C=%d \n", C);
        printf("D=%d \n", D);
        printf("H=%d \n", H);
        printf("W=%d \n", W);

        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const size_t elemsize = bottom_blob.elemsize;

        top_blobs[0].create(W, H, D, C, elemsize, opt.blob_allocator);
        if (top_blobs[0].empty())
            return -100;
        top_blobs[0] = weight_data;
        print_shape(top_blobs[0], "top_blobs[0]");


        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Reshape);

        // set param
        ncnn::ParamDict pd;
        pd.set(2, C);
        pd.set(11, D);
        pd.set(1, H);
        pd.set(0, W);
        op->load_param(pd);
        op->create_pipeline(opt);
        op->forward(top_blobs[0], top_blobs[0], opt);
        op->destroy_pipeline(opt);
        delete op;

        if (bias_term)
        {
            top_blobs[1] = bias_data;
        }
        print_shape(top_blobs[0], "top_blobs[0]");
        print_shape(bias_data, "bias_data");
        return 0;
    }
public:
    // param
    int C;
    int D;
    int H;
    int W;
    int bias_term;

    int weight_data_size;

    // model
    ncnn::Mat weight_data;
    ncnn::Mat bias_data;
};

DEFINE_LAYER_CREATOR(Shell)


class Square : public ncnn::Layer
{
public:
    Square()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        printf("Square input C=%d\n", channels);
        printf("input D=%d\n", d);
        printf("input H=%d\n", h);
        printf("input W=%d\n", w);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                ptr[i] = static_cast<float>(x * x);
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Square)





static int detect_PPYOLOE(const cv::Mat& bgr, std::vector<float>& cls_scores, const char* param_path, const char* bin_path)
{
    ncnn::Net model;

    model.opt.use_vulkan_compute = true;

    model.register_custom_layer("CoordConcat", CoordConcat_layer_creator);
    model.register_custom_layer("Shell", Shell_layer_creator);
    model.register_custom_layer("Square", Square_layer_creator);

    model.load_param(param_path);
    model.load_model(bin_path);

    // get ncnn::Mat with RGB format like PPYOLOE do.
    ncnn::Mat in_rgb = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    ncnn::Mat in_resize;
    // Interp image with cv2.INTER_CUBIC like PPYOLOE do.
    ncnn::resize_bicubic(in_rgb, in_resize, 6, 6);

    // Normalize image with the same mean and std like PPYOLOE do.
//    mean=[123.675, 116.28, 103.53]
//    std=[58.395, 57.12, 57.375]
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {1.0f/58.395f, 1.0f/57.12f, 1.0f/57.375f};
    in_resize.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = model.create_extractor();

    ex.input("images", in_resize);

    ncnn::Mat out;
    ex.extract("output", out);
//    pretty_print(out);
    print_shape(out, "out");
    save_data(out);

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
    detect_PPYOLOE(m, cls_scores, param_path, bin_path);

    return 0;
}
