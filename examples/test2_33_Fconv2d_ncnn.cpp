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
#include <math.h>

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
    int dims = m.dims;
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    printf("%s shape C=%d\n", name, C);
    printf("D=%d\n", D);
    printf("H=%d\n", H);
    printf("W=%d\n", W);
    printf("dims=%d\n", dims);
}



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




static int detect_PPYOLOE(const char* z_path, std::vector<float>& cls_scores, const char* param_path, const char* bin_path)
{
    // get input.   8 4 4 2 8 3 3 1 0
    int img_c = 8;
    int img_h = 4;
    int img_w = 4;
    int out_C = 2;
    int in_C = 8;
    int kH = 3;
    int kW = 3;
    int stride = 1;
    int padding = 0;



    FILE* fp = fopen(z_path, "rb");
    if (!fp)
    {
        printf("fopen %s failed", z_path);
        return -1;
    }
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);
    ncnn::Mat img = mb.load(img_c * img_h * img_w, 0);
    ncnn::Mat weight = mb.load(out_C * in_C * kH * kW, 1);
    fclose(fp);

//    img = img.reshape(img_w, img_h, 1, img_c);
    img = img.reshape(img_w, img_h, img_c);
    weight = weight.reshape(kW, kH, in_C, out_C);

    print_shape(img, "img");
    pretty_print(img);
    print_shape(weight, "weight");
//    pretty_print(weight);
//    save_data(img);
//    save_data(weight);

    ncnn::Net model;

//    model.opt.use_vulkan_compute = true;

    model.opt.use_vulkan_compute = false;
    model.opt.use_fp16_storage = false;
//    model.opt.use_fp16_packed = false;
//    model.opt.use_fp16_storage = false;
//    model.opt.use_fp16_arithmetic = false;

    model.register_custom_layer("Square", Square_layer_creator);
    model.register_custom_layer("Shell", Shell_layer_creator);

    model.load_param(param_path);
    model.load_model(bin_path);

    ncnn::Extractor ex = model.create_extractor();

    ex.input("images", img);

    ncnn::Mat out;
    ex.extract("output", out);
    print_shape(out, "out");
    pretty_print(out);
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

    const char* z_path = argv[1];
    const char* param_path = argv[2];
    const char* bin_path = argv[3];

//    const char* img_c = argv[4];
//    const char* img_h = argv[5];
//    const char* img_w = argv[6];
//    const char* out_C = argv[7];
//    const char* in_C = argv[8];
//    const char* kH = argv[9];
//    const char* kW = argv[10];
//    const char* stride = argv[11];
//    const char* padding = argv[12];

    std::vector<float> cls_scores;
    detect_PPYOLOE(z_path, cls_scores, param_path, bin_path);

    return 0;
}
