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
#include "benchmark.h"

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include <algorithm>
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

static void print_shape(const ncnn::Mat& m, const char* name)
{
    int dims = m.dims;
    int C = m.c;
    int D = m.d;
    int H = m.h;
    int W = m.w;
    size_t elemsize = m.elemsize;
    int elempack = m.elempack;
    size_t cstep = m.cstep;
    printf("%s shape dims=%d, C=%d, D=%d, H=%d, W=%d, elemsize=%d, elempack=%d, cstep=%d\n", name, dims, C, D, H, W, (int)elemsize, elempack, (int)cstep);
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


static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static int detect_squeezenet(const cv::Mat& bgr, const char* param_path, const char* bin_path)
{
//    opts[3].use_packing_layout = true;
//    opts[3].use_fp16_packed = true;
//    opts[3].use_fp16_storage = true;
//    opts[3].use_fp16_arithmetic = false; // FIXME enable me
//    opts[3].use_bf16_storage = false;
//    opts[3].use_shader_pack8 = true;
//    opts[3].use_image_storage = true;
//    opts[3].blob_allocator = &g_blob_pool_allocator;
//    opts[3].workspace_allocator = &g_workspace_pool_allocator;

    ncnn::Net model;
    printf("num_threads=%d\n", model.opt.num_threads);
    printf("use_packing_layout=%d\n", model.opt.use_packing_layout);
    printf("use_fp16_packed=%d\n", model.opt.use_fp16_packed);
    printf("use_fp16_storage=%d\n", model.opt.use_fp16_storage);
    printf("use_fp16_arithmetic=%d\n", model.opt.use_fp16_arithmetic);
    printf("use_shader_pack8=%d\n", model.opt.use_shader_pack8);
    printf("use_image_storage=%d\n", model.opt.use_image_storage);

    model.opt.use_vulkan_compute = false;
    model.opt.use_packing_layout = false;
    model.opt.use_packing_layout = true;
    model.opt.use_sgemm_convolution = false;
    model.opt.use_sgemm_convolution = true;

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

    ncnn::Mat out;

    ncnn::Extractor ex = model.create_extractor();

    ex.input("images", in);
    print_shape(in, "images");

    ex.extract("output", out);
    print_shape(out, "output");
    save_data(out, "output.txt");

    return 0;
}

static int detect_aaa(const char* input_path, const int C, const int H, const int W, const char* param_path, const char* bin_path, const int use_packing_layout, const int use_sgemm_convolution)
{
//    opts[3].use_packing_layout = true;
//    opts[3].use_fp16_packed = true;
//    opts[3].use_fp16_storage = true;
//    opts[3].use_fp16_arithmetic = false; // FIXME enable me
//    opts[3].use_bf16_storage = false;
//    opts[3].use_shader_pack8 = true;
//    opts[3].use_image_storage = true;
//    opts[3].blob_allocator = &g_blob_pool_allocator;
//    opts[3].workspace_allocator = &g_workspace_pool_allocator;

    ncnn::Net model;
    printf("num_threads=%d\n", model.opt.num_threads);
    printf("use_packing_layout=%d\n", model.opt.use_packing_layout);
    printf("use_fp16_packed=%d\n", model.opt.use_fp16_packed);
    printf("use_fp16_storage=%d\n", model.opt.use_fp16_storage);
    printf("use_fp16_arithmetic=%d\n", model.opt.use_fp16_arithmetic);
    printf("use_shader_pack8=%d\n", model.opt.use_shader_pack8);
    printf("use_image_storage=%d\n", model.opt.use_image_storage);

    model.opt.use_vulkan_compute = false;
    model.opt.use_packing_layout = use_packing_layout;
    model.opt.use_sgemm_convolution = use_sgemm_convolution;

    model.load_param(param_path);
    model.load_model(bin_path);

    // get input.
    FILE* fp = fopen(input_path, "rb");
    if (!fp)
    {
        printf("fopen %s failed", input_path);
        return -1;
    }
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);
    ncnn::Mat in0 = mb.load(C*H*W, 0);
    fclose(fp);
    in0 = in0.reshape(W, H, C);
    pretty_print(in0);

    ncnn::Mat out;

    ncnn::Extractor ex = model.create_extractor();

    ex.input("images", in0);
    print_shape(in0, "images");

    ex.extract("output", out);
    print_shape(out, "output");
    save_data(out, "output.txt");

    return 0;
}


int main(int argc, char** argv)
{

    int func_id = atoi(argv[1]);
    const char* imagepath = argv[2];


    if (func_id == 0)
    {
        const char* param_path = argv[3];
        const char* bin_path = argv[4];
        cv::Mat m = cv::imread(imagepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
        detect_squeezenet(m, param_path, bin_path);
    }else if (func_id == 1)
    {
        const char* param_path = argv[3];
        const char* bin_path = argv[4];
        int C = atoi(argv[5]);
        int H = atoi(argv[6]);
        int W = atoi(argv[7]);
        int use_packing_layout = atoi(argv[8]);
        int use_sgemm_convolution = atoi(argv[9]);
        detect_aaa(imagepath, C, H, W, param_path, bin_path, use_packing_layout, use_sgemm_convolution);
    }


    return 0;
}
