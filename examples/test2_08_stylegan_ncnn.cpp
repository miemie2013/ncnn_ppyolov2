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
        int dims = bottom_top_blob.dims;

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


// miemie2013: diffrent from AbsVal Layer, Abs supports 4-dim Tensor.
class Abs : public ncnn::Layer
{
public:
    Abs()
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
        int dims = bottom_top_blob.dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = -ptr[i];
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Abs)



class Rsqrt : public ncnn::Layer
{
public:
    Rsqrt()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        eps = pd.get(0, 0.f);
        scale = pd.get(1, 1.f);
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
//                printf(" %f\n", x);
//              out = 1.f / sqrt(eps + x * scale) = 1.f / sqrt(scale) * 1.f / sqrt(eps/scale + x)
                if (scale == 1.f)
                {
                    ptr[i] = static_cast<float>(1.f / sqrt(eps + x));
                }else
                {
                    float A = static_cast<float>(1.f / sqrt(scale));
                    float B = static_cast<float>(1.f / sqrt(eps/scale + x));
                    ptr[i] = static_cast<float>(A * B);
                }
            }
        }

        return 0;
    }
public:
    float eps;
    float scale;
};

DEFINE_LAYER_CREATOR(Rsqrt)


class Sqrt : public ncnn::Layer
{
public:
    Sqrt()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        eps = pd.get(0, 0.f);
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                ptr[i] = static_cast<float>(sqrt(eps + x));
            }
        }

        return 0;
    }
public:
    float eps;
};

DEFINE_LAYER_CREATOR(Sqrt)


#define PI acos(-1)
class Sin : public ncnn::Layer
{
public:
    Sin()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        scale = pd.get(0, 0.f);
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                ptr[i] = static_cast<float>(sin(scale * PI * x));
            }
        }

        return 0;
    }
public:
    float scale;
};

DEFINE_LAYER_CREATOR(Sin)


class Clamp : public ncnn::Layer
{
public:
    Clamp()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        min_v = pd.get(0, 0.f);
        max_v = pd.get(1, 1.f);
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                ptr[i] = std::min(max_v, std::max(min_v, x));
            }
        }

        return 0;
    }
public:
    float min_v;
    float max_v;
};

DEFINE_LAYER_CREATOR(Clamp)


class Lerp : public ncnn::Layer
{
public:
    Lerp()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob0 = bottom_blobs[0];
        const ncnn::Mat& bottom_blob1 = bottom_blobs[1];
        const ncnn::Mat& coeffs = bottom_blobs[2];
        int w = bottom_blob1.w;
        int h = bottom_blob1.h;
        int d = bottom_blob1.d;
        int channels = bottom_blob1.c;
        int size = w * h * d;

        size_t elemsize = bottom_blob1.elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr0 = bottom_blob0.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            const float* coeffs_ptr = coeffs.channel(q);
            float* out_ptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x0 = ptr0[i];
                float x1 = ptr1[i];
                float coeff = coeffs_ptr[0];
                out_ptr[i] = static_cast<float>(x0 + coeff * (x1 - x0));
            }
        }

        top_blob.dims = bottom_blob1.dims;

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Lerp)


class StyleMixingSwitcher : public ncnn::Layer
{
public:
    StyleMixingSwitcher()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        ws_i = pd.get(0, 0);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& ws0 = bottom_blobs[0];
        const ncnn::Mat& ws1 = bottom_blobs[1];
        const ncnn::Mat& mixing = bottom_blobs[2];
        int w = ws0.w;
        int h = ws0.h;
        int d = ws0.d;
        int channels = ws0.c;
        int size = w * h * d;

        size_t elemsize = ws0.elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr0 = ws0.channel(q);
            const float* ptr1 = ws1.channel(q);
            const float* mixing_ptr = mixing.channel(q);
            float* out_ptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x0 = ptr0[i];
                float x1 = ptr1[i];
                float mixing_ = mixing_ptr[ws_i];
                if (mixing_ > 0.5f)
                {
                    out_ptr[i] = x1;
                }else
                {
                    out_ptr[i] = x0;
                }
            }
        }

        top_blob.dims = ws0.dims;

        return 0;
    }
public:
    int ws_i;
};

DEFINE_LAYER_CREATOR(StyleMixingSwitcher)


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
        target_dims = pd.get(3, 4);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);
        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb)
    {
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
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const size_t elemsize = bottom_blob.elemsize;

        top_blobs[0].create(W, H, D, C, elemsize, opt.blob_allocator);
        if (top_blobs[0].empty())
            return -100;
        top_blobs[0] = weight_data;
//        print_shape(top_blobs[0], "top_blobs[0]");


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

        top_blobs[0].dims = target_dims;

        if (bias_term)
        {
            top_blobs[1] = bias_data;
        }
//        print_shape(top_blobs[0], "top_blobs[0]");
//        print_shape(bias_data, "bias_data");
        return 0;
    }
public:
    // param
    int C;
    int D;
    int H;
    int W;
    int target_dims;
    int bias_term;

    int weight_data_size;

    // model
    ncnn::Mat weight_data;
    ncnn::Mat bias_data;
};

DEFINE_LAYER_CREATOR(Shell)


class Fmatmul : public ncnn::Layer
{
public:
    Fmatmul()
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
        bias_term = pd.get(1, 0);
        weight_data_size = pd.get(2, 0);
        int8_scale_term = pd.get(8, 0);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, ncnn::Mat());

//        if (int8_scale_term)
//        {
//#if NCNN_INT8
//            support_int8_storage = true;
//#else
//            NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
//            return -1;
//#endif
//        }

        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        // call InnerProduct
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::InnerProduct);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output);
        pd.set(1, bias_term);
        pd.set(2, weight_data_size);
        pd.set(8, int8_scale_term);
        pd.set(9, activation_type);
        pd.set(10, activation_params);

        op->load_param(pd);

        // set weights
        ncnn::Mat weights[4];
        weights[0] = bottom_blobs[1];
        if (bias_term)
        {
            weights[1] = bottom_blobs[2];
        }

//#if NCNN_INT8
//        if (int8_scale_term)
//        {
//            weights[2] = weight_data_int8_scales;
//            weights[3] = bottom_blob_int8_scales;
//        }
//#endif

        op->load_model(ncnn::ModelBinFromMatArray(weights));

        op->create_pipeline(opt);

        // forward
        op->forward(bottom_blobs[0], top_blobs[0], opt);

        op->destroy_pipeline(opt);

        delete op;

        return 0;
    }
public:
    // param
    int num_output;
    int bias_term;

    int weight_data_size;

    int int8_scale_term;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    ncnn::Mat activation_params;
//#if NCNN_INT8
//    ncnn::Mat weight_data_int8_scales;
//    ncnn::Mat bottom_blob_int8_scales;
//#endif
};

DEFINE_LAYER_CREATOR(Fmatmul)


class BiasAct : public ncnn::Layer
{
public:
    BiasAct()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        act_type = pd.get(0, 0);
        alpha = pd.get(1, 0.f);
        gain = pd.get(2, 0.f);
        clamp = pd.get(3, 0.f);
        bias_term = pd.get(4, 1);
        return 0;
    }

    float bias_act(float x, float bias, float elu_alpha, float selu_alpha, float selu_lambda, float alphaxlambda) const
    {
        x = x + bias;
        if (act_type == 1)
        {  // relu
            if (x < 0)
                x = 0.f;
        }else if (act_type == 2)
        {  // lrelu
            if (x < 0)
                x *= alpha;
        }else if (act_type == 3)
        {  // tanh
            x = static_cast<float>(tanh(x));
        }else if (act_type == 4)
        {  // sigmoid
            x = static_cast<float>(1.f / (1.f + expf(-x)));
        }else if (act_type == 5)
        {  // elu
            if (x < 0.f)
                x = static_cast<float>(elu_alpha * (exp(x) - 1.f));
        }else if (act_type == 6)
        {  // selu
            if (x < 0.f)
                x = static_cast<float>((exp(x) - 1.f) * alphaxlambda);
            else
                x *= selu_alpha;
        }else if (act_type == 7)
        {  // softplus
            x = log(exp(x) + 1.0f);
        }else if (act_type == 8)
        {  // swish
            x = static_cast<float>(x / (1.f + expf(-x)));
        }

        // scale
        if (gain != 1.f)
            x *= gain;
        // clamp
        if (clamp >= 0.0f)
            x = std::min(clamp, std::max(-clamp, x));
        return x;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        // miemie2013: bottom_blob.dims may be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
        // miemie2013: bias_data.dims must be 1 (because it's from Shell Layer.)
//        print_shape(bottom_blob, "bottom_blob");
//        print_shape(bias_data, "bias_data");

        if (bias_term)
        {
            const ncnn::Mat& bias_data = bottom_blobs[1];
            int b_w = bias_data.w;
            int b_h = bias_data.h;
            int b_d = bias_data.d;
            int b_channels = bias_data.c;

            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int d = bottom_blob.d;
            int channels = bottom_blob.c;
            int size = w * h * d;

            int _111W_array_add_111W_bias = 0;
            int _C1HW_array_add_111W_bias = 0;
            if (channels == 1 && d == 1 && h == 1 && w != 1) {
                if (b_channels == 1 && b_d == 1 && b_h == 1 && b_w != 1) {
                    _111W_array_add_111W_bias = 1;
                }
            }else if (channels != 1 && d == 1 && h != 1 && w != 1) {
                if (b_channels == 1 && b_d == 1 && b_h == 1 && b_w != 1) {
                    _C1HW_array_add_111W_bias = 1;
                }
            }else {
                printf("not implemented.\n");
                return -100;
            }

            size_t elemsize = bottom_blobs[0].elemsize;
            ncnn::Mat& top_blob = top_blobs[0];
            top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            float elu_alpha = 0.1f;
            float selu_alpha = 1.67326324f;
            float selu_lambda = 1.050700987f;
            float alphaxlambda = selu_alpha * selu_lambda;


            if (_111W_array_add_111W_bias)
            {
                const float* bias_ptr = bias_data.channel(0);
                const float* in_ptr = bottom_blob.channel(0);
                float* out_ptr = top_blob.channel(0);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < size; i++)
                {
                    float bias = bias_data[i];
                    float x = in_ptr[i];
                    x = bias_act(x, bias, elu_alpha, selu_alpha, selu_lambda, alphaxlambda);
                    out_ptr[i] = x;
                }
                // miemie2013: you must set top_blobs[0].dims as bottom_blob.dims;
                top_blobs[0].dims = 1;
                return 0;
            }else if (_C1HW_array_add_111W_bias)
            {
                const float* bias_ptr = bias_data.channel(0);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* in_ptr = bottom_blob.channel(q);
                    float* out_ptr = top_blob.channel(q);
                    float bias = bias_ptr[q];

                    for (int i = 0; i < size; i++)
                    {
                        float x = in_ptr[i];
                        x = bias_act(x, bias, elu_alpha, selu_alpha, selu_lambda, alphaxlambda);
                        out_ptr[i] = x;
                    }
                }
                // miemie2013: you must set top_blobs[0].dims as bottom_blob.dims;
                top_blobs[0].dims = bottom_blob.dims;
                return 0;
            }else {
                printf("not implemented.\n");
                return -100;
            }
        }else
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int d = bottom_blob.d;
            int channels = bottom_blob.c;
            int size = w * h * d;

            size_t elemsize = bottom_blobs[0].elemsize;
            ncnn::Mat& top_blob = top_blobs[0];
            top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            float elu_alpha = 0.1f;
            float selu_alpha = 1.67326324f;
            float selu_lambda = 1.050700987f;
            float alphaxlambda = selu_alpha * selu_lambda;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* in_ptr = bottom_blob.channel(q);
                float* out_ptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    float x = in_ptr[i];
                    x = bias_act(x, 0.f, elu_alpha, selu_alpha, selu_lambda, alphaxlambda);
                    out_ptr[i] = x;
                }
            }
            // miemie2013: you must set top_blobs[0].dims as bottom_blob.dims;
            top_blobs[0].dims = bottom_blob.dims;
            return 0;
        }
        return 0;
    }
public:
    // param
    int act_type;
    float alpha;
    float gain;
    float clamp;
    int bias_term;
};

DEFINE_LAYER_CREATOR(BiasAct)



class F4DOp1D : public ncnn::Layer
{
public:
    F4DOp1D()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        dim = pd.get(0, 0);
        op_id = pd.get(1, 0);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& bias_data = bottom_blobs[1];

        int b_w = bias_data.w;
        int b_h = bias_data.h;
        int b_d = bias_data.d;
        int b_channels = bias_data.c;

        int _111W_bias = 0;
        int _C111_bias = 0;
        if (b_channels == 1 && b_d == 1 && b_h == 1) {
            _111W_bias = 1;
        }else if (b_d == 1 && b_h == 1 && b_w == 1) {
            _C111_bias = 1;
        }else {
            printf("not implemented.\n");
            return -100;
        }


        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;
        int size = w * h * d;
        int wh = w * h;


        int in_C = b_w;
        int out_C = channels / in_C;

        size_t elemsize = bottom_blobs[0].elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (_111W_bias)
        {
            if (dim == 1)
            {
                const float* bias_ptr = bias_data.channel(0);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* in_ptr = bottom_blob.channel(q);
                    float* out_ptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float bias = bias_ptr[i / wh];
                        if (op_id==0)
                        {
                            out_ptr[i] = in_ptr[i] * bias;
                        }else if(op_id==1)
                        {
                            out_ptr[i] = in_ptr[i] / bias;
                        }else if(op_id==2)
                        {
                            out_ptr[i] = in_ptr[i] + bias;
                        }else if(op_id==3)
                        {
                            out_ptr[i] = in_ptr[i] - bias;
                        }
                    }
                }
                return 0;
            }else if (dim == 0)
            {
                const float* bias_ptr = bias_data.channel(0);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* in_ptr = bottom_blob.channel(q);
                    float* out_ptr = top_blob.channel(q);
                    float bias = bias_ptr[q];

                    for (int i = 0; i < size; i++)
                    {
                        if (op_id==0)
                        {
                            out_ptr[i] = in_ptr[i] * bias;
                        }else if(op_id==1)
                        {
                            out_ptr[i] = in_ptr[i] / bias;
                        }else if(op_id==2)
                        {
                            out_ptr[i] = in_ptr[i] + bias;
                        }else if(op_id==3)
                        {
                            out_ptr[i] = in_ptr[i] - bias;
                        }
                    }
                }
                return 0;
            }else if (dim == 3)
            {
                const float* bias_ptr = bias_data.channel(0);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* in_ptr = bottom_blob.channel(q);
                    float* out_ptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        float bias = bias_ptr[i % w];
                        if (op_id==0)
                        {
                            out_ptr[i] = in_ptr[i] * bias;
                        }else if(op_id==1)
                        {
                            out_ptr[i] = in_ptr[i] / bias;
                        }else if(op_id==2)
                        {
                            out_ptr[i] = in_ptr[i] + bias;
                        }else if(op_id==3)
                        {
                            out_ptr[i] = in_ptr[i] - bias;
                        }
                    }
                }
                return 0;
            }else {
                printf("not implemented.\n");
                return -100;
            }
        }else if (_C111_bias)
        {
            if (dim == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* bias_ptr = bias_data.channel(q);
                    const float* in_ptr = bottom_blob.channel(q);
                    float* out_ptr = top_blob.channel(q);
                    float bias = bias_ptr[0];

                    for (int i = 0; i < size; i++)
                    {
                        if (op_id==0)
                        {
                            out_ptr[i] = in_ptr[i] * bias;
                        }else if(op_id==1)
                        {
                            out_ptr[i] = in_ptr[i] / bias;
                        }else if(op_id==2)
                        {
                            out_ptr[i] = in_ptr[i] + bias;
                        }else if(op_id==3)
                        {
                            out_ptr[i] = in_ptr[i] - bias;
                        }
                    }
                }
                return 0;
            }else {
                printf("not implemented.\n");
                return -100;
            }
        }
        return 0;
    }
public:
    // param
    int dim;
    int op_id;
};

DEFINE_LAYER_CREATOR(F4DOp1D)



class AddNoise : public ncnn::Layer
{
public:
    AddNoise()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& noise_data = bottom_blobs[1];
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
        // miemie2013: noise_data.dims must be 4 (because it's from Shell Layer.)
//        print_shape(bottom_blob, "bottom_blob");
//        print_shape(noise_data, "noise_data");

        int b_w = noise_data.w;
        int b_h = noise_data.h;
        int b_d = noise_data.d;
        int b_channels = noise_data.c;

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;
        int size = w * h * d;


        int in_C = b_w;
        int out_C = channels / in_C;

        size_t elemsize = bottom_blobs[0].elemsize;
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* noise_ptr = noise_data.channel(0);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float bias = noise_ptr[i];
                float x = in_ptr[i] + bias;
                out_ptr[i] = x;
            }
        }
        // miemie2013: you must set top_blobs[0].dims as bottom_blob.dims;
        top_blobs[0].dims = bottom_blob.dims;
        return 0;
    }
};

DEFINE_LAYER_CREATOR(AddNoise)


class MulConstant : public ncnn::Layer
{
public:
    MulConstant()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        scale = pd.get(0, 1.f);
        bias = pd.get(1, 0.f);
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int channels = bottom_top_blob.c;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                ptr[i] = static_cast<float>(scale * x + bias);
            }
        }

        return 0;
    }
public:
    float scale;
    float bias;
};

DEFINE_LAYER_CREATOR(MulConstant)



class FconvTranspose2d : public ncnn::Layer
{
public:
    FconvTranspose2d()
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
        num_input = pd.get(31, 0);
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
        output_pad_right = pd.get(18, 0);
        output_pad_bottom = pd.get(19, output_pad_right);
        output_w = pd.get(20, 0);
        output_h = pd.get(21, output_w);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, ncnn::Mat());
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const ncnn::Mat& weight_data = bottom_blobs[1];
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
        // miemie2013: weight_data.dims must be 4 (because it's from Shell Layer.)
//        print_shape(bottom_blob, "FconvTranspose2d bottom_blob");
//        print_shape(weight_data, "FconvTranspose2d weight_data");

        // miemie2013: In ncnn's Deconvolution Layer, weight must be flattened.
        ncnn::Mat weight_data2 = weight_data.reshape(kernel_w*kernel_h*num_input*num_output, 1, 1, 1);
        weight_data2.dims = 1;
//        print_shape(weight_data2, "FconvTranspose2d weight_data");

        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output);
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, pad_left);
        pd.set(15, pad_right);
        pd.set(14, pad_top);
        pd.set(16, pad_bottom);
        pd.set(18, output_pad_right);
        pd.set(19, output_pad_bottom);
        pd.set(20, output_w);
        pd.set(21, output_h);
        pd.set(5, bias_term);
        pd.set(6, weight_data_size);
        pd.set(9, activation_type);
        pd.set(10, activation_params);
        op->load_param(pd);

        // set weights
        ncnn::Mat weights[2];
        weights[0] = weight_data2;
        if (bias_term)
        {
            weights[1] = bottom_blobs[2];
        }
        op->load_model(ncnn::ModelBinFromMatArray(weights));

        op->create_pipeline(opt);
        op->forward(bottom_blob, top_blobs[0], opt);
        op->destroy_pipeline(opt);
        delete op;
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
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int output_pad_right;
    int output_pad_bottom;
    int output_w;
    int output_h;
    int bias_term;
    int num_input;

    int weight_data_size;

    // 0=none 1=relu 2=leakyrelu 3=clip 4=sigmoid
    int activation_type;
    ncnn::Mat activation_params;
};

DEFINE_LAYER_CREATOR(FconvTranspose2d)


class Down2 : public ncnn::Layer
{
public:
    Down2()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
//        print_shape(bottom_blob, "Down2 bottom_blob");
        int in_W = bottom_blob.w;
        int in_H = bottom_blob.h;
        int in_C = bottom_blob.c;

        int out_W = in_W / 2;
        int out_H = in_H / 2;
        int out_C = in_C;
        if (in_W % 2 == 1)
            out_W++;
        if (in_H % 2 == 1)
            out_H++;

        top_blob.create(out_W, out_H, out_C, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // miemie2013:
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_C; p++)
        {
            const float* input_ptr = bottom_blob.channel(p);
            float* output_ptr = top_blob.channel(p);

            for (int i = 0; i < out_H; i++)
            {
                for (int j = 0; j < out_W; j++)
                {
                    output_ptr[i * out_W + j] = input_ptr[i * 2 * in_W + j * 2];
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Down2)



class Up2 : public ncnn::Layer
{
public:
    Up2()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
//        print_shape(bottom_blob, "Up2 bottom_blob");
        int in_W = bottom_blob.w;
        int in_H = bottom_blob.h;
        int in_C = bottom_blob.c;

        int out_W = in_W * 2;
        int out_H = in_H * 2;
        int out_C = in_C;

        top_blob.create(out_W, out_H, out_C, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // miemie2013:
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_C; p++)
        {
            const float* input_ptr = bottom_blob.channel(p);
            float* output_ptr = top_blob.channel(p);

            for (int i = 0; i < out_H; i++)
            {
                for (int j = 0; j < out_W; j++)
                {
                    if (i % 2 == 0)
                    {
                        if (j % 2 == 0)
                        {
                            output_ptr[i * out_W + j] = input_ptr[i / 2 * in_W + j / 2];
                        }else if (j % 2 == 1)
                        {
                            output_ptr[i * out_W + j] = 0.f;
                        }
                    }else if (i % 2 == 1)
                    {
                        output_ptr[i * out_W + j] = 0.f;
                    }
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Up2)



class Up4 : public ncnn::Layer
{
public:
    Up4()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        // miemie2013: bottom_blob.dims must be 3 (because it's from StyleGANv2ADA_SynthesisNetwork's const.)
//        print_shape(bottom_blob, "Up4 bottom_blob");
        int in_W = bottom_blob.w;
        int in_H = bottom_blob.h;
        int in_C = bottom_blob.c;

        int out_W = in_W * 4;
        int out_H = in_H * 4;
        int out_C = in_C;

        top_blob.create(out_W, out_H, out_C, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // miemie2013:
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_C; p++)
        {
            const float* input_ptr = bottom_blob.channel(p);
            float* output_ptr = top_blob.channel(p);

            for (int i = 0; i < out_H; i++)
            {
                for (int j = 0; j < out_W; j++)
                {
                    if (i % 4 == 0)
                    {
                        if (j % 4 == 0)
                        {
                            output_ptr[i * out_W + j] = input_ptr[i / 4 * in_W + j / 4];
                        }else if (j % 4 == 1)
                        {
                            output_ptr[i * out_W + j] = 0.f;
                        }else if (j % 4 == 2)
                        {
                            output_ptr[i * out_W + j] = 0.f;
                        }else if (j % 4 == 3)
                        {
                            output_ptr[i * out_W + j] = 0.f;
                        }
                    }else if (i % 4 == 1)
                    {
                        output_ptr[i * out_W + j] = 0.f;
                    }else if (i % 4 == 2)
                    {
                        output_ptr[i * out_W + j] = 0.f;
                    }else if (i % 4 == 3)
                    {
                        output_ptr[i * out_W + j] = 0.f;
                    }
                }
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(Up4)


class Transforms : public ncnn::Layer
{
public:
    Transforms()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int in_W = bottom_blob.w;
        int in_H = bottom_blob.h;
        int in_C = bottom_blob.c;

        int out_W = 3;
        int out_H = 2;

        top_blob.create(out_W, out_H, 1, 1, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* input_ptr = bottom_blob.channel(0);
        float* output_ptr = top_blob.channel(0);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < out_H; i++)
        {
            for (int j = 0; j < out_W; j++)
            {
                if (i == 0)
                {
                    if (j == 0)
                    {
                        output_ptr[i * out_W + j] = input_ptr[0];
                    }else if (j == 1)
                    {
                        output_ptr[i * out_W + j] = 0.f - input_ptr[1];
                    }else if (j == 2)
                    {
                        output_ptr[i * out_W + j] = input_ptr[1] * input_ptr[3] - input_ptr[0] * input_ptr[2];
                    }
                }else if (i == 1)
                {
                    if (j == 0)
                    {
                        output_ptr[i * out_W + j] = input_ptr[1];
                    }else if (j == 1)
                    {
                        output_ptr[i * out_W + j] = input_ptr[0];
                    }else if (j == 2)
                    {
                        output_ptr[i * out_W + j] = 0.f - input_ptr[0] * input_ptr[3] - input_ptr[1] * input_ptr[2];
                    }
                }
            }
        }
        top_blob.dims = 2;
        return 0;
    }
};

DEFINE_LAYER_CREATOR(Transforms)


class StyleganPost : public ncnn::Layer
{
public:
    StyleganPost()
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
        int dims = bottom_top_blob.dims;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                float x = ptr[i];
                x = x * 127.5f + 128.f;
                x = std::min(255.f, std::max(0.f, x));
                ptr[i] = x;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(StyleganPost)



static int detect_PPYOLOE(const char* z_path, const char* param_path, const char* bin_path, int layer_type)
{
    // get input z.
    int z_dim = 512;
    FILE* fp = fopen(z_path, "rb");
    if (!fp)
    {
        printf("fopen %s failed", z_path);
        return -1;
    }
    ncnn::DataReaderFromStdio dr(fp);
    ncnn::ModelBinFromDataReader mb(dr);
    ncnn::Mat z = mb.load(z_dim, 0);
    fclose(fp);

    print_shape(z, "z");

    ncnn::Net model;

    model.opt.use_vulkan_compute = true;

//    model.opt.use_vulkan_compute = false;
//    model.opt.use_fp16_storage = false;
//    model.opt.use_fp16_packed = false;
//    model.opt.use_fp16_storage = false;
//    model.opt.use_fp16_arithmetic = false;

    model.register_custom_layer("Square", Square_layer_creator);
    model.register_custom_layer("Lerp", Lerp_layer_creator);
    model.register_custom_layer("Abs", Abs_layer_creator);
    model.register_custom_layer("Rsqrt", Rsqrt_layer_creator);
    model.register_custom_layer("Sqrt", Sqrt_layer_creator);
    model.register_custom_layer("Sin", Sin_layer_creator);
    model.register_custom_layer("Clamp", Clamp_layer_creator);
    model.register_custom_layer("Shell", Shell_layer_creator);
    model.register_custom_layer("Fmatmul", Fmatmul_layer_creator);
    model.register_custom_layer("BiasAct", BiasAct_layer_creator);
    model.register_custom_layer("F4DOp1D", F4DOp1D_layer_creator);
    model.register_custom_layer("AddNoise", AddNoise_layer_creator);
    model.register_custom_layer("MulConstant", MulConstant_layer_creator);
    model.register_custom_layer("FconvTranspose2d", FconvTranspose2d_layer_creator);
    model.register_custom_layer("Down2", Down2_layer_creator);
    model.register_custom_layer("StyleganPost", StyleganPost_layer_creator);
    model.register_custom_layer("StyleMixingSwitcher", StyleMixingSwitcher_layer_creator);
    model.register_custom_layer("Up2", Up2_layer_creator);
    model.register_custom_layer("Up4", Up4_layer_creator);
    model.register_custom_layer("Transforms", Transforms_layer_creator);

    model.load_param(param_path);
    model.load_model(bin_path);

    ncnn::Extractor ex = model.create_extractor();

    double ws_coeff = 1.0;
    float* coeff_data = new float[1];
    coeff_data[0] = (float)ws_coeff;
    ncnn::Mat coeff(1, coeff_data);

    int num_ws = 30;
    float* mixing_data = new float[num_ws];
    for(int i=0;i<num_ws;i++)
        mixing_data[i] = 0.0f;
    ncnn::Mat mixing(num_ws, mixing_data);

    if (layer_type == 1)
    {
        ex.input("z", z);
        ex.input("coeff", coeff);
    }else if (layer_type == 0)
    {
        ex.input("ws0", z);
        ex.input("ws1", z);
        ex.input("mixing", mixing);
    }else if (layer_type == 2)
    {
        ex.input("images", z);
    }

    ncnn::Mat out;
    ex.extract("output", out);
    print_shape(out, "out");
    save_data(out, "output.txt");

    return 0;
}


int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* z_path = argv[1];
    const char* param_path = argv[2];
    const char* bin_path = argv[3];
    int layer_type = atoi(argv[4]);

    detect_PPYOLOE(z_path, param_path, bin_path, layer_type);

    return 0;
}
