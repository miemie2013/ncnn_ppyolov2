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


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct Bbox
{
    float x0;
    float y0;
    float x1;
    float y1;
    int clsid;
    float score;
};

struct Score
{
    int index;
    float score;
};

bool compare_desc(Bbox bbox1, Bbox bbox2)
{
    return bbox1.score > bbox2.score;
}

float calc_iou(Bbox bbox1, Bbox bbox2)
{
    float area_1 = (bbox1.y1 - bbox1.y0) * (bbox1.x1 - bbox1.x0);
    float area_2 = (bbox2.y1 - bbox2.y0) * (bbox2.x1 - bbox2.x0);
    float inter_x0 = std::max(bbox1.x0, bbox2.x0);
    float inter_y0 = std::max(bbox1.y0, bbox2.y0);
    float inter_x1 = std::min(bbox1.x1, bbox2.x1);
    float inter_y1 = std::min(bbox1.y1, bbox2.y1);
    float inter_w = std::max(0.f, inter_x1 - inter_x0);
    float inter_h = std::max(0.f, inter_y1 - inter_y0);
    float inter_area = inter_w * inter_h;
    float union_area = area_1 + area_2 - inter_area + 0.000000001f;
    return inter_area / union_area;
}

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

class PPYOLODecode : public ncnn::Layer
{
public:
    PPYOLODecode()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        num_classes = pd.get(0, 80);
        anchors = pd.get(1, ncnn::Mat());
        strides = pd.get(2, ncnn::Mat());
        scale_x_y = pd.get(3, 1.f);
        iou_aware_factor = pd.get(4, 0.5f);
        obj_thr = pd.get(5, 0.1f);
        anchor_per_stride = pd.get(6, 3);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const int tensor_num = bottom_blobs.size() - 1;
        const size_t elemsize = bottom_blob.elemsize;
        const ncnn::Mat& im_scale = bottom_blobs[tensor_num];
        const float scale_x = im_scale[0];
        const float scale_y = im_scale[1];

        int out_num = 0;
        for (size_t b = 0; b < tensor_num; b++)
        {
            const ncnn::Mat& tensor = bottom_blobs[b];
            const int w = tensor.w;
            const int h = tensor.h;
            out_num += anchor_per_stride * h * w;
        }

        ncnn::Mat& bboxes = top_blobs[0];
        bboxes.create(4 * out_num, elemsize, opt.blob_allocator);
        if (bboxes.empty())
            return -100;

        ncnn::Mat& scores = top_blobs[1];
        scores.create(num_classes * out_num, elemsize, opt.blob_allocator);
        if (scores.empty())
            return -100;
        float* bboxes_ptr = bboxes;
        float* scores_ptr = scores;

        // decode
        for (size_t b = 0; b < tensor_num; b++)
        {
            const ncnn::Mat& tensor = bottom_blobs[b];
            const int w = tensor.w;
            const int h = tensor.h;
            const int c = tensor.c;
            const bool use_iou_aware = (c == anchor_per_stride * (num_classes + 6));
            const int channel_stride = use_iou_aware ? (c / anchor_per_stride) - 1 : (c / anchor_per_stride);
            const int cx_pos = use_iou_aware ? anchor_per_stride : 0;
            const int cy_pos = use_iou_aware ? anchor_per_stride + 1 : 1;
            const int w_pos = use_iou_aware ? anchor_per_stride + 2 : 2;
            const int h_pos = use_iou_aware ? anchor_per_stride + 3 : 3;
            const int obj_pos = use_iou_aware ? anchor_per_stride + 4 : 4;
            const int cls_pos = use_iou_aware ? anchor_per_stride + 5 : 5;
            float stride = strides[b];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    for (int k = 0; k < anchor_per_stride; k++)
                    {
                        float obj = tensor.channel(obj_pos + k * channel_stride).row(i)[j];
                        obj = static_cast<float>(1.f / (1.f + expf(-obj)));
                        if (use_iou_aware)
                        {
                            float ioup = tensor.channel(k).row(i)[j];
                            ioup = static_cast<float>(1.f / (1.f + expf(-ioup)));
                            obj = static_cast<float>(pow(obj, 1.f - iou_aware_factor) * pow(ioup, iou_aware_factor));
                        }
                        if (obj > obj_thr)
                        {
                            // Grid Sensitive
                            float cx = static_cast<float>(scale_x_y / (1.f + expf(-tensor.channel(cx_pos + k * channel_stride).row(i)[j])) + j - (scale_x_y - 1.f) * 0.5f);
                            float cy = static_cast<float>(scale_x_y / (1.f + expf(-tensor.channel(cy_pos + k * channel_stride).row(i)[j])) + i - (scale_x_y - 1.f) * 0.5f);
                            cx *= stride;
                            cy *= stride;
                            float dw = static_cast<float>(expf(tensor.channel(w_pos + k * channel_stride).row(i)[j]) * anchors[(b * anchor_per_stride + k) * 2]);
                            float dh = static_cast<float>(expf(tensor.channel(h_pos + k * channel_stride).row(i)[j]) * anchors[(b * anchor_per_stride + k) * 2 + 1]);
                            float x0 = cx - dw * 0.5f;
                            float y0 = cy - dh * 0.5f;
                            float x1 = cx + dw * 0.5f;
                            float y1 = cy + dh * 0.5f;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4] = x0 / scale_x;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 1] = y0 / scale_y;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 2] = x1 / scale_x;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 3] = y1 / scale_y;
                            for (int r = 0; r < num_classes; r++)
                            {
                                float score = static_cast<float>(obj / (1.f + expf(-tensor.channel(cls_pos + k * channel_stride + r).row(i)[j])));
                                scores_ptr[((i * w + j) * anchor_per_stride + k) * num_classes + r] = score;
                            }
                        }else
                        {
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4] = 0.f;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 1] = 0.f;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 2] = 1.f;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 3] = 1.f;
                            for (int r = 0; r < num_classes; r++)
                            {
                                scores_ptr[((i * w + j) * anchor_per_stride + k) * num_classes + r] = -1.f;
                            }
                        }
                    }
                }
            }
            bboxes_ptr += h * w * anchor_per_stride * 4;
            scores_ptr += h * w * anchor_per_stride * num_classes;
        }
        bboxes = bboxes.reshape(4, out_num);
        scores = scores.reshape(num_classes, out_num);
        return 0;
    }
public:
    // param
    int num_classes;
    ncnn::Mat anchors;
    ncnn::Mat strides;
    float scale_x_y;
    float iou_aware_factor;
    float obj_thr;
    int anchor_per_stride;
};

DEFINE_LAYER_CREATOR(PPYOLODecode)

class PPYOLODecodeMatrixNMS : public ncnn::Layer
{
public:
    PPYOLODecodeMatrixNMS()
    {
        // miemie2013: if num of input tensors > 1 or num of output tensors > 1, you must set one_blob_only = false
        // And ncnn will use forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) method
        // or forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) method
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        num_classes = pd.get(0, 80);
        anchors = pd.get(1, ncnn::Mat());
        strides = pd.get(2, ncnn::Mat());
        scale_x_y = pd.get(3, 1.f);
        iou_aware_factor = pd.get(4, 0.5f);
        score_threshold = pd.get(5, 0.1f);
        anchor_per_stride = pd.get(6, 3);
        post_threshold = pd.get(7, 0.1f);
        nms_top_k = pd.get(8, 500);
        keep_top_k = pd.get(9, 100);
        kernel = pd.get(10, 0);
        gaussian_sigma = pd.get(11, 2.f);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& bottom_blob = bottom_blobs[0];
        const int tensor_num = bottom_blobs.size() - 1;
        const size_t elemsize = bottom_blob.elemsize;
        const ncnn::Mat& im_scale = bottom_blobs[tensor_num];
        const float scale_x = im_scale[0];
        const float scale_y = im_scale[1];

        int out_num = 0;
        for (size_t b = 0; b < tensor_num; b++)
        {
            const ncnn::Mat& tensor = bottom_blobs[b];
            const int w = tensor.w;
            const int h = tensor.h;
            out_num += anchor_per_stride * h * w;
        }

        ncnn::Mat bboxes;
        bboxes.create(4 * out_num, elemsize, opt.blob_allocator);
        if (bboxes.empty())
            return -100;

        ncnn::Mat scores;
        scores.create(num_classes * out_num, elemsize, opt.blob_allocator);
        if (scores.empty())
            return -100;
        float* bboxes_ptr = bboxes;
        float* scores_ptr = scores;

        // decode
        for (size_t b = 0; b < tensor_num; b++)
        {
            const ncnn::Mat& tensor = bottom_blobs[b];
            const int w = tensor.w;
            const int h = tensor.h;
            const int c = tensor.c;
            const bool use_iou_aware = (c == anchor_per_stride * (num_classes + 6));
            const int channel_stride = use_iou_aware ? (c / anchor_per_stride) - 1 : (c / anchor_per_stride);
            const int cx_pos = use_iou_aware ? anchor_per_stride : 0;
            const int cy_pos = use_iou_aware ? anchor_per_stride + 1 : 1;
            const int w_pos = use_iou_aware ? anchor_per_stride + 2 : 2;
            const int h_pos = use_iou_aware ? anchor_per_stride + 3 : 3;
            const int obj_pos = use_iou_aware ? anchor_per_stride + 4 : 4;
            const int cls_pos = use_iou_aware ? anchor_per_stride + 5 : 5;
            float stride = strides[b];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    for (int k = 0; k < anchor_per_stride; k++)
                    {
                        float obj = tensor.channel(obj_pos + k * channel_stride).row(i)[j];
                        obj = static_cast<float>(1.f / (1.f + expf(-obj)));
                        if (use_iou_aware)
                        {
                            float ioup = tensor.channel(k).row(i)[j];
                            ioup = static_cast<float>(1.f / (1.f + expf(-ioup)));
                            obj = static_cast<float>(pow(obj, 1.f - iou_aware_factor) * pow(ioup, iou_aware_factor));
                        }
                        if (obj > score_threshold)
                        {
                            // Grid Sensitive
                            float cx = static_cast<float>(scale_x_y / (1.f + expf(-tensor.channel(cx_pos + k * channel_stride).row(i)[j])) + j - (scale_x_y - 1.f) * 0.5f);
                            float cy = static_cast<float>(scale_x_y / (1.f + expf(-tensor.channel(cy_pos + k * channel_stride).row(i)[j])) + i - (scale_x_y - 1.f) * 0.5f);
                            cx *= stride;
                            cy *= stride;
                            float dw = static_cast<float>(expf(tensor.channel(w_pos + k * channel_stride).row(i)[j]) * anchors[(b * anchor_per_stride + k) * 2]);
                            float dh = static_cast<float>(expf(tensor.channel(h_pos + k * channel_stride).row(i)[j]) * anchors[(b * anchor_per_stride + k) * 2 + 1]);
                            float x0 = cx - dw * 0.5f;
                            float y0 = cy - dh * 0.5f;
                            float x1 = cx + dw * 0.5f;
                            float y1 = cy + dh * 0.5f;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4] = x0 / scale_x;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 1] = y0 / scale_y;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 2] = x1 / scale_x;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 3] = y1 / scale_y;
                            for (int r = 0; r < num_classes; r++)
                            {
                                float score = static_cast<float>(obj / (1.f + expf(-tensor.channel(cls_pos + k * channel_stride + r).row(i)[j])));
                                scores_ptr[((i * w + j) * anchor_per_stride + k) * num_classes + r] = score;
                            }
                        }else
                        {
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4] = 0.f;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 1] = 0.f;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 2] = 1.f;
                            bboxes_ptr[((i * w + j) * anchor_per_stride + k) * 4 + 3] = 1.f;
                            for (int r = 0; r < num_classes; r++)
                            {
                                scores_ptr[((i * w + j) * anchor_per_stride + k) * num_classes + r] = -1.f;
                            }
                        }
                    }
                }
            }
            bboxes_ptr += h * w * anchor_per_stride * 4;
            scores_ptr += h * w * anchor_per_stride * num_classes;
        }

        // keep bbox whose score > score_threshold
        std::vector<Bbox> bboxes_vec;
        for (int i = 0; i < out_num; i++)
        {
            float x0 = bboxes[i * 4];
            float y0 = bboxes[i * 4 + 1];
            float x1 = bboxes[i * 4 + 2];
            float y1 = bboxes[i * 4 + 3];
            for (int j = 0; j < num_classes; j++)
            {
                float score = scores[i * num_classes + j];
                if (score > score_threshold)
                {
                    Bbox bbox;
                    bbox.x0 = x0;
                    bbox.y0 = y0;
                    bbox.x1 = x1;
                    bbox.y1 = y1;
                    bbox.clsid = j;
                    bbox.score = score;
                    bboxes_vec.push_back(bbox);
                }
            }
        }
        if (bboxes_vec.size() == 0)
        {
            ncnn::Mat& pred = top_blobs[0];
            pred.create(0, 0, elemsize, opt.blob_allocator);
            if (pred.empty())
                return -100;
            return 0;
        }
        // sort and keep top nms_top_k
        int nms_top_k_ = nms_top_k;
        if (bboxes_vec.size() < nms_top_k)
            nms_top_k_ = bboxes_vec.size();
        size_t count {(size_t)nms_top_k_};
        std::partial_sort(std::begin(bboxes_vec), std::begin(bboxes_vec) + count, std::end(bboxes_vec), compare_desc);
        if (bboxes_vec.size() > nms_top_k)
            bboxes_vec.resize(nms_top_k);

        // ---------------------- Matrix NMS ----------------------
        // calc a iou matrix whose shape is [n, n], n is bboxes_vec.size()
        int n = bboxes_vec.size();
        float* decay_iou = new float[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j < i + 1)
                {
                    decay_iou[i * n + j] = 0.f;
                }else
                {
                    bool same_clsid = bboxes_vec[i].clsid == bboxes_vec[j].clsid;
                    if (same_clsid)
                    {
                        float iou = calc_iou(bboxes_vec[i], bboxes_vec[j]);
                        decay_iou[i * n + j] = iou;
                    }else
                    {
                        decay_iou[i * n + j] = 0.f;
                    }
                }
            }
        }

        // get max iou of each col
        float* compensate_iou = new float[n];
        for (int i = 0; i < n; i++)
        {
            float max_iou = decay_iou[i];
            for (int j = 0; j < n; j++)
            {
                if (decay_iou[j * n + i] > max_iou)
                    max_iou = decay_iou[j * n + i];
            }
            compensate_iou[i] = max_iou;
        }

        float* decay_matrix = new float[n * n];
        // get min decay_value of each col
        float* decay_coefficient = new float[n];

        if (kernel == 0) // gaussian
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    decay_matrix[i * n + j] = static_cast<float>(expf(gaussian_sigma * (compensate_iou[i] * compensate_iou[i] - decay_iou[i * n + j] * decay_iou[i * n + j])));
                }
            }
        }else if (kernel == 1) // linear
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    decay_matrix[i * n + j] = (1.f  - decay_iou[i * n + j]) / (1.f  - compensate_iou[i]);
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            float min_v = decay_matrix[i];
            for (int j = 0; j < n; j++)
            {
                if (decay_matrix[j * n + i] < min_v)
                    min_v = decay_matrix[j * n + i];
            }
            decay_coefficient[i] = min_v;
        }
        for (int i = 0; i < n; i++)
        {
            bboxes_vec[i].score *= decay_coefficient[i];
        }
        // ---------------------- Matrix NMS (end) ----------------------

        std::vector<Bbox> bboxes_vec_keep;
        for (int i = 0; i < n; i++)
        {
            if (bboxes_vec[i].score > post_threshold)
            {
                bboxes_vec_keep.push_back(bboxes_vec[i]);
            }
        }
        n = bboxes_vec_keep.size();
        if (n == 0)
        {
            ncnn::Mat& pred = top_blobs[0];
            pred.create(0, 0, elemsize, opt.blob_allocator);
            if (pred.empty())
                return -100;
            return 0;
        }
        // sort and keep keep_top_k
        int keep_top_k_ = keep_top_k;
        if (n < keep_top_k)
            keep_top_k_ = n;
        size_t keep_count {(size_t)keep_top_k_};
        std::partial_sort(std::begin(bboxes_vec_keep), std::begin(bboxes_vec_keep) + keep_count, std::end(bboxes_vec_keep), compare_desc);
        if (bboxes_vec_keep.size() > keep_top_k)
            bboxes_vec_keep.resize(keep_top_k);

        ncnn::Mat& pred = top_blobs[0];
        pred.create(6 * n, elemsize, opt.blob_allocator);
        if (pred.empty())
            return -100;
        float* pred_ptr = pred;
        for (int i = 0; i < n; i++)
        {
            pred_ptr[i * 6] = (float)bboxes_vec_keep[i].clsid;
            pred_ptr[i * 6 + 1] = bboxes_vec_keep[i].score;
            pred_ptr[i * 6 + 2] = bboxes_vec_keep[i].x0;
            pred_ptr[i * 6 + 3] = bboxes_vec_keep[i].y0;
            pred_ptr[i * 6 + 4] = bboxes_vec_keep[i].x1;
            pred_ptr[i * 6 + 5] = bboxes_vec_keep[i].y1;
        }
        pred = pred.reshape(6, n);
        return 0;
    }
public:
    // param
    int num_classes;
    ncnn::Mat anchors;
    ncnn::Mat strides;
    float scale_x_y;
    float iou_aware_factor;
    float score_threshold;
    int anchor_per_stride;
    float post_threshold;
    int nms_top_k;
    int keep_top_k;
    int kernel; // 0=gaussian 1=linear
    float gaussian_sigma;
};

DEFINE_LAYER_CREATOR(PPYOLODecodeMatrixNMS)


static void generate_ppyolo_proposals(const ncnn::Mat& bboxes, const ncnn::Mat& scores, float prob_threshold, std::vector<Object>& objects)
{
    // python中bboxes 的形状是[N, A,  4], ncnn中C=1, H=A=预测框数, W= 4
    // python中scores 的形状是[N, A, 80], ncnn中C=1, H=A=预测框数, W=80
    int C = scores.c;
    int H = scores.h;
    int W = scores.w;
    int num_bbox = H;
    int num_class = W;

    // 因为二者的C都只等于1，所以取第0个
    const float* cls_score_ptr = scores.channel(0);
    const float* bbox_ptr = bboxes.channel(0);

    for (int anchor_idx = 0; anchor_idx < num_bbox; anchor_idx++)
    {
        float x0 = bbox_ptr[0];
        float y0 = bbox_ptr[1];
        float x1 = bbox_ptr[2];
        float y1 = bbox_ptr[3];
        float h = y1 - y0;
        float w = x1 - x0;

        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_prob = cls_score_ptr[class_idx];
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
        cls_score_ptr += num_class;
        bbox_ptr += 4;
    }
}

static void get_ppyolo_out(const ncnn::Mat& pred, std::vector<Object>& objects)
{
    // pred形状是[n, 6]
    int H = pred.h;
    int W = pred.w;
    int num_bbox = H;
    int num_class = W;

    const float* pred_ptr = pred.channel(0);

    for (int i = 0; i < num_bbox; i++)
    {
        int clsid = (int)pred_ptr[0];
        float score = pred_ptr[1];
        float x0 = pred_ptr[2];
        float y0 = pred_ptr[3];
        float x1 = pred_ptr[4];
        float y1 = pred_ptr[5];
        float h = y1 - y0;
        float w = x1 - x0;
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = clsid;
        obj.prob = score;
        objects.push_back(obj);
        pred_ptr += 6;
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}




static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static int detect(const cv::Mat& bgr, std::vector<Object>& objects, const char* param_path, const char* bin_path, const int target_size)
{
    ncnn::Net model;

    model.opt.use_vulkan_compute = true;

    model.register_custom_layer("CoordConcat", CoordConcat_layer_creator);
    model.register_custom_layer("PPYOLODecode", PPYOLODecode_layer_creator);
    model.register_custom_layer("PPYOLODecodeMatrixNMS", PPYOLODecodeMatrixNMS_layer_creator);

    model.load_param(param_path);
    model.load_model(bin_path);

    int img_w = bgr.cols;
    int img_h = bgr.rows;
    float scale_x = (float)target_size / img_w;
    float scale_y = (float)target_size / img_h;

    // get ncnn::Mat with RGB format like PPYOLOv2 do.
    ncnn::Mat in_rgb = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    ncnn::Mat in_resize;
    // Interp image with cv2.INTER_CUBIC like PPYOLOv2 do.
    ncnn::resize_bicubic(in_rgb, in_resize, target_size, target_size);

    // Normalize image with the same mean and std like PPYOLOv2 do.
//    mean=[123.675, 116.28, 103.53]
//    std=[58.395, 57.12, 57.375]
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {1.0f/58.395f, 1.0f/57.12f, 1.0f/57.375f};
    in_resize.substract_mean_normalize(mean_vals, norm_vals);

    float* scale_data = new float[2];
    scale_data[0] = scale_x;
    scale_data[1] = scale_y;
    ncnn::Mat im_scale(2, scale_data);


    if (g_enable_cooling_down)
    {
        // sleep 2 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(2 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        sleep(2);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 2;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }

    ncnn::Mat pred;   // 形状是[n, 6]

    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = model.create_extractor();
        ex.input("images", in_resize);
        ex.input("im_scale", im_scale);
        ex.extract("pred", pred);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();

        {
            ncnn::Extractor ex = model.create_extractor();
            ex.input("images", in_resize);
            ex.input("im_scale", im_scale);
            ex.extract("pred", pred);
        }

        double end = ncnn::get_current_time();

        double time = end - start;

        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    printf("%s spend time:  min = %7.2f  max = %7.2f  avg = %7.2f\n", param_path, time_min, time_max, time_avg);

    ncnn::Extractor ex = model.create_extractor();

    ex.input("images", in_resize);
//    ex.input("in0", in_resize);
    ex.input("im_scale", im_scale);

    // Debug
//    ncnn::Mat out;
//    ex.extract("output", out);
//    ex.extract("out_l", out);
//    print_shape(out, "output");
//    save_data(out, "output.txt");

    {
        ex.extract("pred", pred);
        print_shape(pred, "pred");
        get_ppyolo_out(pred, objects);
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    const char* param_path = argv[2];
    const char* bin_path = argv[3];
    int target_size = atoi(argv[4]);

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect(m, objects, param_path, bin_path, target_size);
    draw_objects(m, objects);

    return 0;
}
