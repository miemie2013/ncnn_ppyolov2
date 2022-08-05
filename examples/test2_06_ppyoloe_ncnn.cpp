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


#define NMS_THRESH  0.6  // nms threshold
#define CONF_THRESH 0.15 // threshold of bounding box prob
#define TARGET_SIZE 640  // target image size


struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

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

static void generate_ppyoloe_proposals(const ncnn::Mat& cls_score, const ncnn::Mat& reg_dist, float scale_x, float scale_y, float prob_threshold, std::vector<Object>& objects)
{
    // python中cls_score的形状是[N, A, 80], ncnn中C=1, H=A=预测框数, W=80
    // python中reg_dist 的形状是[N, A,  4], ncnn中C=1, H=A=预测框数, W= 4
    int C = cls_score.c;
    int H = cls_score.h;
    int W = cls_score.w;
//    printf("C=%d\n", C);
//    printf("H=%d\n", H);
//    printf("W=%d\n", W);
    int num_grid = H;
    int num_class = W;

    // 最大感受野输出的特征图一行（一列）的格子数stride32_grid。设为G，则
    // G*G + (2*G)*(2*G) + (4*G)*(4*G) = 21*G^2 = W
    // 所以G = sqrt(W/21)
    int stride32_grid = sqrt(num_grid / 21);
    int stride16_grid = stride32_grid * 2;
    int stride8_grid = stride32_grid * 4;

    // 因为二者的C都只等于1，所以取第0个
    const float* cls_score_ptr = cls_score.channel(0);
    const float* reg_dist_ptr = reg_dist.channel(0);

    // stride==32的格子结束的位置
    int stride32_end = stride32_grid * stride32_grid;
    // stride==16的格子结束的位置
    int stride16_end = stride32_grid * stride32_grid * 5;
    for (int anchor_idx = 0; anchor_idx < num_grid; anchor_idx++)
    {
        float stride = 32.0f;
        int row_i = 0;
        int col_i = 0;
        if (anchor_idx < stride32_end) {
            stride = 32.0f;
            row_i = anchor_idx / stride32_grid;
            col_i = anchor_idx % stride32_grid;
        }else if (anchor_idx < stride16_end) {
            stride = 16.0f;
            row_i = (anchor_idx - stride32_end) / stride16_grid;
            col_i = (anchor_idx - stride32_end) % stride16_grid;
        }else {  // stride == 8
            stride = 8.0f;
            row_i = (anchor_idx - stride16_end) / stride8_grid;
            col_i = (anchor_idx - stride16_end) % stride8_grid;
        }
        float x_center = 0.5f + (float)col_i;
        float y_center = 0.5f + (float)row_i;
        float x0 = x_center - reg_dist_ptr[0];
        float y0 = y_center - reg_dist_ptr[1];
        float x1 = x_center + reg_dist_ptr[2];
        float y1 = y_center + reg_dist_ptr[3];
        x0 = x0 * stride / scale_x;
        y0 = y0 * stride / scale_y;
        x1 = x1 * stride / scale_x;
        y1 = y1 * stride / scale_y;
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
        cls_score_ptr += cls_score.w;
        reg_dist_ptr += reg_dist.w;
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

static int detect_PPYOLOE(const cv::Mat& bgr, std::vector<Object>& objects, const char* param_path, const char* bin_path)
{
    ncnn::Net model;

    model.opt.use_vulkan_compute = true;

    model.load_param(param_path);
    model.load_model(bin_path);

    int img_w = bgr.cols;
    int img_h = bgr.rows;
    float scale_x = (float)TARGET_SIZE / img_w;
    float scale_y = (float)TARGET_SIZE / img_h;

    // get ncnn::Mat with RGB format like PPYOLOE do.
    ncnn::Mat in_rgb = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows);
    ncnn::Mat in_resize;
    // Interp image with cv2.INTER_CUBIC like PPYOLOE do.
    ncnn::resize_bicubic(in_rgb, in_resize, TARGET_SIZE, TARGET_SIZE);

    // Normalize image with the same mean and std like PPYOLOE do.
//    mean=[123.675, 116.28, 103.53]
//    std=[58.395, 57.12, 57.375]
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float norm_vals[3] = {1.0f/58.395f, 1.0f/57.12f, 1.0f/57.375f};
    in_resize.substract_mean_normalize(mean_vals, norm_vals);

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

    ncnn::Mat cls_score;  // python中的形状是[N, A, 80], ncnn中C=1, H=A=预测框数, W=80
    ncnn::Mat reg_dist;   // python中的形状是[N, A,  4], ncnn中C=1, H=A=预测框数, W= 4

    // warm up
    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        ncnn::Extractor ex = model.create_extractor();
        ex.input("images", in_resize);
        ex.extract("cls_score", cls_score);
        ex.extract("reg_dist", reg_dist);
        std::vector<Object> temp_proposals;
        generate_ppyoloe_proposals(cls_score, reg_dist, scale_x, scale_y, CONF_THRESH, temp_proposals);
        qsort_descent_inplace(temp_proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(temp_proposals, picked, NMS_THRESH);
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
            ex.extract("cls_score", cls_score);
            ex.extract("reg_dist", reg_dist);
            std::vector<Object> temp_proposals;
            generate_ppyoloe_proposals(cls_score, reg_dist, scale_x, scale_y, CONF_THRESH, temp_proposals);
            qsort_descent_inplace(temp_proposals);
            std::vector<int> picked;
            nms_sorted_bboxes(temp_proposals, picked, NMS_THRESH);
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

    // Debug
//    ncnn::Mat out;
//    ex.extract("output", out);
//    ex.extract("out0", out);
//    print_shape(out, "output");
//    save_data(out, "output.txt");

    std::vector<Object> proposals;

    {
        ex.extract("cls_score", cls_score);
        ex.extract("reg_dist", reg_dist);
        generate_ppyoloe_proposals(cls_score, reg_dist, scale_x, scale_y, CONF_THRESH, proposals);
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        //
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
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

    std::vector<Object> objects;
    detect_PPYOLOE(m, objects, param_path, bin_path);
    draw_objects(m, objects);

    return 0;
}
