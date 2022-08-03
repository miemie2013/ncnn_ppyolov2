
# ncnn

## 概述
ncnn实现PPYOLOv2和PPYOLO！

开源摘星计划（WeOpen Star） 是由腾源会 2022 年推出的全新项目，旨在为开源人提供成长激励，为开源项目提供成长支持，助力开发者更好地了解开源，更快地跨越鸿沟，参与到开源的具体贡献与实践中。

不管你是开源萌新，还是希望更深度参与开源贡献的老兵，跟随“开源摘星计划”开启你的开源之旅，从一篇学习笔记、到一段代码的提交，不断挖掘自己的潜能，最终成长为开源社区的“闪亮之星”。

我们将同你一起，探索更多的可能性！

开源摘星计划: https://github.com/weopenprojects/WeOpen-Star/blob/main/README.md

ncnn贡献指南: https://github.com/weopenprojects/WeOpen-Star/issues/27


## 快速开始

按照[miemiedetection](https://github.com/miemie2013/miemiedetection/blob/main/docs/README_PPYOLO.md#NCNN) 文档导出ncnn的PPYOLOv2和PPYOLO模型。

按照官方[how-to-build](https://github.com/Tencent/ncnn/wiki/how-to-build) 文档进行编译ncnn。

编译完成后，
将上文得到的ppyolov2_r50vd_365e.param、ppyolov2_r50vd_365e.bin、...这些文件复制到ncnn_ppyolov2的build/examples/目录下，最后在ncnn_ppyolov2根目录下运行以下命令进行ppyolov2的预测：

```
cd build/examples
./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolo_r18vd.param ppyolo_r18vd.bin 416
./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolo_r50vd_2x.param ppyolo_r50vd_2x.bin 608
./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolov2_r50vd_365e.param ppyolov2_r50vd_365e.bin 640
./test2_06_ppyolo_ncnn ../../my_tests/000000013659.jpg ppyolov2_r101vd_365e.param ppyolov2_r101vd_365e.bin 640
```

每条命令最后1个参数416、608、640表示的是将图片resize到416、608、640进行推理，即target_size参数。

test2_06_ppyolo_ncnn的源码位于ncnn_ppyolov2仓库的examples/test2_06_ppyolo_ncnn.cpp。PPYOLOv2和PPYOLO算法目前在Linux和Windows平台均已成功预测。


## 传送门

算法1群：645796480（人已满） 

算法2群：894642886 

粉丝群：704991252

关于仓库的疑问尽量在Issues上提，避免重复解答。

B站不定时女装: [_糖蜜](https://space.bilibili.com/646843384)

知乎不定时谢邀、写文章: [咩咩2013](https://www.zhihu.com/people/mie-mie-2013)

西瓜视频: [咩咩2013](https://www.ixigua.com/home/2088721227199148/?list_entrance=search)

微信：wer186259

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

AIStudio主页：[asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

欢迎在GitHub或上面的平台关注我（求粉）~

