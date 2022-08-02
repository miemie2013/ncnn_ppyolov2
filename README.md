
# ncnn

## 概述
ncnn实现stylegan2ada和stylegan3！

开源摘星计划（WeOpen Star） 是由腾源会 2022 年推出的全新项目，旨在为开源人提供成长激励，为开源项目提供成长支持，助力开发者更好地了解开源，更快地跨越鸿沟，参与到开源的具体贡献与实践中。

不管你是开源萌新，还是希望更深度参与开源贡献的老兵，跟随“开源摘星计划”开启你的开源之旅，从一篇学习笔记、到一段代码的提交，不断挖掘自己的潜能，最终成长为开源社区的“闪亮之星”。

我们将同你一起，探索更多的可能性！

开源摘星计划: https://github.com/weopenprojects/WeOpen-Star/blob/main/README.md

ncnn贡献指南: https://github.com/weopenprojects/WeOpen-Star/issues/27


## 快速开始

按照官方[how-to-build](https://github.com/Tencent/ncnn/wiki/how-to-build) 文档进行编译ncnn。

然后下载权重文件和随机种子文件：

链接：https://pan.baidu.com/s/1g3DCFU4riVLRil2x6Zdwjw 
提取码：a8og 

只需下载ncnn文件夹里的内容，ncnn文件夹里即为本仓库使用的权重文件、随机种子。其中seeds.zip必须下载，为随机种子文件。

将下载得到的styleganv2ada_512_afhqcat_mapping.param、styleganv2ada_512_afhqcat_mapping.bin、...seeds.zip这些文件复制到ncnn的build/examples/目录下，解压seeds.zip到当前文件夹。最后在ncnn根目录下运行以下命令进行stylegan的预测：


```
cd build/examples


./stylegan 0 512 16 1.0 seeds/458.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin 120 30



./stylegan 0 512 14 1.0 seeds/85.bin styleganv2ada_256_custom_epoch_65_mapping.param styleganv2ada_256_custom_epoch_65_mapping.bin styleganv2ada_256_custom_epoch_65_synthesis.param styleganv2ada_256_custom_epoch_65_synthesis.bin

./stylegan 1 512 14 1.0 seeds/85.bin seeds/100.bin styleganv2ada_256_custom_epoch_65_mapping.param styleganv2ada_256_custom_epoch_65_mapping.bin styleganv2ada_256_custom_epoch_65_synthesis.param styleganv2ada_256_custom_epoch_65_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 14 1.0 seeds/85.bin seeds/100.bin styleganv2ada_256_custom_epoch_65_mapping.param styleganv2ada_256_custom_epoch_65_mapping.bin styleganv2ada_256_custom_epoch_65_synthesis.param styleganv2ada_256_custom_epoch_65_synthesis.bin 120 30



./stylegan 0 512 18 1.0 seeds/458.bin styleganv2ada_1024_metfaces_mapping.param styleganv2ada_1024_metfaces_mapping.bin styleganv2ada_1024_metfaces_synthesis.param styleganv2ada_1024_metfaces_synthesis.bin

./stylegan 1 512 18 1.0 seeds/458.bin seeds/293.bin styleganv2ada_1024_metfaces_mapping.param styleganv2ada_1024_metfaces_mapping.bin styleganv2ada_1024_metfaces_synthesis.param styleganv2ada_1024_metfaces_synthesis.bin 0 1 2 3 4 5 6



./stylegan 0 512 18 1.0 seeds/458.bin styleganv2ada_1024_ffhq_mapping.param styleganv2ada_1024_ffhq_mapping.bin styleganv2ada_1024_ffhq_synthesis.param styleganv2ada_1024_ffhq_synthesis.bin

./stylegan 1 512 18 1.0 seeds/458.bin seeds/293.bin styleganv2ada_1024_ffhq_mapping.param styleganv2ada_1024_ffhq_mapping.bin styleganv2ada_1024_ffhq_synthesis.param styleganv2ada_1024_ffhq_synthesis.bin 0 1 2 3 4 5 6




./stylegan 0 512 16 1.0 seeds/458.bin styleganv3_s_256_custom_epoch_77_mapping.param styleganv3_s_256_custom_epoch_77_mapping.bin styleganv3_s_256_custom_epoch_77_synthesis.param styleganv3_s_256_custom_epoch_77_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin styleganv3_s_256_custom_epoch_77_mapping.param styleganv3_s_256_custom_epoch_77_mapping.bin styleganv3_s_256_custom_epoch_77_synthesis.param styleganv3_s_256_custom_epoch_77_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin styleganv3_s_256_custom_epoch_77_mapping.param styleganv3_s_256_custom_epoch_77_mapping.bin styleganv3_s_256_custom_epoch_77_synthesis.param styleganv3_s_256_custom_epoch_77_synthesis.bin 120 30


./stylegan 0 512 16 1.0 seeds/458.bin stylegan3_r_afhqv2_512_mapping.param stylegan3_r_afhqv2_512_mapping.bin stylegan3_r_afhqv2_512_synthesis.param stylegan3_r_afhqv2_512_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_r_afhqv2_512_mapping.param stylegan3_r_afhqv2_512_mapping.bin stylegan3_r_afhqv2_512_synthesis.param stylegan3_r_afhqv2_512_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_r_afhqv2_512_mapping.param stylegan3_r_afhqv2_512_mapping.bin stylegan3_r_afhqv2_512_synthesis.param stylegan3_r_afhqv2_512_synthesis.bin 120 30


./stylegan 0 512 16 1.0 seeds/458.bin stylegan3_r_ffhq_1024_mapping.param stylegan3_r_ffhq_1024_mapping.bin stylegan3_r_ffhq_1024_synthesis.param stylegan3_r_ffhq_1024_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_r_ffhq_1024_mapping.param stylegan3_r_ffhq_1024_mapping.bin stylegan3_r_ffhq_1024_synthesis.param stylegan3_r_ffhq_1024_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_r_ffhq_1024_mapping.param stylegan3_r_ffhq_1024_mapping.bin stylegan3_r_ffhq_1024_synthesis.param stylegan3_r_ffhq_1024_synthesis.bin 120 30


./stylegan 0 512 16 1.0 seeds/458.bin stylegan3_t_afhqv2_512_mapping.param stylegan3_t_afhqv2_512_mapping.bin stylegan3_t_afhqv2_512_synthesis.param stylegan3_t_afhqv2_512_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_t_afhqv2_512_mapping.param stylegan3_t_afhqv2_512_mapping.bin stylegan3_t_afhqv2_512_synthesis.param stylegan3_t_afhqv2_512_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_t_afhqv2_512_mapping.param stylegan3_t_afhqv2_512_mapping.bin stylegan3_t_afhqv2_512_synthesis.param stylegan3_t_afhqv2_512_synthesis.bin 120 30



./stylegan 0 512 16 1.0 seeds/458.bin stylegan3_t_ffhq_1024_mapping.param stylegan3_t_ffhq_1024_mapping.bin stylegan3_t_ffhq_1024_synthesis.param stylegan3_t_ffhq_1024_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_t_ffhq_1024_mapping.param stylegan3_t_ffhq_1024_mapping.bin stylegan3_t_ffhq_1024_synthesis.param stylegan3_t_ffhq_1024_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_t_ffhq_1024_mapping.param stylegan3_t_ffhq_1024_mapping.bin stylegan3_t_ffhq_1024_synthesis.param stylegan3_t_ffhq_1024_synthesis.bin 120 30



./stylegan 0 512 16 1.0 seeds/458.bin stylegan3_t_metfaces_1024_mapping.param stylegan3_t_metfaces_1024_mapping.bin stylegan3_t_metfaces_1024_synthesis.param stylegan3_t_metfaces_1024_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_t_metfaces_1024_mapping.param stylegan3_t_metfaces_1024_mapping.bin stylegan3_t_metfaces_1024_synthesis.param stylegan3_t_metfaces_1024_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin stylegan3_t_metfaces_1024_mapping.param stylegan3_t_metfaces_1024_mapping.bin stylegan3_t_metfaces_1024_synthesis.param stylegan3_t_metfaces_1024_synthesis.bin 120 30
```

我解释一下这3条命令，其余命令同理

```
./stylegan 0 512 16 1.0 seeds/458.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin 120 30
```
第2个参数可以是0、1、2，0表示图片生成、1表示style_mixing、2表示图像渐变A2B。第3个参数512表示exp配置文件的z_dim，一般是512。第4个参数16表示模型的num_ws，导出时会打印num_ws。第5个参数1.0表示上文的--trunc，设为0时就能请出w_avg妈妈了！ seeds/458.bin表示随机种子文件路径，玩style_mixing和A2B时需要填2个随机种子文件路径。然后是mapping网络的param和bin文件路径、synthesis网络的param和bin文件路径。如果你玩style_mixing，往后可以输入" 0 1 2 3 4 5 6"，表示seeds/458.bin的ws（理论上MappingNetwork里repeat后的结果）的0 1 2 3 4 5 6个被替换成了seeds/293.bin的0 1 2 3 4 5 6个，因此生成的假照片里的猫咪有了seeds/293.bin的动作姿态，却有着seeds/458.bin的皮肤。如果你玩A2B，往后可以输入" 120 30"，表示用120帧实现渐变，生成的视频帧数是30。

stylegan3_r的模型推理非常慢，因为其channel_base和channel_max都比stylegan3_t的模型大。尽量使用stylegan3_t的模型获得更好的体验。

ncnn的stylegan模型由[miemieGAN](https://github.com/miemie2013/miemieGAN) 导出，技术细节请参考本人的知乎文章[全网第一个！stylegan2-ada和stylegan3的pytorch实现二合一！支持导出ncnn！](https://zhuanlan.zhihu.com/p/539140181)


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

