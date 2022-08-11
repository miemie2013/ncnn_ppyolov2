
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import ncnn_utils as ncnn_utils



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        mid_c = 2
        out_c = 2

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=mid_c,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=1,
            bias=True)

        self.bn = nn.BatchNorm2d(mid_c)
        # self.act = nn.Sigmoid()
        self.act = nn.LeakyReLU(0.33)

        self.conv2 = nn.Conv2d(
            in_channels=mid_c,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False)
        torch.nn.init.normal_(self.conv.weight)
        torch.nn.init.normal_(self.conv.bias)
        torch.nn.init.normal_(self.conv2.weight)
        # torch.nn.init.constant_(self.conv2.weight, 5.25)  # 0x40a80000 when fp32, 0x4540 when fp16
        torch.nn.init.normal_(self.bn.weight)
        torch.nn.init.normal_(self.bn.bias)
        torch.nn.init.normal_(self.bn.running_mean)
        torch.nn.init.constant_(self.bn.running_var, 2.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv2(x)

        return x

    def export_ncnn(self, ncnn_data, bottom_names):
        act_name = 'leaky_relu'
        act_param_dict = {'negative_slope' : 0.33}
        if ncnn_utils.support_fused_activation(act_name):
            bottom_names = ncnn_utils.fuse_conv_bn(ncnn_data, bottom_names, self.conv, self.bn, act_name, act_param_dict)
        else:
            bottom_names = ncnn_utils.fuse_conv_bn(ncnn_data, bottom_names, self.conv, self.bn)
            bottom_names = ncnn_utils.activation(ncnn_data, bottom_names, act_name)
        bottom_names = ncnn_utils.conv2d(ncnn_data, bottom_names, self.conv2)
        return bottom_names


fp16 = True
# fp16 = False

model = MyNet()
model.eval()
torch.save(model.state_dict(), "01.pth")


dic = {}


aaaaaaaaa = cv2.imread('my_test.jpg')
aaaaaaaaa = aaaaaaaaa.astype(np.float32)

mean = [117.3, 126.5, 130.2]
std = [108.4, 117.3, 127.6]
mean = np.array(mean)[np.newaxis, np.newaxis, :]
std = np.array(std)[np.newaxis, np.newaxis, :]
aaaaaaaaa -= mean
aaaaaaaaa /= std


x = torch.from_numpy(aaaaaaaaa)
x = x.to(torch.float32)
x = x.permute((2, 0, 1))
x = torch.unsqueeze(x, 0)
x.requires_grad_(False)
y = model(x)


dic['x'] = x.cpu().detach().numpy()
dic['y'] = y.cpu().detach().numpy()


np.savez('01', **dic)
print()

ncnn_output_path = '01_pncnn'
if fp16:
    ncnn_utils.set_convert_to_fp16(True)
ncnn_data, bottom_names = ncnn_utils.create_new_param_bin(ncnn_output_path, input_num=1)
bottom_names = model.export_ncnn(ncnn_data, bottom_names)
# 如果1个张量作为了n(n>1)个层的输入张量，应该用Split层将它复制n份，每1层用掉1个。
bottom_names = ncnn_utils.split_input_tensor(ncnn_data, bottom_names)
ncnn_utils.save_param(ncnn_output_path, ncnn_data, bottom_names,
                      replace_input_names=['images', ],
                      replace_output_names=['output', ])


