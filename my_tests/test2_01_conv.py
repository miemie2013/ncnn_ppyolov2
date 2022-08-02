
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=1,
            bias=True)

        self.bn = nn.BatchNorm2d(2)
        torch.nn.init.normal_(self.conv.weight)
        torch.nn.init.normal_(self.conv.bias)
        torch.nn.init.normal_(self.bn.weight)
        torch.nn.init.normal_(self.bn.bias)
        torch.nn.init.normal_(self.bn.running_mean)
        torch.nn.init.constant_(self.bn.running_var, 2.3)
        # self.act = nn.Sigmoid()
        self.act = nn.LeakyReLU(0.33)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2., mode='nearest')

        return x


class MyNetFuse(nn.Module):
    def __init__(self):
        super(MyNetFuse, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0,
            groups=1,
            bias=True)
        # self.act = nn.Sigmoid()
        self.act = nn.LeakyReLU(0.33)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2., mode='nearest')

        return x



model = MyNet()
model.eval()
torch.save(model.state_dict(), "01.pth")


model2 = MyNetFuse()
model2.eval()
std = model.state_dict()
std2 = model2.state_dict()

conv_w = model.conv.weight.cpu().detach().numpy()
conv_b = model.conv.bias.cpu().detach().numpy()

bn_w = model.bn.weight.cpu().detach().numpy()
bn_b = model.bn.bias.cpu().detach().numpy()
bn_m = model.bn.running_mean.cpu().detach().numpy()
bn_v = model.bn.running_var.cpu().detach().numpy()
eps = model.bn.eps
'''
合并卷积层和BN层。推导：
y = [(conv_w * x + conv_b) - bn_m] / sqrt(bn_v + eps) * bn_w + bn_b
= [conv_w * x + (conv_b - bn_m)] / sqrt(bn_v + eps) * bn_w + bn_b
= conv_w * x / sqrt(bn_v + eps) * bn_w + (conv_b - bn_m) / sqrt(bn_v + eps) * bn_w + bn_b
= conv_w * bn_w / sqrt(bn_v + eps) * x + (conv_b - bn_m) / sqrt(bn_v + eps) * bn_w + bn_b

所以
new_conv_w = conv_w * bn_w / sqrt(bn_v + eps)
new_conv_b = (conv_b - bn_m) / sqrt(bn_v + eps) * bn_w + bn_b

'''

new_conv_w = conv_w * (bn_w / np.sqrt(bn_v + eps)).reshape((-1, 1, 1, 1))
new_conv_b = (conv_b - bn_m) / np.sqrt(bn_v + eps) * bn_w + bn_b

std2['conv.weight'].copy_(torch.Tensor(new_conv_w))
std2['conv.bias'].copy_(torch.Tensor(new_conv_b))
model2.load_state_dict(std2)


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
y2 = model2(x)


dic['x'] = x.cpu().detach().numpy()
dic['y'] = y.cpu().detach().numpy()


np.savez('01', **dic)
print()
