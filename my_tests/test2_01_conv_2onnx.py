
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


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


model = MyNet()
model.eval()
state_dict = torch.load('01.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

output_name = '01.onnx'
input = "images"
output = "output"
dynamic = True
opset = 11

dummy_input = torch.randn(1, 3, 4, 4)

torch.onnx._export(
    model,
    dummy_input,
    output_name,
    input_names=[input],
    output_names=[output],
    dynamic_axes={input: {0: 'batch'},
                  output: {0: 'batch'}} if dynamic else None,
    opset_version=opset,
)
mod = torch.jit.trace(model, dummy_input)
mod.save("01.pt")

print()
