
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
        # self.act = nn.Sigmoid()
        self.act = nn.LeakyReLU(0.33)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=2., mode='nearest')

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
