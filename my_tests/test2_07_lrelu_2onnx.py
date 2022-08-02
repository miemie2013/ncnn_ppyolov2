
import torch
from my_tests.mmdet_nets import LRELU

model = LRELU()
model.eval()
state_dict = torch.load('07.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

output_name = '07.onnx'
input = "images"
output = "output"
dynamic = True
opset = 11

dummy_input = torch.randn(1, 3, 6, 6)

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

print()
