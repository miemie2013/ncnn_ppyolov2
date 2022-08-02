
import torch
from my_tests.mmdet_nets import RepVggBlock


model = RepVggBlock(3, 2, act='hardsigmoid', act_name='hardsigmoid')
model.eval()
state_dict = torch.load('03.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

output_name = '03.onnx'
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
mod.save("03.pt")

print()
