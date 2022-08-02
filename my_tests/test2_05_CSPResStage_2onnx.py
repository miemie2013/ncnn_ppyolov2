
import torch
from my_tests.mmdet_nets import CSPResStage2, BasicBlock, get_act_fn

act = 'swish'
trt = False
act_name = act
act = get_act_fn(act, trt=trt) if act is None or isinstance(act, (str, dict)) else act
model = CSPResStage2(BasicBlock, 2, 2, 1, 2, act=act, act_name=act_name)
model.eval()
state_dict = torch.load('05.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

output_name = '05.onnx'
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

print()
