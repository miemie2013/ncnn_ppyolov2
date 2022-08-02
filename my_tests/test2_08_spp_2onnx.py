
import torch
from my_tests.mmdet_nets import SPP, get_act_fn

ch_mid = 3
act = 'swish'
trt = False
act_name = act
act = get_act_fn(
    act, trt=trt) if act is None or isinstance(act,
                                               (str, dict)) else act
model = SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act, act_name=act_name)
model.eval()
state_dict = torch.load('08.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

output_name = '08.onnx'
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
