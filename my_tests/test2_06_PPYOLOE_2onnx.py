
import torch
from my_tests.mmdet_nets import CSPResNet, PPYOLOE, CustomCSPPAN, PPYOLOEHead

depth_mult = 0.33
width_mult = 0.50
backbone = dict(
    layers=[3, 6, 6, 3],
    channels=[64, 128, 256, 512, 1024],
    # return_idx=[3],
    return_idx=[1, 2, 3],
    use_large_stem=True,
    depth_mult=depth_mult,
    width_mult=width_mult,
)
fpn2 = dict(
    in_channels=[int(256 * width_mult), int(512 * width_mult), int(1024 * width_mult)],
    out_channels=[768, 384, 192],
    stage_num=1,
    block_num=3,
    act='swish',
    spp=True,
    depth_mult=depth_mult,
    width_mult=width_mult,
)
head2 = dict(
    in_channels=[int(768 * width_mult), int(384 * width_mult), int(192 * width_mult)],
    fpn_strides=[32, 16, 8],
    grid_cell_scale=5.0,
    grid_cell_offset=0.5,
    static_assigner_epoch=100,
    use_varifocal_loss=True,
    num_classes=80,
    loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5, },
    eval_size=(640, 640),
)
bb = CSPResNet(**backbone)
fpn = CustomCSPPAN(**fpn2)
head = PPYOLOEHead(static_assigner=None, assigner=None, nms_cfg=None, **head2)
model = PPYOLOE(bb, fpn, head)
# model = PPYOLOE(bb, fpn, None)
# model = PPYOLOE(bb, None, None)
model.eval()
state_dict = torch.load('06.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

output_name = '06.onnx'
input = "images"
output = "output"
dynamic = True
opset = 11
target_size = 640

dummy_input = torch.randn(1, 3, target_size, target_size)

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
mod.save("06.pt")

print()
