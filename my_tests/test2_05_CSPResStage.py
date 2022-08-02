
import torch
import cv2
import numpy as np
import ncnn_utils as ncnn_utils
from my_tests.mmdet_nets import CSPResStage2, BasicBlock, get_act_fn

act = 'swish'
trt = False
act_name = act
act = get_act_fn(act, trt=trt) if act is None or isinstance(act, (str, dict)) else act
model = CSPResStage2(BasicBlock, 2, 2, 1, 2, act=act, act_name=act_name)
model.eval()
torch.save(model.state_dict(), "05.pth")

bp = open('05_pncnn.bin', 'wb')
pp = ''
layer_id = 0
tensor_id = 0
pp += 'Input\tlayer_%.8d\t0 1 tensor_%.8d\n' % (layer_id, tensor_id)
layer_id += 1
tensor_id += 1

ncnn_data = {}
ncnn_data['bp'] = bp
ncnn_data['pp'] = pp
ncnn_data['layer_id'] = layer_id
ncnn_data['tensor_id'] = tensor_id
bottom_names = ncnn_utils.newest_bottom_names(ncnn_data)
bottom_names = model.export_ncnn(ncnn_data, bottom_names)

# 如果1个张量作为了n(n>1)个层的输入张量，应该用Split层将它复制n份，每1层用掉1个。
bottom_names = ncnn_utils.split_input_tensor(ncnn_data, bottom_names)
pp = ncnn_data['pp']
layer_id = ncnn_data['layer_id']
tensor_id = ncnn_data['tensor_id']
pp = pp.replace('tensor_%.8d' % (0,), 'images')
pp = pp.replace(bottom_names[-1], 'output')
pp = '7767517\n%d %d\n'%(layer_id, tensor_id) + pp
with open('05_pncnn.param', 'w', encoding='utf-8') as f:
    f.write(pp)
    f.close()


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


np.savez('05', **dic)
print()
