
import numpy as np


dic2 = np.load('31.npz')
y2 = dic2['y']

ncnn_output = '../build/examples/output.txt'

with open(ncnn_output, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
line = line[:-1]
ss = line.split(',')
y = []
for s in ss:
    y.append(float(s))
y = np.array(y).astype(np.float32)
y = np.reshape(y, y2.shape)
print(y2.shape)

# y = y[:, :, 44:46, 17:19]
# y2 = y2[:, :, 44:46, 17:19]

ddd = np.sum((y - y2) ** 2)
print('sum  ddd=%.9f' % ddd)

ddd2 = np.mean((y - y2) ** 2)
print('mean ddd=%.9f' % ddd2)

import torch
import cv2



img = torch.from_numpy(y)
img = img.permute((0, 2, 3, 1)) * 127.5 + 128
img = img.clamp(0, 255)
img = img.to(torch.uint8)
img_rgb = img.cpu().detach().numpy()[0]
img_bgr = img_rgb[:, :, [2, 1, 0]]
cv2.imwrite('1111.jpg', img_bgr)



img = torch.from_numpy(y2)
img = img.permute((0, 2, 3, 1)) * 127.5 + 128
img = img.clamp(0, 255)
img = img.to(torch.uint8)
img_rgb = img.cpu().detach().numpy()[0]
img_bgr = img_rgb[:, :, [2, 1, 0]]
cv2.imwrite('1112.jpg', img_bgr)


print()

