
import numpy as np


dic2 = np.load('30.npz')
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

ddd = np.sum((y - y2) ** 2)
print('ddd=%.9f' % ddd)


print()

