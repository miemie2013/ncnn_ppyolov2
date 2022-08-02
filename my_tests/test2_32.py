
import numpy as np


dic2 = np.load('32.npz')
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

# y = y[0][259:261, 376:378]
# y2 = y2[0][259:261, 376:378]

ddd = np.sum((y - y2) ** 2)
print('sum  ddd=%.9f' % ddd)

ddd2 = np.mean((y - y2) ** 2)
print('mean ddd=%.9f' % ddd2)


print()

