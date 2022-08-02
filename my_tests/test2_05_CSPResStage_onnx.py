
import onnxruntime
import cv2
import numpy as np



model = '05.onnx'
input = "images"
output = "output"
dynamic = True
opset = 11




def preproc(img, swap=(2, 0, 1)):
    img = img.astype(np.float32)

    mean = [117.3, 126.5, 130.2]
    std = [108.4, 117.3, 127.6]
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    img -= mean
    img /= std


    img = img.transpose(swap)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img


origin_img = cv2.imread('my_test.jpg')
img = preproc(origin_img)

session = onnxruntime.InferenceSession(model)
ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
output = session.run(None, ort_inputs)
y = output[0]




dic2 = np.load('05.npz')
y2 = dic2['y']

ddd = np.sum((y - y2) ** 2)
print('ddd=%.6f' % ddd)



print()
