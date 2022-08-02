
import onnxruntime
import cv2
import numpy as np
from my_tests.mmdet_trans import ResizeImage, NormalizeImage, Permute
from my_tests.mmdet_pre import PPYOLOEValTransform


model = '09.onnx'
input = "images"
output = "output"
dynamic = True
opset = 11




# 预测时的数据预处理
context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
to_rgb = True
target_size = 6

# NormalizeImage
normalizeImage2 = dict(
    # mean=[0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225],
    # is_scale=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    is_scale=False,
    is_channel_first=False,
)
# Permute
permute2 = dict(
    to_bgr=False,
    channel_first=True,
)
# ResizeImage
resizeImage2 = dict(
    target_size=640,
    interp=2,
)
print(cv2.INTER_LINEAR)
print(cv2.INTER_CUBIC)
print(cv2.INTER_NEAREST)

resizeImage = ResizeImage(target_size=target_size, interp=resizeImage2['interp'])
normalizeImage = NormalizeImage(**normalizeImage2)
permute = Permute(**permute2)
preproc = PPYOLOEValTransform(context, to_rgb, resizeImage, normalizeImage, permute)

img = cv2.imread('my_test.jpg')

img_info = {"id": 0}
height, width = img.shape[:2]
img_info["height"] = height
img_info["width"] = width
img_info["raw_img"] = img

img, scale_factor = preproc(img)

session = onnxruntime.InferenceSession(model)
ort_inputs = {session.get_inputs()[0].name: img}
output = session.run(None, ort_inputs)
y = output[0]




dic2 = np.load('09.npz')
y2 = dic2['y']

ddd = np.sum((y - y2) ** 2)
print('ddd=%.6f' % ddd)



print()
