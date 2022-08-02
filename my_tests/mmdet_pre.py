
import cv2
import numpy as np






class PPYOLOEValTransform:
    def __init__(self, context, to_rgb, resizeImage, normalizeImage, permute):
        self.context = context
        self.to_rgb = to_rgb
        self.resizeImage = resizeImage
        self.normalizeImage = normalizeImage
        self.permute = permute

    def __call__(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.resizeImage(sample, context)
        sample = self.normalizeImage(sample, context)
        sample = self.permute(sample, context)

        pimage = np.expand_dims(sample['image'], axis=0)
        scale_factor = np.array([[sample['scale_factor'][1], sample['scale_factor'][0]]]).astype(np.float32)
        return pimage, scale_factor





