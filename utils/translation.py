import torch
import numpy as np


class Translation(object):

    def __init__(self, p):
        self.p = p
        
    def trans(self, img):
        w = img.size(2)
        shiftw = np.random.randint(w//6, w//3)
        r = np.random.random()
        if r < 0.25:
            img[:, :-shiftw] = img[:, shiftw:]
            img[:, -shiftw:] = 0
        elif r < 0.5:
            img[:, shiftw:] = img[:, :-shiftw]
            img[:, :shiftw] = 0
        elif r < 0.75:
            img[:, :, :-shiftw] = img[:, :, shiftw:]
            img[:, :, -shiftw:] = 0
        else:
            img[:, :, shiftw:] = img[:, :, :-shiftw]
            img[:, :, :shiftw] = 0
        return img

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if self.p > np.random.random():
            img = self.trans(img)
        return img
