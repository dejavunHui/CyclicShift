import torch
from torch import nn
import numpy as np
from PIL import Image

class Moving(object):

    def __init__(self, p=0.5):
        print('use moving')
        self.p = p
    def __call__(self, img):
        p = np.random.random()
        if p < self.p:
            img = self.move(img)
        return img

    def move(self, img):
        w, h = img.width, img.height
        w = min(w, h)
        rate = np.random.randint(w // 4, w // 2)
        img_data = np.asarray(img)
        img_moving = np.zeros_like(img_data)
        directions = np.random.random()
        if directions < 0.25:
            #left
            img_moving[: rate] = img_data[-rate: ]
            img_moving[rate: ] = img_data[: -rate]   
        elif directions < 0.5:
            img_moving[-rate: ] = img_data[: rate]
            img_moving[: -rate] = img_data[rate: ]           
        elif directions < 0.75:
            img_moving[:, -rate: ] = img_data[:, : rate]
            img_moving[:, : -rate] = img_data[:, rate: ]   
        else:
            img_moving[:, : rate] = img_data[:, -rate: ]
            img_moving[:, rate: ] = img_data[:, : -rate]
        return Image.fromarray(img_moving)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W//4, W // 2)
    cy = np.random.randint(H//4, H // 2)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
                                       
