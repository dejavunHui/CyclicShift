
import torch
import numpy as np


class Shuffle(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, p):
        self.p = p
        
    def trans(self, img):
        new_img = torch.zeros_like(img)
        patches = []
        patch_w = img.size(2) // 4
        for x in np.split(img, 4, 1):
            patches.extend(np.split(x, 4, 2))

        np.random.shuffle(patches)
        for i, patch in enumerate(patches):
            # print(patch.shape)
            x = i // 4
            y = i % 4
            new_img[:, x*patch_w:x*patch_w+patch_w, y*patch_w:y*patch_w+patch_w] = patch
        return new_img

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
