import torch
from torchvision.transforms.v2 import functional


class ToDtype(object):
    def __init__(self,  dtype = torch.float32, scale: bool = False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, sample):
        image = sample['image']

        new_dtype_image = functional.to_dtype(image, dtype=self.dtype, scale=self.scale)


        return { **sample, 'image': new_dtype_image }