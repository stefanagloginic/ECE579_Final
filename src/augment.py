import torchvision
import cv2 as cv
import numpy as np
import random
import torch

class Augment(object):
    def __init__(self, transform_dict = None):
        self.transform_dict = transform_dict

    # check if transformed point is located within image boundaries
    def _checkBoundaries(self, p):
        # x dimension
        if p[0] < 0:
            px = 0
        else:
            px = p[0]

        # y dimension
        if p[1] < 0:
            py = 0
        else:
            py = p[1]

        return (int(px), int(py), px and py)

    # Returns a 2D rotation matrix.
    # Args:
    #     angle: The angle of rotation in radians.
    # Returns:
    #     A 2D rotation matrix.
    def get_rotation_matrix2d(self, angle):
        angleAsTensor = torch.tensor(angle)
        cos = torch.cos(angleAsTensor)
        sin = torch.sin(angleAsTensor)

        return torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]])


    def __call__(self, sample):
        img, img_bbox, keypoints, = sample['image'], sample['img_bbox'], sample['joints']
        aug_keypoints = []
        _, cy, cx = img.shape
        cy //= 2
        cx //= 2

        if self.transform_dict['Flip']:
            flip = random.choice([True, False])
            if flip:
                img = torchvision.transforms.functional.hflip(img)
        if self.transform_dict['Rotate']:
            r = random.randint(-180,180)
            img = torch.permute(img, (1, 2, 0))
            img = np.array(img).astype(np.float32)
            M_rot = cv.getRotationMatrix2D(center=(img.shape[0] // 2, img.shape[1] // 2), angle=r, scale=1.0)
            img = cv.warpAffine(img, M_rot, (img.shape[0], img.shape[1]), borderMode=cv.BORDER_CONSTANT, borderValue=1)
            img = torch.tensor(img)
            img = torch.from_numpy(np.array(img).astype(np.uint8))
            img = torch.permute(img,(2, 0, 1))
                    
        # transform keypoints
        for i in range(len(keypoints)):
            px, py, visible = keypoints[i]
            p = np.array([px, py, visible])

            if (px or py):
                # apply flip
                if self.transform_dict['Flip'] and flip:
                    p[0] = cx + (cx - p[0])

                # apply rotation
                if self.transform_dict['Rotate']:
                    p = np.dot(M_rot, p)

            p = self._checkBoundaries(p)
            aug_keypoints.append(p)

        return { **sample, 'image': img, 'joints': aug_keypoints}