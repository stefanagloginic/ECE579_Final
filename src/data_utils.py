import torch
import math
from collections.abc import Iterable
from src.keypoints import is_valid_keypoint
# from pycocotools.mask import decode as decode_RLE

def is_finite_number(number):
    return (not isinstance(number, bool)) and isinstance(number, (int, float)) and math.isfinite(number)

# This function return true if a keypoint is not a valid number aka Null or None or NaN
def is_bad_keypoint(keypoint):
    for coord in keypoint:
        if (not is_finite_number(coord)):
            return True
    
    return False

def is_joint_outside_image(joint, image):
    _channels, height, width = image.shape
    x, y, _visible = joint

    return x < 0 or x > width or y < 0 or y > height

def has_bad_joints(joints, image):
    if not isinstance(joints, Iterable):
        return True
    
    unlabeled_keypoint = 0
    
    for joint in joints:
        if (is_bad_keypoint(joint) or is_joint_outside_image(joint, image)):
            return True
        
        if (not is_valid_keypoint(joint)):
            unlabeled_keypoint = unlabeled_keypoint + 1
        
    all_joints_unlabled = unlabeled_keypoint == len(joints)

    if (all_joints_unlabled):
        return True

    return False

def has_bad_image(image):
    return not torch.is_tensor(image)

def does_sample_have_multliple_dogs(sample):
    if (sample['is_multiple_dogs']):
     return sample['is_multiple_dogs']

def is_bad_training_sample(sample):
    image, joints = sample['image'], sample['joints']
    
    return has_bad_image(image) or has_bad_joints(joints, image)

def is_bad_evaluation_sample(sample):
    return is_bad_training_sample(sample) or does_sample_have_multliple_dogs(sample)


# def get_seg_from_entry(sample):
#     img_height, img_width, seg = sample['img_height'], sample['img_width'], sample['seg']

#     rle = {
# 		"size": [img_height, img_width],
# 		"counts": seg
# 	}

#     decoded = decode_RLE(rle)
    
#     return decoded