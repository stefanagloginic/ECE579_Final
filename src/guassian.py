import torch
from src.keypoints import is_valid_keypoint
import numpy as np

def gaussian_heatmap(height, width, center, std_dev=4):
    x_axis = torch.arange(width).float() - center[0]
    y_axis = torch.arange(height).float() - center[1]
    x, y = torch.meshgrid(y_axis, x_axis, indexing='ij')
    
    return  torch.exp(-((x ** 2 + y ** 2) / (2 * std_dev ** 2)))

def empty_gaussian_heatmap(height, width):
    return torch.zeros(height, width)


def get_heatmaps(image, keypoints, std_dev=4):
    _channels, height, width = image.shape

    heatmaps = []

    for keypoint in keypoints:
        if (is_valid_keypoint(keypoint)):
            heatmaps.append(gaussian_heatmap(height, width, keypoint, std_dev=std_dev))
        else:
            heatmaps.append(empty_gaussian_heatmap(height, width))

    return torch.stack(heatmaps)

def CoM_activation(heatmap):
    height, width = heatmap.shape
    hm_np = heatmap.numpy()
    hm_sum = np.sum(hm_np)

    if (hm_sum == 0):
        # If there is no activation then 
        # we can safely return (0, 0)
        return (0, 0)

    index_map = [j for i in range(height) for j in range(width)]
    index_map = np.reshape(index_map, newshape=(height, width))

    x_score_map = hm_np * index_map / hm_sum
    y_score_map = hm_np * np.transpose(index_map) / hm_sum

    px = np.sum(np.sum(x_score_map, axis=None))
    py = np.sum(np.sum(y_score_map, axis=None))

    return (px, py)

def max_activation(heatmap):
    height, width = heatmap.shape

    max_coords = (0, 0)
    max_active = float('-inf')
    for y in range(height):
        for x in range(width):
            activation = heatmap[y][x]

            if (activation > max_active):
                max_active = activation
                max_coords = (x, y)

    return max_coords

def heatmaps_to_keypoints(heatmaps, keypoint_func):
    channel, _height, _width = heatmaps.shape

    keypoints = []
    for c in range(channel):
        heatmap = heatmaps[c]
        keypoints.append(keypoint_func(heatmap))

    return keypoints
    

def heatmaps_to_keypoints_max_activation(heatmaps):
    return heatmaps_to_keypoints(heatmaps, max_activation)

def heatmaps_to_keypoints_CoM(heatmaps):
    return heatmaps_to_keypoints(heatmaps, CoM_activation)



