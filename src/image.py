import torchvision
import torch
from src.keypoints import is_valid_keypoint
from torchvision.transforms import v2

def add_bbox(image, img_bbox):
    x0, y0, width, height = img_bbox
    bbox = [x0, y0, x0 + width, y0 + height]
    bbox = torch.tensor(bbox, dtype=torch.int)
    bbox = bbox.unsqueeze(0)
    return torchvision.utils.draw_bounding_boxes(image, bbox, width=1, colors=(255,255,0))

def add_keypoints(image, keypoints, colors, skip_visible=False):
    image_with_keypoints = image
    for index, keypoint in enumerate(keypoints):
        x, y, visible = keypoint
        keypoints_tensor = torch.tensor([[keypoint]])
        color = colors[index]

        should_skip = skip_visible and visible == 0

        if (is_valid_keypoint(keypoint) and (not should_skip)):
            image_with_keypoints = torchvision.utils.draw_keypoints(image_with_keypoints, keypoints_tensor, colors=color, radius=1)

    return image_with_keypoints


def add_bbox_and_keypoints(image, img_bbox, keypoints, colors):
    image_with_bbox = add_bbox(image, img_bbox=img_bbox)
    image_with_keypoints = add_keypoints(image_with_bbox, keypoints=keypoints, colors=colors)

    return image_with_keypoints

def add_image_transformations(image):
    curr_transform = v2.Compose([
        v2.RandomRotation(degrees=(1, 180)),
        v2.RandomHorizontalFlip(p=0.5)
    ])

    return curr_transform(image)
