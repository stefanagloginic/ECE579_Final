from torchvision import transforms

def is_keypoint_inside_image(image, x, y):
    _channels, height, width = image.shape

    return x >= 0 and x <= width and y >=0 and y <= height

## This transformer crops image around its bounding box and marks keypoints outside of it as unlabeled
class CropAroundBoundingBox(object):
    def __call__(self, sample):
        image, img_bbox, joints, = sample['image'], sample['img_bbox'], sample['joints']

        _channels, height, width = image.shape
        x0, y0, width, height = img_bbox

        cropped_image = transforms.functional.crop(image, top=y0, left=x0, height=height, width=width)

        def resize_joint(joint):
            x, y, visible = joint

            new_x = x - x0
            new_y = y - y0

            is_inside_bounding_box = is_keypoint_inside_image(cropped_image, new_x, new_y)

            if (not is_inside_bounding_box):
                ## Treate it as unlabeled
                return [0, 0, 0]

            return [new_x, new_y, visible]


        resized_joints = list(map(resize_joint, joints))

        def resize_bbox(bbox):
            x0, y0, width, height = bbox
            return [0, 0, width, height]

        resized_img_bbox = resize_bbox(img_bbox)
        

        return { **sample, 'image': cropped_image, 'img_bbox': resized_img_bbox,'joints': resized_joints }