from torchvision import transforms

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        
        new_h, new_w = int(output_size), int(output_size)

        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, sample):
        image, img_bbox, joints, = sample['image'], sample['img_bbox'], sample['joints']

        _channels, height, width = image.shape

        resized_image = transforms.functional.resize(image, (self.new_h, self.new_w))

        w_ratio = self.new_w / width
        h_ratio = self.new_h  / height

        def resize_joint(joint):
            x, y, visible = joint
            return [x * w_ratio, y * h_ratio, visible]


        resized_joints = list(map(resize_joint, joints))

        def resize_bbox(bbox):
            x0, y0, width, height = bbox
            return [x0 * w_ratio, y0 * h_ratio, width * w_ratio, height * h_ratio]

        resized_img_bbox = resize_bbox(img_bbox)
        

        return { **sample, 'image': resized_image, 'img_bbox': resized_img_bbox,'joints': resized_joints }