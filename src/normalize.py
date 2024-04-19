from torchvision import transforms

TRAIN_DATASET_MEAN = [0.4831, 0.4644, 0.3996]
TRAIN_DATA_SET_STD = [0.2530, 0.2472, 0.2546]

CROPPED_TRAIN_DATASET_MEAN = [0.4843, 0.4510, 0.3914]
CROPPED_TRAIN_DATASET_STD = [0.2508, 0.2434, 0.2458]

class Normalize(object):
    def __init__(self, mean, std):

        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image = sample['image']

        normalized_image = transforms.functional.normalize(image, mean=self.mean, std=self.std)

        return { **sample, 'image': normalized_image }
    

