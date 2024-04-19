from torchvision import transforms


class ToTensor(object):
    def __init__(self):
        return

    def __call__(self, sample):
        image = sample['image']

        tensor_image = transforms.functional.to_tensor(image)

        print('tensor_image', tensor_image)

        return { **sample, 'image': tensor_image }