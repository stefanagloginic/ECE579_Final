class ToImage(object):
    def __init__(self, std_dev=4):
        assert isinstance(std_dev, (int))

        self.std_dev = std_dev

    def __call__(self, sample):
        image = sample['image']

        return image