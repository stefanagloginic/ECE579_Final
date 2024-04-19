from src.guassian import get_heatmaps


class HeatMaps(object):
    def __init__(self, std_dev=4):
        assert isinstance(std_dev, (int))

        self.std_dev = std_dev

    def __call__(self, sample):
        image, joints = sample['image'], sample['joints']

        heatmaps = get_heatmaps(image, joints, std_dev = self.std_dev)

        return { **sample, 'heatmaps': heatmaps }