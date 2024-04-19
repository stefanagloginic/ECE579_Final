from src.guassian import get_heatmaps


class ToImageAndHeatMaps(object):
    def __call__(self, sample):
        image, heatmaps = sample['image'], sample['heatmaps']

        return image, heatmaps