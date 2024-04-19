class ToDevice(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        image, heatmaps = sample

        return image.to(self.device), heatmaps.to(self.device)