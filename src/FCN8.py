import torch.nn as nn

class FCN8(nn.Module):
    # num_classes is the number of keypoints
    def __init__(self, num_classes=24):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.conv_layer_5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.conv_layer_6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.conv_layer_7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.score_7 = nn.Conv2d(4096, num_classes, 1, stride=1, padding='same')
        self.score_4 = nn.Sequential(
            nn.Conv2d(512, num_classes, 1, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.score_3 = nn.Sequential(
            nn.Conv2d(256, num_classes, 1, stride=1, padding='same'),
            nn.ReLU(),
        )
        self.upsample_score_7 = nn.ConvTranspose2d(
            num_classes, num_classes, 2, stride=2, bias=False)
        
        self.upsample_score_4 = nn.ConvTranspose2d(
            num_classes, num_classes, 2, stride=2, bias=False)
        
        self.heatmaps = nn.Sequential(
            nn.ConvTranspose2d(num_classes, num_classes, 8, stride=8),
            nn.Conv2d(num_classes, num_classes, 1, stride=1, padding='same')
        )



    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        
        pool_3 = x

        x = self.conv_layer_4(x)

        pool_4 = x

        x = self.conv_layer_5(x)
        x = self.conv_layer_6(x)
        x = self.conv_layer_7(x)

        pool_7 = x

        x = self.score_7(x)

        x = self.upsample_score_7(x)

        score_4 = self.score_4(pool_4)

        x = x + score_4

        x = self.upsample_score_4(x)

        score_3 = self.score_3(pool_3)

        x = x + score_3

        x = self.heatmaps(x)

        return x