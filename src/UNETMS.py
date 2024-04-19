import torch
import torch.nn as nn
class Concatlayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim=dim

    def forward(self,x,y):
        return torch.cat([x,y],dim=self.dim)
    
class UNETMS(nn.Module):
    # num_classes is the number of keypoints
    def __init__(self, num_classes=24):
        super().__init__()
        self.num_classes=num_classes
        in_hight, in_width=6,6
        in_hight = (2 * (in_hight - 1) + 2 - in_hight) // 2
        in_width  = (2 * (in_width - 1) + 2 - in_width) // 2
                  
        # downsampling layers
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),#
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )
        self.conv_layer_1_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), padding=0, stride=2, ceil_mode=True)
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )
        self.conv_layer_2_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), padding=0, stride=2, ceil_mode=True)
        )

        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )
        self.conv_layer_3_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), padding=0, stride=2, ceil_mode=True)
        )

        self.conv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )
        self.conv_layer_4_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), padding=0, stride=2, ceil_mode=True)
        )
        self.conv_layer_5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )

        # upsampling layers
        self.conv_layer_6_transposed = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2, 2), stride=2)
        self.cat_6 = Concatlayer()
        self.conv_layer_6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )

        in_hight, in_width=1,1
        in_hight = (2 * (in_hight - 1) + 2 - in_hight) // 2
        in_width  = (2 * (in_width - 1) + 2 - in_width) // 2

        self.conv_layer_7_transposed = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2, 2), stride=2)
        self.cat_7 = Concatlayer()
        self.conv_layer_7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )

        self.conv_layer_8_transposed = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 2), stride=2)
        self.cat_8 = Concatlayer()
        self.conv_layer_8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )

        self.conv_layer_9_transposed = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=2)
        self.cat_9 = Concatlayer()
        self.conv_layer_9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )

        self.cat_image_heatmap = Concatlayer()

        self.stage_2_conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=27, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),#
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU()
        )

        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=(1, 1), stride=1, padding='valid')
    
    def forward(self, x):
        # downsampling layers
        x1 = self.conv_layer_1(x)
        # print('x1.shape: ',x1.shape)
        x1_p = self.conv_layer_1_pool(x1)
        # print('x1_p.shape: ',x1_p.shape)
        x2 = self.conv_layer_2(x1_p)
        # print('x2.shape: ',x2.shape)
        x2_p = self.conv_layer_2_pool(x2)
        # print('x2_p.shape: ',x2_p.shape)
        x3 = self.conv_layer_3(x2_p)
        # print('x3.shape: ',x3.shape)
        x3_p = self.conv_layer_3_pool(x3)
        # print('x3_p.shape: ',x3_p.shape)
        x4 = self.conv_layer_4(x3_p)
        # print('x4.shape: ',x4.shape)
        x4_p = self.conv_layer_4_pool(x4)
        # print('x4_p.shape: ',x4_p.shape)
        x5 = self.conv_layer_5(x4_p)
        # print('x5.shape: ',x5.shape)

        # upsampling layers
        t6 = self.conv_layer_6_transposed(x5)
        # print('t6.shape: ',t6.shape)
        c6 = self.cat_6(x4, t6)
        # print('c6.shape: ',c6.shape)
        x6 = self.conv_layer_6(c6)
        # print('x6.shape: ',x6.shape)

        t7 = self.conv_layer_7_transposed(x6)
        # print('t7.shape: ',t7.shape)
        c7 = self.cat_7(x3, t7)
        # print('c7.shape: ',c7.shape)
        x7 = self.conv_layer_7(c7)
        # print('x7.shape: ',x7.shape)

        t8 = self.conv_layer_8_transposed(x7)
        c8 = self.cat_8(x2, t8)
        x8 = self.conv_layer_8(c8)

        t9 = self.conv_layer_9_transposed(x8)
        c9 = self.cat_9(x1, t9)
        x9 = self.conv_layer_9(c9)
        # print('x9 shape ', x9.shape)
        x10 = self.output(x9)
        # print('x10 shape ', x10.shape)

        image_heatmap = self.cat_image_heatmap(x, x10)

        x1 = self.stage_2_conv_layer_1(image_heatmap)
        # print('x1.shape: ',x1.shape)
        x1_p = self.conv_layer_1_pool(x1)
        # print('x1_p.shape: ',x1_p.shape)
        x2 = self.conv_layer_2(x1_p)
        # print('x2.shape: ',x2.shape)
        x2_p = self.conv_layer_2_pool(x2)
        # print('x2_p.shape: ',x2_p.shape)
        x3 = self.conv_layer_3(x2_p)
        # print('x3.shape: ',x3.shape)
        x3_p = self.conv_layer_3_pool(x3)
        # print('x3_p.shape: ',x3_p.shape)
        x4 = self.conv_layer_4(x3_p)
        # print('x4.shape: ',x4.shape)
        x4_p = self.conv_layer_4_pool(x4)
        # print('x4_p.shape: ',x4_p.shape)
        x5 = self.conv_layer_5(x4_p)
        # print('x5.shape: ',x5.shape)

        # upsampling layers
        t6 = self.conv_layer_6_transposed(x5)
        # print('t6.shape: ',t6.shape)
        c6 = self.cat_6(x4, t6)
        # print('c6.shape: ',c6.shape)
        x6 = self.conv_layer_6(c6)
        # print('x6.shape: ',x6.shape)

        t7 = self.conv_layer_7_transposed(x6)
        # print('t7.shape: ',t7.shape)
        c7 = self.cat_7(x3, t7)
        # print('c7.shape: ',c7.shape)
        x7 = self.conv_layer_7(c7)
        # print('x7.shape: ',x7.shape)

        t8 = self.conv_layer_8_transposed(x7)
        c8 = self.cat_8(x2, t8)
        x8 = self.conv_layer_8(c8)

        t9 = self.conv_layer_9_transposed(x8)
        c9 = self.cat_9(x1, t9)
        x9 = self.conv_layer_9(c9)
        # print('x9 shape ', x9.shape)
        x10 = self.output(x9)
        # print('x10 shape ', x10.shape)

        return x10