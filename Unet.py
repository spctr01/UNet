import torch
import torch.nn as nn


def dul_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace= True)
    )
    return conv

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dwn_conv1 = dul_conv(1,64)
        self.dwn_conv2 = dul_conv(64,128)
        self.dwn_conv3 = dul_conv(128,256)
        self.dwn_conv4 = dul_conv(256,512)
        self.dwn_conv5 = dul_conv(512,1024)
    
    def forward(self, image):
        #encoder
        x1 = self.dwn_conv1(image)
        print(x1.size())
        x2 = self.max_pool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.max_pool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.max_pool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.max_pool(x7)
        x9 = self.dwn_conv5(x8)
        print(x9.size())

    
if __name__ == "__main__":
    image = torch.rand((1, 1, 572, 572))
    model = Unet()
    model(image)
    
