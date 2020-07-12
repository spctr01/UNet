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

def crop_tensor(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size 
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        #downconvolution
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dwn_conv1 = dul_conv(1,64)
        self.dwn_conv2 = dul_conv(64,128)
        self.dwn_conv3 = dul_conv(128,256)
        self.dwn_conv4 = dul_conv(256,512)
        self.dwn_conv5 = dul_conv(512,1024)

        #Upconvolution
        self.trans_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size= 2, stride= 2)
        self.up_conv1 = dul_conv(1024, 512)
        self.trans_conv2 = nn.ConvTranspose2d(512, 256, kernel_size= 2, stride= 2)
        self.up_conv2 = dul_conv(512, 256)
        self.trans_conv3 = nn.ConvTranspose2d(256, 128, kernel_size= 2, stride= 2)
        self.up_conv3 = dul_conv(256, 128)
        self.trans_conv4 = nn.ConvTranspose2d(128, 64, kernel_size= 2, stride= 2)
        self.up_conv4 = dul_conv(128, 64)

        self.out = nn.Conv2d(64, 2, kernel_size=1)


    def forward(self, image):
        #encoder
        x1 = self.dwn_conv1(image)
        x2 = self.max_pool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.max_pool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.max_pool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.max_pool(x7)
        x9 = self.dwn_conv5(x8)

        
        #decoder
        x = self.trans_conv1(x9)
        y = crop_tensor(x7, x)
        x = self.up_conv1(torch.cat([x, y], 1))

        x = self.trans_conv2(x)
        y = crop_tensor(x5, x)
        x = self.up_conv2(torch.cat([x, y], 1))

        x = self.trans_conv3(x)
        y = crop_tensor(x3, x)
        x = self.up_conv3(torch.cat([x, y], 1))

        x = self.trans_conv4(x)
        y = crop_tensor(x1, x)
        x = self.up_conv4(torch.cat([x, y], 1))

        x = self.out(x)
        return x
        
        


    
if __name__ == "__main__":
    image = torch.rand((1, 1, 572, 572))
    model = Unet()
    model(image)
    
