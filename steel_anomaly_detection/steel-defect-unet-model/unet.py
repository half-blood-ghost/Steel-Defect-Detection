# The U-net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# convolution and ReLu layers
class convReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(convReLU, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.layer3 = nn.Dropout()
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# The unet   
class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.encode1 = convReLU(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encode2 = convReLU(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encode3 = convReLU(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encode4 = convReLU(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.encode5 = convReLU(512, 1024)
        
        self.upsample4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decode4 = convReLU(1024, 512)
        self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode3 = convReLU(512, 256)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode2 = convReLU(256, 128)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode1 = convReLU(128, 64)
        
        self.decode0 = nn.Conv2d(64, 5, kernel_size=1)
        self.softmax = nn.Softmax2d()
    
    def forward(self, x):
        enc1 = self.encode1(x)
        enc2 = self.encode2(self.pool1(enc1))
        enc3 = self.encode3(self.pool2(enc2))
        enc4 = self.encode4(self.pool3(enc3))
        
        enc5 = self.encode5(self.pool4(enc4))
        
        dec4 = self.upsample4(enc5)
        dec4 = torch.cat((enc4, dec4), dim = 1)
        dec4 = self.decode4(dec4)
        dec3 = self.upsample3(dec4)
        dec3 = torch.cat((enc3, dec3), dim = 1)
        dec3 = self.decode3(dec3)
        dec2 = self.upsample2(dec3)
        dec2 = torch.cat((enc2, dec2), dim = 1)
        dec2 = self.decode2(dec2)
        dec1 = self.upsample1(dec2)
        dec1 = torch.cat((enc1, dec1), dim = 1)
        dec1 = self.decode1(dec1)
        
        dec = self.decode0(dec1)
        result = self.softmax(dec)
        
        return result