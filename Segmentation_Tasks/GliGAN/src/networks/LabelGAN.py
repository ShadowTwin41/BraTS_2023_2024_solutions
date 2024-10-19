import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN


class Discriminator(nn.Module):
    def __init__(self, channel=512, out_class=1, in_channels:int=3):
        super(Discriminator, self).__init__()

        self.conv1 = SN(nn.Conv3d(in_channels, channel//8, kernel_size=4, stride=2, padding=1))
        self.conv2 = SN(nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1))
        self.conv3 = SN(nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1))
        self.conv4 = SN(nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1))
        self.conv5 = SN(nn.Conv3d(channel, out_class, kernel_size=4, stride=1, padding=0))
        
    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.conv2(h1), negative_slope=0.2)
        h3 = F.leaky_relu(self.conv3(h2), negative_slope=0.2)
        h4 = F.leaky_relu(self.conv4(h3), negative_slope=0.2)
        h5 = self.conv5(h4)
        output = h5
        
        return output

class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100,num_units=750):
        super(Code_Discriminator, self).__init__()

        self.l1 = nn.Sequential(SN(nn.Linear(code_size, num_units)),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(SN(nn.Linear(num_units, num_units)),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = SN(nn.Linear(num_units, 1))
        
    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = h3
            
        return output

class Encoder(nn.Module):
    def __init__(self, channel=512, out_class=1, in_channels:int=3):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.InstanceNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.InstanceNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.InstanceNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, out_class, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)
        output = h5
        
        return output
    
class Generator(nn.Module):
    def __init__(self, noise:int=100, channel:int=64, out_channels:int=3):
        super(Generator, self).__init__()
        _c = channel
        

        self.leaky_relu = nn.LeakyReLU()
        self.noise = noise
        
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.InstanceNorm3d(_c*2)
        
        self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.InstanceNorm3d(_c)
        
        self.tp_conv5 = nn.Conv3d(_c, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, noise):

        noise = noise.view(-1,self.noise,1,1,1)
        h = self.tp_conv1(noise)
        h = self.leaky_relu(self.bn1(h))
        
        h = nn.functional.interpolate(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = self.leaky_relu(self.bn2(h))
     
        h = nn.functional.interpolate(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = self.leaky_relu(self.bn3(h))

        h = nn.functional.interpolate(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = self.leaky_relu(self.bn4(h))

        h = nn.functional.interpolate(h,scale_factor = 2)
        h = self.tp_conv5(h)

        h = torch.tanh(h)

        return h