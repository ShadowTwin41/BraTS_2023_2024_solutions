from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, args, channel=768, use_sigmoid=False):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.use_sigmoid = use_sigmoid
        
        self.conv1 = SN(nn.Conv3d(self.in_channels, self.channel//16, kernel_size=(4,4,4), stride=(2, 2, 2), padding=1))
        self.conv2 = SN(nn.Conv3d(self.channel//16, self.channel//8, kernel_size=(4,4,4), stride=(2, 2, 2), padding=1))
        self.conv3 = SN(nn.Conv3d(self.channel//8, self.channel//4, kernel_size=(4,4,4), stride=(2, 2, 2), padding=1))
        self.conv4 = SN(nn.Conv3d(self.channel//4, self.channel//2, kernel_size=(4,4,4), stride=(2, 2, 2), padding=1))
        self.conv5 = SN(nn.Conv3d(self.channel//2, self.channel, kernel_size=(4,4,4), stride=(2, 2, 2), padding=1))
        self.conv6 = nn.Conv3d(self.channel, self.out_channels, kernel_size=(3,3,3), stride=(1,1,1), padding=0) #remove the last SN
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.conv2(h1), negative_slope=0.2)
        h3 = F.leaky_relu(self.conv3(h2), negative_slope=0.2)
        h4 = F.leaky_relu(self.conv4(h3), negative_slope=0.2)
        h5 = F.leaky_relu(self.conv5(h4), negative_slope=0.2)
        h6 = self.conv6(h5)
        
        if self.use_sigmoid=="True":
            #print("USING SIGMOID")
            output = self.sigmoid(h6) # sigmoid added for prob. GANs (for saturing GANs -> The network is very unstabel, maybe sigmoid is not a good option )
        else:
            output = h6
        return output