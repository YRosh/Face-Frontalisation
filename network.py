import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


''' Generator network for 128x128 RGB images '''
class G(nn.Module):
    
    def __init__(self):
        super(G, self).__init__()
        
        self.downsample = nn.Sequential(
            # Input HxW = 128x128
            nn.Conv2d(3, 16, 4, 2, 1), # Output HxW = 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1), # Output HxW = 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1), # Output HxW = 2x2
            nn.MaxPool2d((2,2)),
        )
            # At this point, we arrive at our low D representation vector, which is 512 dimensional.
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 1, 0, bias = False) # Output HxW = 4x4
        self.norm1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(True)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False) # Output HxW = 8x8
        self.norm2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False) # Output HxW = 16x16
        self.norm3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False) # Output HxW = 32x32
        self.norm4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False) # Output HxW = 64x64
        self.norm5 = nn.BatchNorm2d(16)
        self.deconv6 = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False) # Output HxW = 128x128
        self.tan = nn.Tanh()
    
    def forward(self, x, weights):
        x = self.downsample(x)
        x = self.relu(self.norm1(self.deconv1(torch.max(x,weights[5]))))
        x = self.relu(self.norm2(self.deconv2(torch.max(x,weights[4]))))
        x = self.relu(self.norm3(self.deconv3(torch.max(x,weights[3]))))
        x = self.relu(self.norm4(self.deconv4(torch.max(x,weights[2]))))
        x = self.relu(self.norm5(self.deconv5(torch.max(x,weights[1]))))
        x = self.tan(self.deconv6(torch.max(x,weights[0])))

        return x

''' Discriminator network for 128x128 RGB images '''
class D(nn.Module):
    
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
                                  nn.Conv2d(3, 16, 4, 2, 1),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(16, 32, 4, 2, 1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(32, 64, 4, 2, 1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(64, 128, 4, 2, 1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(128, 256, 4, 2, 1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(256, 512, 4, 2, 1),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(512, 1, 4, 2, 1, bias = False),
                                  nn.Sigmoid()
                                  )
    
    
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

class Dfront(nn.Module):
    
    def __init__(self):
        super(Dfront, self).__init__()
        self.main = nn.Sequential(
                                  nn.Conv2d(3, 16, 4, 2, 1),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(16, 32, 4, 2, 1),
                                  nn.BatchNorm2d(32),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(32, 64, 4, 2, 1),
                                  nn.BatchNorm2d(64),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(64, 128, 4, 2, 1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(128, 256, 4, 2, 1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(256, 512, 4, 2, 1),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, inplace = True),
                                  nn.Conv2d(512, 1, 4, 2, 1, bias = False),
                                  nn.Sigmoid()
                                  )
    
    
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

class autoencoder(nn.Module):
    
    def __init__(self):
        super(autoencoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 4, 2, 1) # Output HxW = 64x64
        self.norm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1) # Output HxW = 32x32
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1) # Output HxW = 16x16
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1) # Output HxW = 8x8
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 4, 2, 1) # Output HxW = 4x4
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 4, 2, 1) # Output HxW = 2x2
        self.pool = nn.MaxPool2d((2,2))

        # At this point, we arrive at our low D representation vector, which is 512 dimensional.

        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 1, 0, bias = False) # Output HxW = 4x4
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False) # Output HxW = 8x8
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False) # Output HxW = 16x16
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False) # Output HxW = 32x32
        self.deconv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False) # Output HxW = 64x64
        self.deconv6 = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False) # Output HxW = 128x128
        self.tan = nn.Tanh()

    
    def forward(self, x):
        features = []
        x = self.relu(self.norm1(self.conv1(x)))
        features.append(x.data)
        x = self.relu(self.norm2(self.conv2(x)))
        features.append(x.data)
        x = self.relu(self.norm3(self.conv3(x)))
        features.append(x.data)
        x = self.relu(self.norm4(self.conv4(x)))
        features.append(x.data)
        x = self.relu(self.norm5(self.conv5(x)))
        features.append(x.data)
        x = self.pool(self.conv6(x))
        features.append(x.data)
        x = self.relu(self.norm5(self.deconv1(x)))
        x = self.relu(self.norm4(self.deconv2(x)))
        x = self.relu(self.norm3(self.deconv3(x)))
        x = self.relu(self.norm2(self.deconv4(x)))
        x = self.relu(self.norm1(self.deconv5(x)))
        x = self.tan(self.deconv6(x))
        return x, features

class Gfront(nn.Module):
    
    def __init__(self):
        super(Gfront, self).__init__()
        self.features = []
        self.conv1 = nn.Conv2d(3, 16, 4, 2, 1) # Output HxW = 64x64
        self.norm1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1) # Output HxW = 32x32
        self.norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1) # Output HxW = 16x16
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1) # Output HxW = 8x8
        self.norm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 4, 2, 1) # Output HxW = 4x4
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, 4, 2, 1) # Output HxW = 2x2
        self.pool = nn.MaxPool2d((2,2))

        # At this point, we arrive at our low D representation vector, which is 512 dimensional.

        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 1, 0, bias = False) # Output HxW = 4x4
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False) # Output HxW = 8x8
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False) # Output HxW = 16x16
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False) # Output HxW = 32x32
        self.deconv5 = nn.ConvTranspose2d(32, 16, 4, 2, 1, bias = False) # Output HxW = 64x64
        self.deconv6 = nn.ConvTranspose2d(16, 3, 4, 2, 1, bias = False) # Output HxW = 128x128
        self.tan = nn.Tanh()

    
    def forward(self, x):
        features = []
        x = self.relu(self.norm1(self.conv1(x)))
        features.append(x.data)
        x = self.relu(self.norm2(self.conv2(x)))
        features.append(x.data)
        x = self.relu(self.norm3(self.conv3(x)))
        features.append(x.data)
        x = self.relu(self.norm4(self.conv4(x)))
        features.append(x.data)
        x = self.relu(self.norm5(self.conv5(x)))
        features.append(x.data)
        x = self.pool(self.conv6(x))
        features.append(x.data)
        x = self.relu(self.norm5(self.deconv1(x)))
        x = self.relu(self.norm4(self.deconv2(x)))
        x = self.relu(self.norm3(self.deconv3(x)))
        x = self.relu(self.norm2(self.deconv4(x)))
        x = self.relu(self.norm1(self.deconv5(x)))
        x = self.tan(self.deconv6(x))
        return x, features

