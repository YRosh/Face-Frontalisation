from __future__ import print_function
import time
import math
import random
import os
from os import listdir
from os.path import join
from PIL import Image
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pytorch_ssim
from torchvision import transforms
from IPython.display import display
from math import log10
from lpips_pytorch import LPIPS, lpips
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from matplotlib import pyplot as plt

from data import ImagePipeline
import networks
#from vgg import Vgg16

np.random.seed(42)
random.seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)

# Where is your training dataset at?
datapath1 = '/content/drive/My Drive/CGAN2/dataset2'
datapath2 = '/content/drive/My Drive/CGAN2/dataset3'
# You can also choose which GPU you want your model to be trained on below:
gpu_id = 0
device = torch.device("cuda", gpu_id)

train_pipe = ImagePipeline(datapath1, image_size=128, random_shuffle=True, batch_size=10, device_id=gpu_id)
train_pipe.build()
m_train = train_pipe.epoch_size()
print("Size of the training set: ", m_train)
train_pipe_loader = DALIGenericIterator(train_pipe, ["profiles", "frontals"], m_train)

train_pipe_F = ImagePipeline(datapath2, image_size=128, random_shuffle=True, batch_size=10, device_id=gpu_id)
train_pipe_F.build()
m_train_F = train_pipe_F.epoch_size()
print("Size of the frontal training set: ", m_train_F)
train_pipe_loader_F = DALIGenericIterator(train_pipe_F, ["profiles", "frontals"], m_train_F)

# Generator:
netG = networks.G().to(device)
netG.apply(networks.weights_init)

# Discriminator:
netD = networks.D().to(device)
netD.apply(networks.weights_init)

# Autoencoder - Frontal Faces:
autoencoder = networks.autoencoder().to(device)
autoencoder.apply(networks.weights_init)

# Here is where you set how important each component of the loss function is:
L1_factor = 0
L2_factor = 1
GAN_factor = 0.005

criterion = nn.BCELoss() # Binary cross entropy loss
criterion_mse = nn.MSELoss()
criterion_lpips = LPIPS(
    net_type='alex',
    version='0.1'
)

# Optimizers for the generator and the discriminator (Adam is a fancier version of gradient descent with a few more bells and whistles that is used very often):
optimizerD = optim.Adam(netD.parameters(), lr = 0.0001)
optimizerG = optim.Adam(netG.parameters(), lr = 0.0001)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

# Create a directory for the output files
try:
    os.mkdir('output')
except OSError:
    pass

try:
    f = open('output/log.txt', 'w+')
except IOERROR:
    pass

#vgg = Vgg16(requires_grad=False).to(device)
ssim_loss = pytorch_ssim.SSIM(window_size = 11)

start_time = time.time()
mse_loss = torch.nn.MSELoss()

# Let's train for 30 epochs (meaning, we go through the entire training set 30 times):
for epoch in range(200):
    
    # Lets keep track of the loss values for each epoch:
    loss_L1 = 0
    loss_L2 = 0
    loss_gan = 0
    autoencoder_loss = 0

    ssim_val = 0
    lpips_val = 0
    psnr_val = 0
    mse_val = 0

    # Your train_pipe_loader will load the images one batch at a time
    # The inner loop iterates over those batches:
    
    for i, (data, data_F) in enumerate(zip(train_pipe_loader, train_pipe_loader_F), 0):
        print('\r'+str(i), end='')
        #These are your images from the current batch:
        profile = data[0]['profiles']
        frontal = data[0]['frontals']

        profile_F = data_F[0]['profiles']
        frontal_F = data_F[0]['frontals']
        
        # LPIPS MSE SSIM PSNR MSSIM
        # TRAINING THE AUTOENCODER

        autoencoder.zero_grad()
        input_f = Variable(profile_F).type('torch.FloatTensor').to(device)
        gt_f = Variable(frontal_F).type('torch.FloatTensor').to(device)
        output_f, frontal_features = autoencoder(input_f)
        loss = criterion_mse(output_f, gt_f)
        autoencoder_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        #ACTUAL GENERATOR DISCRIMINATOR PAIR

        netD.zero_grad()
        real = Variable(frontal).type('torch.FloatTensor').to(device)
        target = Variable(torch.ones(real.size()[0])).to(device)
        output = netD(real)
        # D should accept the GT images
        errD_real = criterion(output, target)
        
        profile = Variable(profile).type('torch.FloatTensor').to(device)
        generated = netG(profile, frontal_features)
        target = Variable(torch.zeros(real.size()[0])).to(device)
        output = netD(generated.detach()) # detach() because we are not training G here
        
        # D should reject the synthetic images
        errD_fake = criterion(output, target)
        
        errD = errD_real + errD_fake
        errD.backward()
        # Update D
        optimizerD.step()
        
        # TRAINING THE GENERATOR
        netG.zero_grad()
        target = Variable(torch.ones(real.size()[0])).to(device)
        output = netD(generated)
        
        # G wants to :
        # (a) have the synthetic images be accepted by D (= look like frontal images of people)
        errG_GAN = criterion(output, target)
        
        # (b) have the synthetic images resemble the ground truth frontal image
        errG_L1 = torch.mean(torch.abs(real - generated))
        errG_L2 = torch.mean(torch.pow((real - generated), 2))
        
        errG = GAN_factor * errG_GAN + L1_factor * errG_L1 + L2_factor * errG_L2
        
        loss_L1 += errG_L1.item()
        loss_L2 += errG_L2.item()
        loss_gan += errG_GAN.item()
        
        errG.backward()
        # Update G
        optimizerG.step()

        # Index value calculations
        mse = criterion_mse(generated, real)
        ssim_val += pytorch_ssim.ssim(generated, real)
        psnr_val += 20*log10(255)-10*log10(mse.item())
        lpips_val += lpips(generated, real, net_type='alex', version='0.1').item()
        mse_val += mse.item()
        
    if epoch == 0:
        print('First training epoch completed in ',(time.time() - start_time),' seconds')
    
    # reset the DALI iterator
    train_pipe_loader.reset()
    train_pipe_loader_F.reset()

    # Print the absolute values of three losses to screen:

    print('[%d/200] Training absolute losses: L1 %.7f ; L2 %.7f ; BCE %.7f' % ((epoch + 1), loss_L1/m_train, loss_L2/m_train, loss_gan/m_train))
    print('        Training frontal losses: %.7f' % (autoencoder_loss/m_train_F))
    # Print to the log file
    f.write('[%d/200] Training absolute losses: L1 %.7f ; L2 %.7f ; BCE %.7f' % ((epoch + 1), loss_L1/m_train, loss_L2/m_train, loss_gan/m_train))
    f.write('\n')
    f.write('        Training frontal losses: %.7f' % (autoencoder_loss/m_train_F))
    f.write('\n')
    f.write("SSIM Index: %.7f" % (ssim_val/m_train))
    f.write('\n')
    f.write("PSNR Index: %.7f" % (psnr_val/m_train))
    f.write('\n')
    f.write("LPIPS Index: %.7f" % (lpips_val/m_train))
    f.write('\n')
    f.write("MSE Loss Index: %.7f"% (mse_val/m_train))
    f.write('\n---------------------------------------------------------------\n')
    f.flush()
    # Save the inputs, outputs, and ground truth frontals to files:
vutils.save_image(profile.data, 'output/%03d_input.jpg' % epoch, normalize=True)
vutils.save_image(real.data, 'output/%03d_real.jpg' % epoch, normalize=True)
vutils.save_image(generated.data, 'output/%03d_generated.jpg' % epoch, normalize=True)
#vutils.save_image(front_gen.data, 'output/%03d_frontGenerated.jpg' % epoch, normalize=True)

vutils.save_image(imput_f.data, 'output/%03d_F_input.jpg' % epoch, normalize=True)
vutils.save_image(gt_f.data, 'output/%03d_F_real.jpg' % epoch, normalize=True)
vutils.save_image(output_f.data, 'output/%03d_F_generated.jpg' % epoch, normalize=True)
#vutils.save_image(front_gen.data, 'output/%03d_frontGenerated.jpg' % epoch, normalize=True)

    # Save the pre-trained Generator as well
torch.save(netG,'output/netG_%d.pt' % epoch)
torch.save(autoencoder, 'output/autoencoder_%d.pt' % epoch )
f.close()