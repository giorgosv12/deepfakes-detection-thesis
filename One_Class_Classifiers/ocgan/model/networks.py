# Original code can be found on https://github.com/xiehousen/OCGAN-Pytorch.
# In order to use the networks for non square images, some parts of the code were altered
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

def weights_init(mod):

    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
# target output size of 5x7

# ###
class get_encoder(nn.Module):
    """
    OCGAN ENCODER NETWORK
    """

    def __init__(self):
        super(get_encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=3//2)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=3//2, stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=3//2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=3//2, stride=2)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 32, 3)
        self.conv6 = nn.Conv2d(32, 32, 3, stride = 2)
        self.leaky = nn.LeakyReLU()

    def forward(self, input):
        output = self.leaky(self.conv1(input))
        output = self.leaky(self.conv2(output))
        output = self.batch_norm_1(output)
        output = self.leaky(self.conv3(output))
        output = self.leaky(self.conv4(output))
        output = self.batch_norm_2(output)

        output = self.leaky(self.conv5(output))
        output = torch.tanh(self.conv6(output))
        output = output.view(output.size(0),-1)

        return output


class get_decoder(nn.Module):
    """
    OCGAN DECODER NETWORK
    """
    def __init__(self):
        super(get_decoder, self).__init__()
        self.conv2 = nn.ConvTranspose2d(32, 32, 3, padding=0, stride=2, output_padding=(1,0))
        self.conv3 = nn.ConvTranspose2d(32, 32, 3, padding=0)
        self.batch_norm_1 = nn.BatchNorm2d(32)  

        self.conv5 = nn.ConvTranspose2d(32, 64, 3, padding=1, stride=2, output_padding=(1,1))
        self.conv6 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(64)    

        self.conv8 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=(1,1))
        self.conv9 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(64)    

        self.conv10 = nn.ConvTranspose2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = input.view(input.size(0),32,10,18)
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))
        output = self.batch_norm_1(output)

        output = self.relu(self.conv5(output))
        output = self.relu(self.conv6(output))
        output = self.batch_norm_2(output)

        output = self.relu(self.conv8(output))
        output = self.relu(self.conv9(output))
        output = self.batch_norm_3(output)

        output = torch.sigmoid(self.conv10(output))   

        return output


class get_disc_latent(nn.Module):
    """
    DISCRIMINATOR latent NETWORK
    """

    def __init__(self):
        super(get_disc_latent, self).__init__()
        self.dense_1 = nn.Linear(5760, 1000)
        self.batch_norm_1 = nn.BatchNorm1d(1000)

        self.dense_2 = nn.Linear(1000, 256)
        self.batch_norm_2 = nn.BatchNorm1d(256)

        self.dense_3 = nn.Linear(256, 128)
        self.batch_norm_3 = nn.BatchNorm1d(128)

        self.dense_4 = nn.Linear(128, 64)
        self.batch_norm_4 = nn.BatchNorm1d(64)

        self.dense_5 = nn.Linear(64, 32)
        self.batch_norm_5 = nn.BatchNorm1d(32)

        self.dense_6 = nn.Linear(32, 16)
        self.batch_norm_6 = nn.BatchNorm1d(16)

        self.dense_7 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, input):

        output = input.view(input.size(0),-1)
        output = self.batch_norm_1(self.dense_1(output))
        output = F.relu(output)
        
        output = self.batch_norm_2(self.dense_2(output))
        output = F.relu(output)

        output = self.batch_norm_3(self.dense_3(output))
        output = F.relu(output)

        output = self.batch_norm_4(self.dense_4(output))
        output = F.relu(output)

        output = self.batch_norm_5(self.dense_5(output))
        output = F.relu(output)

        output = self.batch_norm_6(self.dense_6(output))
        output = F.relu(output)

        torch.autograd.set_detect_anomaly(True)

        output = self.dense_7(output).clone()
        output = self.sigmoid(output)

        return output

class get_disc_visual(nn.Module):
    """
    DISCRIMINATOR vision  NETWORK
    """

    def __init__(self):
        super(get_disc_visual, self).__init__()
        self.conv1 = nn.Conv2d(1,16,5,stride=2,padding=5//2)
        self.batch_norm_1 = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(16,16,5,stride=2,padding=5//2)
        self.batch_norm_2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16,16,5,stride=2,padding=5//2)
        self.batch_norm_3 = nn.BatchNorm2d(16)
                
        self.conv4 = nn.Conv2d(16,1,5,stride=2,padding=5//2)
        self.conv5 = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, input):

        output = self.batch_norm_1(self.conv1(input))
        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)
        output = torch.sigmoid(self.conv5(output))
        output = output.view(output.size(0), -1)

        return output


class get_classifier(nn.Module):
    """
    Classfier NETWORK
    """

    def __init__(self):
        super(get_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1,32,5,stride=1,padding=5//2)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(32,64,5,stride=1,padding=5//2)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,64,5,stride=1,padding=5//2)
        self.batch_norm_3 = nn.BatchNorm2d(64)
                
        self.conv4 = nn.Conv2d(64,1,5,stride=1,padding=5//2)
        self.conv5 = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, input):
        output = self.batch_norm_1(self.conv1(input))
        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)

        output = torch.sigmoid(self.conv5(output))
        output = output.view(output.size(0), -1)

        return output