import torch
from torch import nn

class convBlock(nn.Module):
    """
    A convolutional block consisting of a convolutional layer, batch normalization, and Leaky ReLU activation.
    """
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        """
        Initialize the convolutional block.
        :param in_channels: Number of channels in the input tensor.
        :param out_channels: Number of channels in the output tensor.
        :param stride: Stride for the convolutional layer.
        :param kernel_size: Size of the convolutional kernel. Default is 3.
        :param padding: Padding for the convolutional layer. Default is 1.
        """
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size, out_channels=out_channels,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        """
        Perform a forward pass through the convolutional block.
        :param input: Input tensor to be passed through the block.
        :return: Output tensor after passing through the block.
        """
        return self.lrelu(self.bn(self.conv(input)))


class customLoss(nn.Module):
    """
    A custom loss function consisting of a combination of Mean Squared Error and Kullback-Leibler divergence.
    """
    def __init__(self):
        """
        Initialize the custom loss function.
        """
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, recon, x, mu, logvar):
        """
        Perform a forward pass through the custom loss function.
        :param recon: Reconstructed input tensor.
        :param x: Original input tensor.
        :param mu: Mean of the latent representation.
        :param logvar: Log variance of the latent representation.
        :return: The combined loss.
        """
        loss_mse = self.mse_loss(recon, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_mse + loss_kld


class convBlockT(nn.Module):
    """
    A transposed convolutional block consisting of a transposed convolutional layer, batch normalization, and Leaky ReLU activation.
    """
    def __init__(self, in_channels, out_channels, stride, out_padding=(0, 0)):
        """
        Initialize the transposed convolutional block.
        :param in_channels: Number of channels in the input tensor.
        :param out_channels: Number of channels in the output tensor.
        :param stride: Stride for the transposed convolutional layer.
        :param out_padding: Additional zero-padding for the output of transposed convolutional layer. Default is (0, 0).
        """
        super(convBlockT, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,kernel_size=3, out_channels=out_channels,
                                       stride=stride, padding=1, output_padding=out_padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        """
        Perform a forward pass through the transposed convolutional block.
        :param input: Input tensor to be passed through the block.
        :return: Output tensor after passing through the block.
        """
        return self.lrelu(self.bn(self.conv(input)))


class Dense_VAE(nn.Module):
    """
    Class for creating a Dense Variational Autoencoder model.
    """

    def __init__(self, latent_dims):
        """
        Initialize the Dense_VAE model.
        :param latent_dims: Number of dimensions in the latent space representation.
        """
        super(Dense_VAE, self).__init__()

        self.c1 = convBlock(1, 32, 1)
        self.c2 = convBlock(32, 64, 2)
        self.c3 = convBlock(64, 128, 2)
        self.c4 = convBlock(128, 256, 2)
        self.c5 = convBlock(256, 256, 1)
        self.c1_3 = convBlock(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.c1_3 = convBlock(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)
        self.c1_4 = convBlock(in_channels=32, out_channels=128, kernel_size=1, stride=4, padding=0)
        self.c2_4 = convBlock(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0)

        self.c1_5 = convBlock(in_channels=32, out_channels=256, kernel_size=1, stride=8, padding=0)
        self.c2_5 = convBlock(in_channels=64, out_channels=256, kernel_size=1, stride=4, padding=0)
        self.c3_5 = convBlock(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0)

        self.lrelu = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(256)

        self.fc_mu = nn.Linear(256 * 12 * 20, latent_dims)
        self.fc_var = nn.Linear(256 * 12 * 20, latent_dims)

        self.decoder_input = nn.Sequential(nn.Linear(latent_dims, 256 * 12 * 20))

        self.ct1 = convBlockT(in_channels=256, out_channels=256, stride=1, out_padding=(0,0))
        self.ct2 = convBlockT(in_channels=256, out_channels=128, stride=2, out_padding=(1, 0))
        self.ct3 = convBlockT(in_channels=128, out_channels=64, stride=2, out_padding=(1, 1))
        self.ct4 = convBlockT(in_channels=64, out_channels=32, stride=2, out_padding=(1, 1))
        self.ct5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, stride=1, padding=1,kernel_size=3, output_padding=(0,0))

    def encode(self, input):
        """
        Encode the input tensor to the latent representation.
        :param input: Input tensor to be encoded.
        :return: Mean and log variance of the latent representation.
        """
        conv1 = self.c1(input)
        conv2 = self.c2(conv1)
        conv1_3 = self.c1_3(conv1)
        conv3 = self.c3(conv2 + conv1_3)
        conv1_4 = self.c1_4(conv1)
        conv2_4 = self.c2_4(conv2)
        conv4 = self.c4(conv3 + conv1_4 + conv2_4)
        conv1_5 = self.c1_5(conv1)
        conv2_5 = self.c2_5(conv2)
        conv3_5 = self.c3_5(conv3)
        conv5 = self.c5(conv4 + conv1_5 + conv2_5 + conv3_5)

        result = self.lrelu(self.bn(conv5 + conv4 + conv1_5 + conv2_5 + conv3_5))
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        """
        Decode the latent representation to the input tensor.
        :param z: Latent representation to be decoded.
        :return: Reconstructed input tensor.
        """
        result_in = self.decoder_input(z)
        result_in = result_in.view(-1, 256, 12, 20)

        convT1 = self.ct1(result_in)
        convT2 = self.ct2(convT1)
        convT3 = self.ct3(convT2)
        convT4 = self.ct4(convT3)
        convT5 = self.ct5(convT4)
        return convT5

    def reparameterize(self, mu, logvar):
        """
        Perform the reparameterization trick to obtain a sample from the latent distribution.
        :param mu: Mean of the latent representation.
        :param logvar: Log variance of the latent representation.
        :return: A sample from the latent distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        """
        Perform a forward pass through the VAEs.
        :param input: Input tensor to be passed through the VAEs.
        :return: Reconstructed input tensor, mean and log variance of the latent representation.
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var