import torch.nn.functional as F
import torch


class ModuleList(torch.nn.ModuleList):

    def forward(self, x: torch.Tensor):
        for i, module in enumerate(self):
            x = module(x)
        return x


class Generator_Net(torch.nn.Module):
    """
	Generator network for generating fake data
	"""

    def __init__(self, activation=torch.nn.LeakyReLU, in_channels: int = 1, n_channels: int = 32, kernel_size: int = 4,
                 std: float = 0.155):
        """
		Initialization of class object.
			:param activation: Activation function to be used in the network
			:param in_channels: Number of input channels
			:param n_channels: Number of channels in the hidden layers
			:param kernel_size: Size of the convolutional kernel
			:param std: Standard deviation of the random noise added to the input data
		"""

        super(Generator_Net, self).__init__()

        self.activation = activation
        self.in_channels = in_channels
        self.n_c = n_channels
        self.k_size = kernel_size
        self.std = std

        self.Encoder = ModuleList(
            [torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size, bias=False, stride=2, padding=1),
             torch.nn.BatchNorm2d(self.n_c),
             self.activation(),
             torch.nn.Conv2d(self.n_c, self.n_c * 2, self.k_size, bias=False, stride=2, padding=1),
             torch.nn.BatchNorm2d(self.n_c * 2),
             self.activation(),
             torch.nn.Conv2d(self.n_c * 2, self.n_c * 4, self.k_size, bias=False, stride=2, padding=1),
             torch.nn.BatchNorm2d(self.n_c * 4),
             self.activation(),
             torch.nn.Conv2d(self.n_c * 4, self.n_c * 8, self.k_size, bias=False, stride=2, padding=1),
             torch.nn.BatchNorm2d(self.n_c * 8),
             self.activation()])

        self.Decoder = ModuleList([torch.nn.ConvTranspose2d(self.n_c * 8, n_channels * 4, self.k_size, bias=False,
                                                            stride=2, padding=1, output_padding=(0, 1)),
                                   torch.nn.BatchNorm2d(n_channels * 4),
                                   self.activation(),
                                   torch.nn.ConvTranspose2d(self.n_c * 4, n_channels * 2, self.k_size, bias=False,
                                                            stride=2, padding=1, output_padding=(0, 1)),
                                   torch.nn.BatchNorm2d(n_channels * 2),
                                   self.activation(),
                                   torch.nn.ConvTranspose2d(self.n_c * 2, n_channels, self.k_size, bias=False, stride=2,
                                                            padding=1, output_padding=(0, 0)),
                                   torch.nn.BatchNorm2d(n_channels),
                                   self.activation(),
                                   torch.nn.ConvTranspose2d(self.n_c, self.in_channels, self.k_size, bias=False,
                                                            stride=2, padding=1, output_padding=(0, 0)),
                                   torch.nn.BatchNorm2d(self.in_channels),
                                   self.activation()])

    def forward(self, x: torch.Tensor, noise: bool = True):
        """
		Forward pass through the network
			:param x: Input data
			:param noise: Boolean indicating whether to add noise to the input or not
			:return: Generated fake data
		"""

        x_hat = self.add_noise(x) if noise else x
        return self.Decoder.forward(self.Encoder.forward(x_hat))

    def add_noise(self, x):
        """
		Add random noise to the input data
			:param x: Input data
			:return: Input data with added random noise
		"""

        noise = torch.randn_like(x) * self.std
        x_hat = x + noise

        return x_hat


class Discriminator_Net(torch.nn.Module):
    """
	Discriminator network for a Generative Adversarial Network (GAN).
	"""

    def __init__(self, in_resolution: tuple, activation=torch.nn.LeakyReLU, in_channels: int = 1, n_channels: int = 32,
                 kernel_size: int = 4):
        """
		Initialize the discriminator network with given parameters
			:param in_resolution: input resolution of images (height, width)
			:param activation: activation function to be used in the network, default is LeakyReLU
			:param in_channels: number of input channels in an image, default is 1
			:param n_channels: number of channels in the convolutional layers, default is 32
			:param kernel_size: size of the kernel to be used in the convolutional layers, default is 4
		"""

        super(Discriminator_Net, self).__init__()

        self.activation = activation
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.n_c = n_channels
        self.k_size = kernel_size

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size, bias=False, padding=1, stride=2),
            torch.nn.BatchNorm2d(self.n_c),
            self.activation(),
            torch.nn.Conv2d(self.n_c, self.n_c * 2, self.k_size, bias=False, stride=2, padding=1),
            torch.nn.BatchNorm2d(self.n_c * 2),
            self.activation(),
            torch.nn.Conv2d(self.n_c * 2, self.n_c * 4, self.k_size, bias=False, stride=2, padding=1),
            torch.nn.BatchNorm2d(self.n_c * 4),
            self.activation(),
            torch.nn.Conv2d(self.n_c * 4, self.n_c * 8, self.k_size, bias=False, stride=2, padding=1),
            torch.nn.BatchNorm2d(self.n_c * 8),
            self.activation())

        # Compute output dimension after conv part of D network
        self.out_dim = self._compute_out_dim()

        self.fc = torch.nn.Linear(self.out_dim, 1)

    def _compute_out_dim(self):
        """
		Computes the output dimension of the convolutional layers by passing a dummy tensor through the network.
		Returns the computed dimension.
		"""

        test_x = torch.Tensor(1, self.in_channels, self.in_resolution[0], self.in_resolution[1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        test_x = self.cnn(test_x)
        out_dim = torch.prod(torch.tensor(test_x.shape[1:])).item()
        for p in self.cnn.parameters():
            p.requires_grad = True

        return out_dim

    def forward(self, x: torch.Tensor):
        """
		Forward pass through the discriminator network.
			:param x: input tensor with shape (batch_size, in_channels, height, width)
			:return: output tensor with shape (batch_size, 1)
		"""

        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc(x)
        return out


def R_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:
    """
    Calculate reconstruction loss (L_r) for generator network.
		:param d_net: Discriminator network
		:param x_real: Real input
		:param x_fake: Fake input
		:param lambd: Scalar value for weighting between reconstruction loss and generator loss
		:return: Dictionary with reconstruction loss, generator loss, and total reconstruction loss
    """

    # Get predictions for fake input
    pred = d_net(x_fake)
    y = torch.ones_like(pred)

    # Calculate reconstruction loss
    rec_loss = F.mse_loss(x_fake, x_real)

    # Calculate generator loss
    gen_loss = F.binary_cross_entropy_with_logits(pred, y)

    # Calculate total reconstruction loss
    L_r = gen_loss + lambd * rec_loss

    return {'rec_loss': rec_loss, 'gen_loss': gen_loss, 'L_r': L_r}


def D_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
    """
    Calculate total loss for discriminator network.
		:param d_net: Discriminator network
		:param x_real: Real input
		:param x_fake: Fake input
		:return: Total loss for discriminator network
    """

    # Get predictions for real and fake input
    pred_real = d_net(x_real)
    pred_fake = d_net(x_fake.detach())

    y_real = torch.ones_like(pred_real)
    y_fake = torch.zeros_like(pred_fake)

    # Calculate loss for real input
    real_loss = F.binary_cross_entropy_with_logits(pred_real, y_real)

    # Calculate loss for fake input
    fake_loss = F.binary_cross_entropy_with_logits(pred_fake, y_fake)

    # Return total loss
    return real_loss + fake_loss
