# =======================================
# Library Imports
# =======================================
import copy
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Literal


# =======================================
# MLP GANs Generator and Discriminator
# =======================================
class MLPGenerator(nn.Module):
    def __init__(
        self, 
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: Optional[float] = None,
        batch_norm: Optional[bool] = False
    ) -> None:
        """
        MLP Generator for GANs

        Parameters:
        -----------
        latent_dim: int
            Dimensionality of the latent space (input noise vector)
        output_dim: int
            Dimensionality of the generated data (e.g. 784 for 28x28 images)
        hidden_dims: List[int]
            List of hidden layer sizes (e.g. [128, 64])
        dropout: Optional[float]
            Dropout rate for hidden layers (default: None, no dropout)
        batch_norm: Optional[bool]
            Whether to use batch normalization in hidden layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> gen = MLPGenerator(latent_dim=32, output_dim=784, hidden_dims=[128, 64])
        >>> z = torch.randn(16, 32)  # Batch of 16 noise vectors
        >>> fake_data = gen(z)  # Generated data of shape (16, 784)
        """
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(True))
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the MLP Generator

        Parameters:
        -----------
        z: torch.Tensor
            Input noise vector of shape (batch_size, latent_dim)

        Returns:
        --------
        torch.Tensor
            Generated data of shape (batch_size, output_dim)

        Usage Example:
        --------------
        >>> gen = MLPGenerator(latent_dim=32, output_dim=784, hidden_dims=[128, 64])
        >>> z = torch.randn(16, 32)  # Batch of 16 noise vectors
        >>> fake_data = gen(z)  # Generated data of shape (16, 784)
        """
        return self.net(z)


class MLPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 128],
        spectral_norm_on: Optional[bool] = False
    ) -> None:
        """
        MLP Discriminator for GANs

        Parameters:
        -----------
        input_dim: int
            Dimensionality of the input data (e.g. 784 for 28x28 images)
        hidden_dims: List[int]
            List of hidden layer sizes (e.g. [64, 128])
        spectral_norm_on: Optional[bool]
            Whether to use spectral normalization in linear layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> disc = MLPDiscriminator(input_dim=784, hidden_dims=[128, 64])
        >>> x = torch.randn(16, 784)  # Batch of 16 data samples
        >>> validity = disc(x)  # Discriminator output of shape (16, 1)
        """
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layer = nn.Linear(prev, h)
            if spectral_norm_on:
                layer = spectral_norm(layer)
            layers.append(layer)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = h

        final_layer = nn.Linear(prev, 1)
        if spectral_norm_on:
            final_layer = spectral_norm(final_layer)
        layers.append(final_layer)
        
        self.net = nn.Sequential(*layers)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the MLP Discriminator

        Parameters:
        -----------
        x: torch.Tensor
            Input data of shape (batch_size, input_dim)

        Returns:
        --------
        torch.Tensor
            Discriminator output of shape (batch_size, 1)

        Usage Example:
        --------------
        >>> disc = MLPDiscriminator(input_dim=784, hidden_dims=[128, 64])
        >>> x = torch.randn(16, 784)  # Batch of 16 data samples
        >>> validity = disc(x)  # Discriminator output of shape (16, 1)
        """
        return self.net(x)
    

# =======================================
# Deep Convolutional GANs Generator and Discriminator
# =======================================
class DCGANGenerator(nn.Module):
    def __init__(
        self, 
        image_size: int,
        image_channels: int = 1,
        conv_channels: List[int] = [128, 64],
        latent_dim: int = 32,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: Optional[float] = None,
        batch_norm: Optional[bool] = False
    ) -> None:
        """
        DCGAN Generator 

        Parameters:
        -----------
        image_size: int
            Size of the generated images (e.g. 28 for 28x28 images)
        image_channels: int
            Number of channels in the generated images (e.g. 1 for grayscale, 3 for RGB)
        conv_channels: List[int]
            List of convolutional layer channel sizes (e.g. [128, 64])
        latent_dim: int
            Dimensionality of the latent space (input noise vector)
        kernel_size: int
            Kernel size for convolutional layers (default: 3)
        stride: int
            Stride for convolutional layers (default: 2)
        padding: int
            Padding for convolutional layers (default: 1)
        dropout: Optional[float]
            Dropout rate for convolutional layers (default: None, no dropout)
        batch_norm: Optional[bool]
            Whether to use batch normalization in convolutional layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> gen = DCGANGenerator(image_size=28, image_channels=1, conv_channels
        >>> z = torch.randn(16, 32)  # Batch of 16 noise vectors
        >>> fake_images = gen(z)  # Generated images of shape (16, 1, 28, 28)
        """
        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels

        self.init_size = image_size // (2 ** len(conv_channels))

        self.fc = nn.Linear(latent_dim, conv_channels[0] * self.init_size * self.init_size)

        conv_layers: List[nn.Module] = []
        c = conv_channels[0] 
        for h in conv_channels[1:]:
            conv_layers.append(nn.ConvTranspose2d(c, h, kernel_size, stride, padding))
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(h))
            conv_layers.append(nn.ReLU(True))
            if dropout is not None:
                conv_layers.append(nn.Dropout2d(dropout))
            c = h
        conv_layers.append(nn.ConvTranspose2d(c, image_channels, kernel_size, stride, padding))
        conv_layers.append(nn.Tanh())
        self.conv = nn.Sequential(*conv_layers)

        self._init_weights()

    def _init_weights(
        self
    ) -> None:
        """
        Initialize weights of the DCGAN Generator using normal distribution

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> gen = DCGANGenerator(image_size=28, image_channels=1, conv_channels=[128, 64], latent_dim=32)
        >>> gen._init_weights()  # Initializes weights of the generator
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the DCGAN Generator

        Parameters:
        -----------
        z: torch.Tensor
            Input noise vector of shape (batch_size, latent_dim)

        Returns:
        --------
        torch.Tensor
            Generated images of shape (batch_size, image_channels, image_size, image_size)

        Usage Example:
        --------------
        >>> gen = DCGANGenerator(image_size=28, image_channels=1, conv_channels=[128, 64], latent_dim=32)
        >>> z = torch.randn(16, 32)  # Batch of 16 noise vectors
        >>> fake_images = gen(z)  # Generated images of shape (16, 1, 28, 28)
        """
        h = self.fc(z)
        h = h.view(h.size(0), -1, self.init_size, self.init_size)
        x = self.conv(h)
        return F.interpolate(x, size=(self.image_size, self.image_size))


class DCGANDiscriminator(nn.Module):
    def __init__(
        self,
        image_size: int,
        image_channels: int = 1,
        conv_channels: List[int] = [64, 128],
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        spectral_norm_on: Optional[bool] = False
    ) -> None:
        """
        DCGAN Discriminator

        Parameters:
        -----------
        image_size: int
            Size of the input images (e.g. 28 for 28x28 images)
        image_channels: int
            Number of channels in the input images (e.g. 1 for grayscale, 3 for RGB)
        conv_channels: List[int]
            List of convolutional layer channel sizes (e.g. [64, 128])
        kernel_size: int
            Kernel size for convolutional layers (default: 3)
        stride: int
            Stride for convolutional layers (default: 2)
        padding: int
            Padding for convolutional layers (default: 1)
        spectral_norm_on: Optional[bool]
            Whether to use spectral normalization in convolutional layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> disc = DCGANDiscriminator(image_size=28, image_channels=1, conv_channels=[64, 128])
        >>> x = torch.randn(16, 1, 28, 28)  # Batch of 16 images
        >>> validity = disc(x)  # Discriminator output of shape (16, 1)
        """
        super().__init__()
        self.image_size = image_size

        conv_layers: List[nn.Module] = []
        c = image_channels
        for h in conv_channels:
            conv_layer = nn.Conv2d(c, h, kernel_size, stride, padding)
            if spectral_norm_on:
                conv_layer = spectral_norm(conv_layer)
            conv_layers.append(conv_layer)
            conv_layers.append(nn.LeakyReLU(0.2, inplace=True))
            c = h
        self.conv = nn.Sequential(*conv_layers)

        # infer automatically the size of the flattened conv output
        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, image_size, image_size)
            conv_out = self.conv(dummy)
            conv_out_dim = conv_out.view(1, -1).size(1)

        final_layer = nn.Linear(conv_out_dim, 1)
        if spectral_norm_on:
            final_layer = spectral_norm(final_layer)
        self.fc = final_layer

        self.init_weights()

    def init_weights(
        self
    ) -> None:
        """
        Initialize weights of the DCGAN Discriminator using normal distribution

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> disc = DCGANDiscriminator(image_size=28, image_channels=1, conv_channels=[64, 128])
        >>> disc.init_weights()  # Initializes weights of the discriminator
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the DCGAN Discriminator

        Parameters:
        -----------
        x: torch.Tensor
            Input images of shape (batch_size, image_channels, image_size, image_size)

        Returns:
        --------
        torch.Tensor
            Discriminator output of shape (batch_size, 1)

        Usage Example:
        --------------
        >>> disc = DCGANDiscriminator(image_size=28, image_channels=1, conv_channels=[64, 128])
        >>> x = torch.randn(16, 1, 28, 28)  # Batch of 16 images
        >>> validity = disc(x)  # Discriminator output of shape (16, 1)
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


# =======================================
# Conditional GANs Generator and Discriminator (MLP version)
# =======================================
class CGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: List[int],
        num_classes: int,
        output_dim: int,
        dropout: Optional[float] = None,
        batch_norm: Optional[bool] = False
    ) -> None:
        """
        Conditional GAN Generator

        Parameters:
        -----------
        latent_dim: int
            Dimensionality of the latent space (input noise vector)
        hidden_dim: List[int]
            List of hidden layer sizes (e.g. [128, 64])
        num_classes: int
            Number of classes for conditioning
        output_dim: int
            Dimensionality of the generated data (e.g. 784 for 28x28 images)
        dropout: Optional[float]
            Dropout rate for hidden layers (default: None, no dropout)
        batch_norm: Optional[bool]
            Whether to use batch normalization in hidden layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> gen = CGANGenerator(latent_dim=32, hidden_dim=[128, 64], num_classes=10, output_dim=784)
        >>> z = torch.randn(16, 32)  # Batch of 16 noise vectors
        >>> labels = torch.randint(0, 10, (16,))  # Batch of 16 class labels
        >>> fake_data = gen(z, labels)  # Generated data of shape (16, 784)
        """
        super().__init__()
        self.embed = nn.Embedding(num_classes, latent_dim)

        layers = []
        prev = latent_dim * 2
        for h in hidden_dim:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Conditional GAN Generator

        Parameters:
        -----------
        z: torch.Tensor
            Input noise vector of shape (batch_size, latent_dim)
        y: torch.Tensor
            Class labels for conditioning of shape (batch_size,)

        Returns:
        --------
        torch.Tensor
            Generated data of shape (batch_size, output_dim)

        Usage Example:
        --------------
        >>> gen = CGANGenerator(latent_dim=32, hidden_dim=[128, 64], num_classes=10, output_dim=784)
        >>> z = torch.randn(16, 32)  # Batch of 16 noise vectors
        >>> labels = torch.randint(0, 10, (16,))  # Batch of 16 class labels
        >>> fake_data = gen(z, labels)  # Generated data of shape (16, 784)
        """
        y_embed = self.embed(y)
        x = torch.cat([z, y_embed], dim=1)
        return self.net(x)


class CGANDiscriminator(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: List[int],
        num_classes: int,
        spectral_norm_on: Optional[bool] = False
    ) -> None:
        """
        Conditional GAN Discriminator

        Parameters:
        -----------
        input_dim: int
            Dimensionality of the input data (e.g. 784 for 28x28 images)
        hidden_dim: List[int]
            List of hidden layer sizes (e.g. [64, 128])
        num_classes: int
            Number of classes for conditioning
        spectral_norm_on: Optional[bool]
            Whether to use spectral normalization in linear layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> disc = CGANDiscriminator(input_dim=784, hidden_dim=[64, 128], num_classes=10)
        >>> x = torch.randn(16, 784)  # Batch of 16 data samples
        >>> labels = torch.randint(0, 10, (16,))  # Batch of 16 class labels
        >>> validity = disc(x, labels)  # Discriminator output of shape (16, 1)
        """
        super().__init__()
        self.embed = nn.Embedding(num_classes, input_dim)

        layers = []
        prev = input_dim * 2
        for h in hidden_dim:
            layer = nn.Linear(prev, h)
            if spectral_norm_on:
                layer = spectral_norm(layer)
            layers.append(layer)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = h

        final_layer = nn.Linear(prev, 1)
        if spectral_norm_on:
            final_layer = spectral_norm(final_layer)
        layers.append(final_layer)

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Conditional GAN Discriminator

        Parameters:
        -----------
        x: torch.Tensor
            Input data of shape (batch_size, input_dim)
        y: torch.Tensor
            Class labels for conditioning of shape (batch_size,)

        Returns:
        --------
        torch.Tensor
            Discriminator output of shape (batch_size, 1)

        Usage Example:
        --------------
        >>> disc = CGANDiscriminator(input_dim=784, hidden_dim=[64, 128], num_classes=10)
        >>> x = torch.randn(16, 784)  # Batch of 16 data samples
        >>> labels = torch.randint(0, 10, (16,))  # Batch of 16 class labels
        >>> validity = disc(x, labels)  # Discriminator output of shape (16, 1)
        """
        y_embed = self.embed(y)
        x = torch.cat([x, y_embed], dim=1)
        return self.net(x)

# =======================================
# Generative Adversarial Networks (GANs)
# =======================================
@dataclass
class GANConfig:
    architecture: Literal["GAN", "CGAN", "DCGAN", "UnrolledGAN", "StyleGANs"] = "GAN"
    loss: Literal["Default", "Wasserstein", "LeastSquare"] = "Default"
    latent_dim: int = 32   

    # For MLPs
    input_dim: int = 784    
    hidden_dims: Tuple[int, ...] = (128, 64)

    # For DCGANs
    image_size: int = 28
    image_channels: int = 1
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1

    # For CGANs
    num_classes: int = 10

    # For Unrolled GANs
    unrolled_steps: int = 5 

    # For WGANs
    weight_clip: float = 0.01
    gradient_penalty_lambda: float = 10.0
    n_critic: int = 5

    # For LSGANs
    lsgan_lambda: float = 0.5

    # For StyleGANs

    # Regularization
    dropout: Optional[float] = None
    batch_norm: Optional[bool] = False
    spectral_norm_on: Optional[bool] = False

    # Training 
    gamma: float = 0.5 
    learning_rate: float = 1e-3
    step_size: int = 20
    weight_decay: float = 1e-5
    beta1: float = 0.5
    beta2: float = 0.999

    # EMA Sampling
    is_ema: bool = False
    ema_decay: float = 0.999
    
    def save(
        self, 
        path: str
    ) -> None:
        """
        Save GAN configuration to a file

        Parameters:
        -----------
        path: str
            File path to save the configuration (e.g. "gan_config.json")
        
        Returns:
        --------
        None

        Usage Example:
        >>> cfg = GANConfig(model_type="gan", architecture="mlp", input_dim=784, hidden_dims=(128, 64), latent_dim=32)
        >>> cfg.save("gan_config.json")
        """
        torch.save(self.__dict__, path)

    def load(
        self,
        path: str
    ) -> None:
        """
        Load GAN configuration from a file

        Parameters:
        -----------
        path: str
            File path to load the configuration from (e.g. "gan_config.json")

        Returns:
        --------
        None

        Usage Example:
        >>> cfg = GANConfig()
        >>> cfg.load("gan_config.json")
        """
        state_dict = torch.load(path)
        for k, v in state_dict.items():
            setattr(self, k, v)


@dataclass
class GANMetrics:
    G_loss: float = 0.0
    D_loss: float = 0.0
    
    def update(
        self, 
        batch_metrics: Dict[str, torch.Tensor],
        batch_size: int
    ) -> None:
        """
        Update metrics by accumulating values from a batch

        Parameters:
        -----------
        batch_metrics: Dict[str, torch.Tensor]
            Dictionary containing metric values for the current batch (e.g. {"loss": ..., "recon": ..., "kld": ..., "vq": ...})
        batch_size: int
            Number of samples in the current batch (used for weighted accumulation)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> metrics = GANMetrics()
        >>> batch = 
        >>> metrics.update(batch)
        """
        for k in self.__dict__:
            if k in batch_metrics:
                value = batch_metrics[k]

                if isinstance(value, torch.Tensor):
                    value = value.item()

                setattr(self, k, getattr(self, k) + value * batch_size)

    def normalize(
        self,
        n: int
    )-> None:
        """
        Normalize accumulated metrics by the number of samples

        Parameters:
        -----------
        n: int
            Total number of samples (e.g. size of dataset)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> metrics = GANMetrics(
        >>> metrics.normalize(n=100)
        """
        for k in self.__dict__:
            setattr(self, k, getattr(self, k) / n)


class GAN(nn.Module):
    def __init__(
        self, 
        cfg: GANConfig, 
        device="cpu"
    ) -> None:
        """
        GAN model class that encapsulates the generator, discriminator, training loop, and evaluation metrics
        It supports multiple GAN architectures (GAN, CGAN, DCGAN) and loss functions (Default, Wasserstein, LeastSquare)

        Parameters:
        -----------
        cfg: GANConfig
            Configuration object containing all hyperparameters and settings for the GAN model
        device: str
            Device to run the model on (e.g. "cpu", "cuda", "mps")
        
        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> cfg = GANConfig(architecture="DCGAN", loss="Wasserstein", latent
        >>> gan = GAN(cfg, device="cuda")
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Select architecture
        if cfg.architecture == "DCGAN":
            self.G = DCGANGenerator(cfg.image_size, cfg.image_channels, cfg.hidden_dims, cfg.latent_dim, cfg.kernel_size, cfg.stride, cfg.padding, cfg.dropout, cfg.batch_norm)
            self.D = DCGANDiscriminator(cfg.image_size, cfg.image_channels, cfg.hidden_dims, cfg.kernel_size, cfg.stride, cfg.padding, cfg.spectral_norm_on)
        elif cfg.architecture == "CGAN":
            self.G = CGANGenerator(cfg.latent_dim, cfg.hidden_dims, cfg.num_classes, cfg.input_dim, cfg.dropout, cfg.batch_norm)
            self.D = CGANDiscriminator(cfg.input_dim, cfg.hidden_dims, cfg.num_classes, cfg.spectral_norm_on)
        else:
            self.G = MLPGenerator(cfg.latent_dim, cfg.input_dim, cfg.hidden_dims, cfg.dropout, cfg.batch_norm)
            self.D = MLPDiscriminator(cfg.input_dim, cfg.hidden_dims, cfg.spectral_norm_on)

        self.G.to(device)
        self.D.to(device)

        self.ema_G = copy.deepcopy(self.G).eval() if cfg.is_ema else None

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    def gradient_penalty(self, real, fake):
        batch_size = real.size(0)

        alpha = torch.rand(batch_size, 1, device=self.device)
        if real.dim() == 4:
            alpha = alpha.view(batch_size, 1, 1, 1)

        interpolates = alpha * real + (1 - alpha) * fake
        interpolates.requires_grad_(True)

        d_interpolates = self.D(interpolates)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    def discriminator_loss(self, real, fake, real_data=None, fake_data=None):
        if self.cfg.loss == "Wasserstein":
            loss = -(real.mean() - fake.mean())
            if real_data is not None and fake_data is not None:
                gp = self.gradient_penalty(real_data, fake_data)
                loss += self.cfg.gradient_penalty_lambda * gp
            return loss

        if self.cfg.loss == "LeastSquare":
            return 0.5 * ((real - 1) ** 2 + (fake ** 2)).mean()

        # default GAN / DCGAN
        real_labels = torch.ones_like(real)
        fake_labels = torch.zeros_like(fake)
        return F.binary_cross_entropy_with_logits(real, real_labels) + F.binary_cross_entropy_with_logits(fake, fake_labels)
    
    def generator_loss(self, fake):
        if self.cfg.loss == "Wasserstein":
            return -fake.mean()

        if self.cfg.loss == "LeastSquare":
            return 0.5 * ((fake - 1) ** 2).mean()

        # default GAN / DCGAN
        real_labels = torch.ones_like(fake)
        return F.binary_cross_entropy_with_logits(fake, real_labels)

    def train_step(self, x, y=None):
        # Safely unpack input if it's a tuple (e.g. from dataloader)
        if isinstance(x, (list, tuple)):
            if len(x) == 2:
                x, y = x
            else:
                x = x[0]

        x = x.to(self.device)

        if y is not None:
            y = y.to(self.device)

        if self.cfg.architecture != "DCGAN":
            x = x.view(x.size(0), -1)

        batch_size = x.size(0)

        # =======================
        # Unrolled GAN training
        # =======================
        if self.cfg.architecture == "UnrolledGAN":
            backup = {k: v.clone() for k, v in self.D.state_dict().items()}

            # Unroll the discriminator for k steps
            for _ in range(self.cfg.unrolled_steps):
                z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)
                fake = self.G(z).detach()

                D_loss = self.discriminator_loss(self.D(x), self.D(fake), x, fake)

                self.opt_D.zero_grad()
                D_loss.backward()
                self.opt_D.step()

            # Train the generator with the updated discriminator
            z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)
            fake = self.G(z)

            G_loss = self.generator_loss(self.D(fake))

            self.opt_G.zero_grad()
            G_loss.backward()
            self.opt_G.step()

            self.D.load_state_dict(backup)

            return GANMetrics(G_loss.item(), D_loss.item())

        # ======================
        # Train Discriminator
        # ======================
        z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)

        n_critic = self.cfg.n_critic if self.cfg.loss == "Wasserstein" else 1

        for _ in range(n_critic):
            z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)

            if self.cfg.architecture == "CGAN":
                fake = self.G(z, y).detach()
                real_out = self.D(x, y)
                fake_out = self.D(fake, y)
            else:
                fake = self.G(z).detach()
                real_out = self.D(x)
                fake_out = self.D(fake)

            D_loss = self.discriminator_loss(real_out, fake_out, x, fake)

            self.opt_D.zero_grad()
            D_loss.backward()
            self.opt_D.step()

        # WGAN weight clipping (optional)
        if self.cfg.loss == "Wasserstein":
            for p in self.D.parameters():
                p.data.clamp_(-self.cfg.weight_clip, self.cfg.weight_clip)

        # ======================
        # Train Generator
        # ======================
        z = torch.randn(batch_size, self.cfg.latent_dim, device=self.device)

        if self.cfg.architecture == "CGAN":
            fake = self.G(z, y)
            fake_out = self.D(fake, y)
        else:
            fake = self.G(z)
            fake_out = self.D(fake)

        G_loss = self.generator_loss(fake_out)

        self.opt_G.zero_grad()
        G_loss.backward()
        self.opt_G.step()

        if self.cfg.is_ema:
            with torch.no_grad():
                for ema_param, param in zip(self.ema_G.parameters(), self.G.parameters()):
                    ema_param.data.mul_(self.cfg.ema_decay).add_(param.data, alpha=1 - self.cfg.ema_decay)

        return GANMetrics(G_loss.item(), D_loss.item())

    def fit(self, dataloader, epochs, verbose=True):
        history = []

        for epoch in range(epochs):
            meter = GANMetrics()

            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        x, y = batch
                    else:
                        x, y = batch[0], None
                else:
                    x, y = batch, None

                metrics = self.train_step(x, y)
                meter.G_loss += metrics.G_loss
                meter.D_loss += metrics.D_loss

            meter.G_loss /= len(dataloader)
            meter.D_loss /= len(dataloader)
            history.append(meter)

            if verbose:
                print(f"Epoch {epoch+1:03d} | G {meter.G_loss:.4f} | D {meter.D_loss:.4f}")

        return history

    @torch.no_grad()
    def sample(self, n, labels=None):
        z = torch.randn(n, self.cfg.latent_dim, device=self.device)

        # Select generator (EMA or not)
        if self.cfg.is_ema and hasattr(self, "ema_G"):
            G = self.ema_G
        else:
            G = self.G

        # Generate samples based on architecture
        if self.cfg.architecture == "CGAN":
            if labels is None:
                raise ValueError("CGAN requires labels for sampling")
            labels = labels.to(self.device)
            samples = G(z, labels)
        else:
            samples = G(z)
        
        # Output formatting based on architecture
        if self.cfg.architecture == "DCGAN":
            return samples.cpu()

        # MLP / tabular (keep flat)
        if samples.dim() == 2:
            return samples.cpu()

        # fallback (image reshape)
        return samples.view(n, 1, self.cfg.image_size, self.cfg.image_size).cpu()


# === FILE: NRT/NRT_GANs/test.py ===