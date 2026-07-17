# =======================================
# Library Imports
# =======================================
import copy
from dataclasses import dataclass
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from tqdm import tqdm
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

        n_upsamples = len(conv_channels)
        self.init_size = image_size // (2 ** n_upsamples)
        if self.init_size * (2 ** n_upsamples) != image_size:
            raise ValueError(f"image_size={image_size} not compatible with conv_channels={conv_channels}")

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
        return self.conv(h)


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

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, C, 1, 1)
            nn.Flatten(),             # (B, C)
            nn.Linear(c, 1)           # (B, 1)
        )

        if spectral_norm_on:
            self.head[2] = spectral_norm(self.head[2])

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
        return self.head(h)


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
# StyleGAN Generator and Discriminator (not the most complicated version, but a simplified one)
# =======================================
# Mapping Network 
class MappingNetwork(nn.Module):
    def __init__(self, 
        latent_dim: int = 64,
        style_dim: int = 64,
        n_layers: int = 4
    ) -> None:
        """
        Mapping Network for StyleGAN: Transforms latent vector z into style vector w

        Parameters:
        -----------
        latent_dim: int
            Dimensionality of the input latent vector z (default: 64)
        style_dim: int
            Dimensionality of the output style vector w (default: 64)
        n_layers: int
            Number of fully connected layers in the mapping network (default: 4)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> mapping = MappingNetwork(latent_dim=64, style_dim=64, n_layers=4)
        >>> z = torch.randn(16, 64)  # Batch of 16 latent vectors
        >>> w = mapping(z)  # Output style vectors of shape (16, 64)
        """
        super().__init__()

        layers = []
        in_dim = latent_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = style_dim

        self.net = nn.Sequential(*layers)

    def forward(self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Mapping Network

        Parameters:
        -----------
        z: torch.Tensor
            Input latent vector of shape (batch_size, latent_dim)

        Returns:
        --------       
        torch.Tensor
            Output style vector of shape (batch_size, style_dim)

        Usage Example:
        --------------
        >>> mapping = MappingNetwork(latent_dim=64, style_dim=64, n_layers=4)
        >>> z = torch.randn(16, 64)  # Batch of 16 latent vectors
        >>> w = mapping(z)  # Output style vectors of shape (16, 64)
        """
        return self.net(z)


# AdaIN Layer for Style Modulation
class AdaIN(nn.Module):
    def __init__(self, 
        channels: int,
        style_dim: int
    ) -> None:
        """
        Adaptive Instance Normalization (AdaIN) layer for style modulation in StyleGAN

        Parameters:
        -----------
        channels: int
            Number of channels in the input feature map (e.g. 128, 64, 32)
        style_dim: int
            Dimensionality of the style vector w (e.g. 64)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> adain = AdaIN(channels=128, style_dim=64)
        >>> x = torch.randn(16, 128, 8, 8)  # Batch of 16 feature maps
        >>> w = torch.randn(16, 64)  # Batch of 16 style vectors
        >>> out = adain(x, w)  # Output feature maps of shape (16, 128, 8, 8) modulated by style
        """
        super().__init__()
        self.style = nn.Linear(style_dim, channels * 2)

    def forward(self, 
        x: torch.Tensor,
        w: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the AdaIN layer

        Parameters:
        -----------
        x: torch.Tensor
            Input feature map of shape (batch_size, channels, height, width)
        w: torch.Tensor
            Style vector of shape (batch_size, style_dim)

        Returns:
        --------
        torch.Tensor
            Output feature map of shape (batch_size, channels, height, width) modulated by the style vector

        Usage Example:
        --------------
        >>> adain = AdaIN(channels=128, style_dim=64)
        >>> x = torch.randn(16, 128, 8, 8)  # Batch of 16 feature maps
        >>> w = torch.randn(16, 64)  # Batch of 16 style vectors
        >>> out = adain(x, w)  # Output feature maps of shape (16, 128, 8, 8) modulated by style
        """
        # (B, C)
        style = self.style(w)
        scale, bias = style.chunk(2, dim=1)

        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)

        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True).clamp(min=1e-8)

        x = (x - mean) / std
        return scale * x + bias


# Styled Conv Block
class StyledConvBlock(nn.Module):
    def __init__(self, 
        in_c: int,
        out_c: int, 
        style_dim: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        noise_weight: float = 0.0
    ) -> None:
        """
        Styled Convolutional Block for StyleGAN Generator, consisting of a convolutional layer followed by AdaIN and activation

        Parameters:
        -----------
        in_c: int
            Number of input channels for the convolutional layer (e.g. 128, 64)
        out_c: int
            Number of output channels for the convolutional layer (e.g. 128, 64, 32)
        style_dim: int
            Dimensionality of the style vector w for AdaIN (e.g. 64)
        kernel_size: int
            Kernel size for the convolutional layer (default: 3)    
        stride: int
            Stride for the convolutional layer (default: 1)
        padding: int
            Padding for the convolutional layer (default: 1)
        noise_weight: float
            Initial weight for noise injection (default: 0.0, no noise)

        Returns:
        --------
        None

        Usage Example: 
        -------------- 
        >>> block = StyledConvBlock(in_c=128, out_c=64, style_dim=64)
        >>> x = torch.randn(16, 128, 8, 8)  # Batch of 16 feature maps
        >>> w = torch.randn(16, 64)  # Batch of 16 style vectors
        >>> out = block(x, w)  # Output feature maps of shape (16, 64, 8, 8) modulated by style
        """
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.adain = AdaIN(out_c, style_dim)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.noise_weight = noise_weight
        self.noise = nn.Parameter(torch.zeros(1, out_c, 1, 1)) # learned per-channel noise scaling for stochastic variation

    def forward(self, 
        x: torch.Tensor,
        w: torch.Tensor, 
        use_noise: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the Styled Convolutional Block

        Parameters:
        -----------
        x: torch.Tensor
            Input feature map of shape (batch_size, in_c, height, width)
        w: torch.Tensor
            Style vector of shape (batch_size, style_dim) for AdaIN modulation
        use_noise: bool
            Whether to apply stochastic noise injection (default: True)

        Returns:
        --------
        torch.Tensor
            Output feature map of shape (batch_size, out_c, height, width) modulated by style and optionally perturbed by noise

        Usage Example:
        --------------
        >>> block = StyledConvBlock(in_c=128, out_c=64, style_dim=64)
        >>> x = torch.randn(16, 128, 8, 8)  # Batch of 16 feature maps
        >>> w = torch.randn(16, 64)  # Batch of 16 style vectors
        >>> noise = torch.randn(16, 64, 8, 8)  # Optional noise for stochastic variation
        >>> out = block(x, w, noise)  # Output feature maps of shape (16, 64, 8, 8) modulated by style and noise
        """
        x = self.conv(x)

        # stochastic noise injection (scaled)
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        if use_noise:
            x = x + self.noise_weight * torch.randn_like(x) + self.noise_weight * self.noise

        x = self.adain(x, w)
        return self.act(x)
    

# StyleGAN Generator
class StyleGANGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        style_dim: int = 64,
        channels: Tuple[int, ...] = (128, 64, 32),
        image_channels: int = 3, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        noise_weight: float = 0.05,
        style_mixing_prob: float = 0.9,
        use_style_mixing: bool = True,
        use_noise: bool = True,
        upsample_mode: Literal["nearest", "bilinear"] = "bilinear"
    ) -> None:
        """
        StyleGAN Generator architecture that consists of a mapping network to transform the latent vector into a style vector, 
        followed by a series of styled convolutional blocks that progressively upsample the feature maps, 
        and a final convolutional layer to produce the output image. 
        The generator also supports style mixing and stochastic noise injection for improved diversity.

        Parameters:
        -----------
        latent_dim: int
            Dimensionality of the input latent vector z (default: 64)
        style_dim: int
            Dimensionality of the style vector w (default: 64)
        channels: Tuple[int, ...]
            Tuple specifying the number of channels for each convolutional block (default: (128, 64, 32))
        image_channels: int
            Number of channels in the output image (e.g. 3 for RGB, 1 for grayscale, default: 3)
        kernel_size: int
            Kernel size for convolutional layers (default: 3)
        stride: int
            Stride for convolutional layers (default: 1)
        padding: int
            Padding for convolutional layers (default: 1)
        noise_weight: float
            Initial weight for noise injection in styled convolutional blocks (default: 0.0, no noise)
        style_mixing_prob: float
            Probability of applying style mixing during training (default: 0.9)
        use_style_mixing: bool
            Whether to apply style mixing during training (default: True)
        use_noise: bool
            Whether to apply stochastic noise injection in styled convolutional blocks (default: True)
        upsample_mode: Literal["nearest", "bilinear"]
            Upsampling mode for feature maps (default: "bilinear")

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> G = StyleGANGenerator(latent_dim=64, style_dim=64, channels=(128, 64, 32), image_channels=3)
        >>> z = torch.randn(16, 64)  # Batch of 16 latent vectors
        >>> images = G(z)  # Output images of shape (16, 3, 32, 32) modulated by style
        """
        super().__init__()

        self.mapping = MappingNetwork(latent_dim, style_dim)

        # learned constant input (4x4)
        self.constant = nn.Parameter(torch.randn(1, channels[0], 4, 4))

        self.blocks = nn.ModuleList()

        in_c = channels[0]
        for out_c in channels[1:]:
            self.blocks.append(StyledConvBlock(in_c, out_c, style_dim, kernel_size, stride, padding, noise_weight))
            in_c = out_c

        self.to_rgb = nn.Conv2d(in_c, image_channels, 1)

        self.style_mixing_prob = style_mixing_prob
        self.use_style_mixing = use_style_mixing
        self.use_noise = use_noise
        self.upsample_mode = upsample_mode

    def forward(self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the StyleGAN Generator

        Parameters:
        -----------
        z: torch.Tensor
            Input latent vector of shape (batch_size, latent_dim)
        
        Returns:
        --------
        torch.Tensor
            Output images of shape (batch_size, image_channels, height, width) generated by the StyleGAN architecture

        Usage Example:
        --------------
        >>> G = StyleGANGenerator(latent_dim=64, style_dim=64, channels=(128, 64, 32), image_channels=3)
        >>> z = torch.randn(16, 64)  # Batch of 16 latent vectors
        >>> images = G(z)  # Output images of shape (16, 3, 32, 32) modulated by style
        """
        B = z.size(0)

        # two styles for style mixing
        w1 = self.mapping(z)
        w2 = self.mapping(torch.randn_like(z))
        mix = (torch.rand(1).item() < self.style_mixing_prob) and self.use_style_mixing
        x = self.constant.repeat(B, 1, 1, 1)

        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode, align_corners=False if self.upsample_mode == "bilinear" else None)
            w = w2 if mix and i >= len(self.blocks) // 2 else w1
            x = block(x, w, use_noise=self.use_noise)

        return torch.tanh(self.to_rgb(x))


# StyleDiscriminator
class StyleGANDiscriminator(nn.Module):
    def __init__(self,
        image_channels: int = 3,
        channels: Tuple[int, ...] = (32, 64, 128), 
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ) -> None:
        """
        StyleGAN Discriminator architecture that consists of a series of convolutional blocks that progressively downsample the input image,
        followed by a final fully connected layer to produce the discriminator output

        Parameters:
        -----------
        image_channels: int
            Number of channels in the input image (e.g. 3 for RGB, 1 for grayscale, default: 3)
        channels: Tuple[int, ...]
            Tuple specifying the number of channels for each convolutional block in reverse order compared to the generator (default: (32, 64, 128))
        kernel_size: int
            Kernel size for convolutional layers (default: 4)
        stride: int
            Stride for convolutional layers (default: 2)
        padding: int
            Padding for convolutional layers (default: 1)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> D = StyleGANDiscriminator(image_channels=3, channels=(32, 64, 128))
        >>> images = torch.randn(16, 3, 32, 32)  # Batch of 16 images
        >>> output = D(images)  # Discriminator output of shape (16, 1)
        """
        super().__init__()

        layers = []
        in_c = image_channels

        for out_c in channels:
            conv = nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, kernel_size, stride, padding))
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = out_c

        self.conv = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 1)
        )

    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the StyleGAN Discriminator

        Parameters:
        -----------
        x: torch.Tensor
            Input images of shape (batch_size, image_channels, height, width)

        Returns:
        --------        
        torch.Tensor
            Discriminator output of shape (batch_size, 1) representing the real/fake prediction for each input image
        
        Usage Example:
        --------------
        >>> D = StyleGANDiscriminator(image_channels=3, channels=(32, 64, 128))
        >>> images = torch.randn(16, 3, 32, 32)  # Batch of 16 images
        >>> output = D(images)  # Discriminator output of shape (16, 1)
        """
        h = self.conv(x)
        return self.head(h)
    

# =======================================
# Generative Adversarial Networks (GANs)
# =======================================
@dataclass
class GANConfig:
    architecture: Literal["GAN", "CGAN", "DCGAN", "MLP_UnrolledGAN", "DC_UnrolledGAN", "StyleGAN"] = "GAN"
    loss: Literal["Default", "Wasserstein", "LeastSquare"] = "Default"
    latent_dim: int = 32   

    # For MLPs
    input_dim: int = 784    
    hidden_dims: Tuple[int, ...] = (128, 64) # to put at "conv_channels" for DCGANs and "style_channels" for StyleGANs

    # For DCGANs
    image_size: int = 28
    image_channels: int = 1 # also for StyleGANs
    kernel_size: int = 4 # also for StyleGANs discriminator
    stride: int = 2 # also for StyleGANs discriminator
    padding: int = 1 # also for StyleGANs discriminator
    noise_coef: float = 0.03 # also for StyleGANs

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
    style_dim: int = 64
    # also use latent_dim for the mapping network input
    # and image_channels for the final output channels of the generator and input channels of the discriminator
    kernel_size_style_gen: int = 3
    stride_style_gen: int = 1
    padding_style_gen: int = 1
    noise_weight: float = 0.05
    mixing_prob: float = 0.9

    # Regularization
    dropout: Optional[float] = None
    batch_norm: Optional[bool] = False
    spectral_norm_on: Optional[bool] = False

    # Training 
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
        if cfg.architecture == "StyleGAN":
            self.G = StyleGANGenerator(cfg.latent_dim, cfg.style_dim, cfg.hidden_dims, cfg.image_channels, cfg.kernel_size_style_gen, cfg.stride_style_gen, cfg.padding_style_gen, cfg.noise_weight, cfg.mixing_prob)
            self.D = StyleGANDiscriminator(cfg.image_channels, cfg.hidden_dims[::-1], cfg.kernel_size, cfg.stride, cfg.padding)
        elif cfg.architecture == "DCGAN":
            self.G = DCGANGenerator(cfg.image_size, cfg.image_channels, cfg.hidden_dims, cfg.latent_dim, cfg.kernel_size, cfg.stride, cfg.padding, cfg.dropout, cfg.batch_norm)
            self.D = DCGANDiscriminator(cfg.image_size, cfg.image_channels, cfg.hidden_dims[::-1], cfg.kernel_size, cfg.stride, cfg.padding, cfg.spectral_norm_on)
        elif cfg.architecture == "CGAN":
            self.G = CGANGenerator(cfg.latent_dim, cfg.hidden_dims, cfg.num_classes, cfg.input_dim, cfg.dropout, cfg.batch_norm)
            self.D = CGANDiscriminator(cfg.input_dim, cfg.hidden_dims[::-1], cfg.num_classes, cfg.spectral_norm_on)
        else:
            self.G = MLPGenerator(cfg.latent_dim, cfg.input_dim, cfg.hidden_dims, cfg.dropout, cfg.batch_norm)
            self.D = MLPDiscriminator(cfg.input_dim, cfg.hidden_dims[::-1], cfg.spectral_norm_on) # [::-1] reverse hiddens dims won't generally be needed for MLP since convention preferes symmetric architectures, but we do it for consistency with DCGANs

        self.G.to(device)
        self.D.to(device)

        self.ema_G = copy.deepcopy(self.G).eval() if cfg.is_ema else None

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        if cfg.architecture == "StyleGAN": # reduce the learning rate of the discriminator
            self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.learning_rate * 0.4, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
        else:
            self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

    def gradient_penalty(self, 
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient penalty for WGAN-GP

        Parameters:
        -----------
        real: torch.Tensor
            Real data samples of shape (batch_size, input_dim) for MLP or (batch_size, image_channels, image_size, image_size) for DCGAN
        fake: torch.Tensor
            Fake data samples generated by the generator, same shape as real

        Returns:
        --------        
        torch.Tensor
            Scalar tensor representing the gradient penalty

        Usage Example:
        --------------
        >>> gan = GAN(GANConfig(architecture="DCGAN", loss="Wasserstein"), device="cuda")
        >>> real = torch.randn(16, 1, 28, 28).to("cuda")  # Batch of 16 real images
        >>> z = torch.randn(16, 32).to("cuda")  # Batch of 16 noise vectors
        >>> fake = gan.G(z)  # Generate fake images
        >>> gp = gan.gradient_penalty(real, fake)  # Compute gradient penalty
        """
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
    
    def discriminator_loss(self, 
        real: torch.Tensor, 
        fake: torch.Tensor,
        real_data: Optional[torch.Tensor] = None,
        fake_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the discriminator loss based on the selected loss function in the configuration

        Parameters:
        -----------
        real: torch.Tensor
            Discriminator output for real data (shape: (batch_size, 1))
        fake: torch.Tensor
            Discriminator output for fake data (shape: (batch_size, 1))
        real_data: Optional[torch.Tensor]
            Real data samples (used for gradient penalty in WGAN-GP, shape: (batch_size, input_dim) for MLP or (batch_size, image_channels, image_size, image_size) for DCGAN)
        fake_data: Optional[torch.Tensor]
            Fake data samples generated by the generator (used for gradient penalty in WGAN-GP, same shape as real_data)

        Returns:
        --------        
        torch.Tensor
            Scalar tensor representing the discriminator loss

        Usage Example:
        --------------
        >>> gan = GAN(GANConfig(architecture="DCGAN", loss="Wasserstein"), device="cuda")
        >>> real = torch.randn(16, 1, 28, 28).to("cuda")  # Batch of 16 real images
        >>> z = torch.randn(16, 32).to("cuda")  # Batch of 16 noise vectors
        >>> fake = gan.G(z)  # Generate fake images
        >>> D_loss = gan.discriminator_loss(gan.D(real), gan.D(fake), real, fake)  # Compute discriminator loss with gradient penalty
        """
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
    
    def generator_loss(self, 
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the generator loss based on the selected loss function in the configuration

        Parameters:
        -----------
        fake: torch.Tensor
            Discriminator output for fake data (shape: (batch_size, 1))
        
        Returns:
        --------
        torch.Tensor
            Scalar tensor representing the generator loss

        Usage Example:
        --------------
        >>> gan = GAN(GANConfig(architecture="DCGAN", loss="Wasserstein"), device="cuda")
        >>> z = torch.randn(16, 32).to("cuda")  # Batch of 16 noise vectors
        >>> fake = gan.G(z)  # Generate fake images
        >>> G_loss = gan.generator_loss(gan.D(fake))  # Compute generator loss
        """
        if self.cfg.loss == "Wasserstein":
            return -fake.mean()

        if self.cfg.loss == "LeastSquare":
            return 0.5 * ((fake - 1) ** 2).mean()

        # default GAN / DCGAN
        real_labels = torch.ones_like(fake)
        return F.binary_cross_entropy_with_logits(fake, real_labels)

    def train_step(self, 
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> GANMetrics:
        """
        Perform a single training step for the GAN, including both discriminator and generator updates
        Depending on the configuration, this method can handle different architectures (GAN, CGAN, DCGAN) and loss functions (Default, Wasserstein, LeastSquare)
        as well as unrolled GAN training for improved stability, repeating discriminator updates for WGANs, and EMA updates for the generator

        Parameters:
        -----------
        x: torch.Tensor
            Input data batch (shape: (batch_size, input_dim) for MLP or (batch_size, image_channels, image_size, image_size) for DCGAN)
        y: Optional[torch.Tensor]
            Class labels for conditioning (only used for CGAN, shape: (batch_size,))

        Returns:
        --------
        GANMetrics
            Object containing the generator and discriminator losses for this training step

        Usage Example:
        --------------
        >>> gan = GAN(GANConfig(architecture="DCGAN", loss="Wasserstein"), device="cuda")
        >>> x = torch.randn(16, 1, 28, 28).to("cuda")  # Batch of 16 real images
        >>> metrics = gan.train_step(x)  # Perform a training step and get metrics
        """
        # Safely unpack input if it's a tuple (e.g. from dataloader)
        if isinstance(x, (list, tuple)):
            if len(x) == 2:
                x, y = x
            else:
                x = x[0]

        x = x.to(self.device)

        if y is not None:
            y = y.to(self.device)

        is_image_model = self.cfg.architecture in ["DCGAN", "DC_UnrolledGAN", "StyleGAN"]

        if not is_image_model:
            x = x.view(x.size(0), -1)

        batch_size = x.size(0)

        if is_image_model and self.cfg.noise_coef > 0:
            x = x + self.cfg.noise_coef * torch.randn_like(x)

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

        n_critic = 1
        if self.cfg.architecture in ["DCGAN", "DC_UnrolledGAN"]:
            n_critic = 2
        if self.cfg.loss == "Wasserstein":
            n_critic = self.cfg.n_critic 

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

            # R1 gradient penalty (critical for StyleGAN stability)
            if self.cfg.architecture == "StyleGAN":
                x.requires_grad_(True)
                real_pred = self.D(x)
                grad = torch.autograd.grad(
                    outputs=real_pred.sum(),
                    inputs=x,
                    create_graph=True
                )[0]
                r1_penalty = grad.view(batch_size, -1).pow(2).sum(1).mean()
                D_loss = D_loss + 10.0 * r1_penalty

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

    def fit(self, 
        dataloader: DataLoader,
        epochs: int,
        verbose: bool = True
    ) -> List[GANMetrics]:
        """
        Train the GAN model for a specified number of epochs using the provided dataloader

        Parameters:
        -----------
        dataloader: torch.utils.data.DataLoader
            Dataloader providing batches of real data (and optionally labels for CGAN)
        epochs: int
            Number of epochs to train the model
        verbose: bool
            Whether to print training progress and metrics for each epoch (default: True)
        
        Returns:
        --------
        List[GANMetrics]
            List of GANMetrics objects containing the generator and discriminator losses for each epoch

        Usage Example:
        --------------
        >>> gan = GAN(GANConfig(architecture="DCGAN", loss="Wasserstein"), device="cuda")
        >>> dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=64, shuffle=True)
        >>> history = gan.fit(dataloader, epochs=50)
        """
        history = []

        for epoch in tqdm(range(epochs), desc="Training process"):
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
    def sample(self, 
        n: int,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples from the trained generator

        Parameters:
        -----------
        n: int
            Number of samples to generate
        labels: Optional[torch.Tensor]
            Class labels for conditioning (only used for CGAN, shape: (n,))

        Returns:
        --------
        torch.Tensor
            Generated samples (shape depends on architecture: (n, input_dim) for MLP or (n, image_channels, image_size, image_size) for DCGAN)

        Usage Example:
        --------------
        >>> gan = GAN(GANConfig(architecture="DCGAN", loss="Wasserstein"), device="cuda")
        >>> samples = gan.sample(16)  # Generate 16 samples from the trained generator
        """
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
        if self.cfg.architecture in ["DCGAN", "DC_UnrolledGAN", "StyleGAN"]:
            return samples.cpu()

        # MLP / tabular (keep flat)
        if samples.dim() == 2:
            return samples.cpu()

        # fallback (image reshape)
        return samples.view(n, 1, self.cfg.image_size, self.cfg.image_size).cpu()
    
    def save(
        self, 
        path: str, 
        print_message: bool = False
    ) -> None:
        """
        Save the GAN model state to a file

        Parameters:
        -----------
        path: str
            File path to save the model state (e.g. "models/gan.pth")
        print_message: bool
            Whether to print a confirmation message after saving (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> gan.save("models/gan.pth")
        """
        if not path.endswith(".pth"):
            raise ValueError("File path must end with .pth")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        payload = {
            'cfg': self.cfg.__dict__,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'opt_G_state_dict': self.opt_G.state_dict(),
            'opt_D_state_dict': self.opt_D.state_dict(),
            'ema_G_state_dict': self.ema_G.state_dict() if self.cfg.is_ema and hasattr(self, "ema_G") else None
        }
        torch.save(payload, path)
        if print_message:
            print(f"Saved GAN model to {path}")

    def load(
        self,
        path: str,
        print_message: bool = False
    ) -> None:
        """
        Load the GAN model state from a file

        Parameters:
        -----------
        path: str
            File path to load the model state from (e.g. "models/gan.pth")
        print_message: bool
            Whether to print a confirmation message after loading (default: False)

        Returns:
        --------
        None

        Usage Example:
        >>> gan.load("models/gan.pth")
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        
        state_dict = torch.load(path, map_location=self.device)
        self.cfg = GANConfig(**state_dict['cfg'])
        self.G.load_state_dict(state_dict['G_state_dict'])
        self.D.load_state_dict(state_dict['D_state_dict'])
        self.opt_G.load_state_dict(state_dict['opt_G_state_dict'])
        self.opt_D.load_state_dict(state_dict['opt_D_state_dict'])
        if self.cfg.is_ema and state_dict['ema_G_state_dict'] is not None:
            if not hasattr(self, "ema_G"):
                self.ema_G = copy.deepcopy(self.G).eval()
            self.ema_G.load_state_dict(state_dict['ema_G_state_dict'])
        if print_message:
            print(f"Loaded GAN model from {path}")


# === FILE: NRT/NRT_GANs/test.py ===