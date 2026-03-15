# =======================================
# Library Imports
# =======================================
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
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Literal


from src.data.utils import (
    reparameterize
)


# =======================================
# MLP Encoder / Decoder
# =======================================
class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        latent_dim: int = 32,
        dropout: float = 0.2,
        use_batchnorm: bool = False
    ) -> None:
        """
        MLP Encoder for VAE
        
        Parameters:
        -----------
        input_dim: int
            Dimensionality of input data (e.g. 784 for 28x28 images)
        hidden_dims: Tuple[int, ...]
            Sizes of hidden layers (default: (128, 64))
        latent_dim: int
            Dimensionality of latent space (default: 32)
        dropout: float
            Dropout rate for hidden layers (default: 0.2)
        use_batchnorm: bool
            Whether to use BatchNorm after hidden layers (default: False)
        
        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> encoder = MLPEncoder(input_dim=784, hidden_dims=(128, 64), latent_dim=32)
        >>> x = torch.randn(16, 784)  # batch of 16 samples
        >>> mu, logvar = encoder(x)
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(prev_dim, latent_dim)
        self.logvar = nn.Linear(prev_dim, latent_dim)

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MLP Encoder

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim)

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            mu: Tensor of shape (batch_size, latent_dim) representing mean of latent distribution
            logvar: Tensor of shape (batch_size, latent_dim) representing log-variance of latent distribution

        Usage Example:
        --------------
        >>> encoder = MLPEncoder(input_dim=784, hidden_dims=(128, 64), latent_dim=32)
        >>> x = torch.randn(16, 784)  # batch of 16 samples
        >>> mu, logvar = encoder(x)
        """
        h = self.net(x)
        return self.mu(h), self.logvar(h)
    

class MLPDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 128),
        latent_dim: int = 32,
        dropout: float = 0.2,
        use_batchnorm: bool = False
    ) -> None:
        """
        MLP Decoder for VAE

        Parameters:
        -----------
        output_dim: int
            Dimensionality of output data (e.g. 784 for 28x28 images)
        hidden_dims: Tuple[int, ...]
            Sizes of hidden layers (default: (64, 128))
        latent_dim: int
            Dimensionality of latent space (default: 32)
        dropout: float
            Dropout rate for hidden layers (default: 0.2)
        use_batchnorm: bool
            Whether to use BatchNorm after hidden layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> decoder = MLPDecoder(output_dim=784, hidden_dims=(128, 64), latent_dim=32)
        >>> z = torch.randn(16, 32)  # batch of 16 latent vectors
        >>> x_hat = decoder(z)
        """
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through MLP Decoder

        Parameters:
        -----------
        z: torch.Tensor
            Latent tensor of shape (batch_size, latent_dim)

        Returns:
        --------
        torch.Tensor
            Reconstructed tensor of shape (batch_size, output_dim)

        Usage Example:
        --------------
        >>> decoder = MLPDecoder(output_dim=784, hidden_dims=(128, 64), latent_dim=32)
        >>> z = torch.randn(16, 32)  # batch of 16 latent vectors
        >>> x_hat = decoder(z)
        """
        return torch.sigmoid(self.net(z))
    

# =======================================
# CNN Encoder / Decoder
# =======================================
class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        in_channels: int = 1,
        conv_channels: Tuple[int, ...] = (32, 64),
        latent_dim: int = 32,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.2,
        use_batchnorm: bool = False
    ) -> None:
        """
        CNN Encoder for VAE

        Parameters:
        -----------
        input_size: int
            Height/Width of input images (assumes square)
        in_channels: int
            Number of channels in input images (default: 1 for grayscale)
        conv_channels: Tuple[int, ...]
            Number of channels for each convolutional layer (default: (32, 64))
        latent_dim: int
            Dimensionality of latent space (default: 32)
        kernel_size: int
            Size of convolutional kernels (default: 4)
        stride: int
            Stride for convolutional layers (default: 2)
        padding: int
            Padding for convolutional layers (default: 1)
        dropout: float
            Dropout rate for convolutional layers (default: 0.2)
        use_batchnorm: bool
            Whether to use BatchNorm after convolutional layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> encoder = CNNEncoder(input_size=28, in_channels=1, conv_channels=(32, 64), latent_dim=32)
        >>> x = torch.randn(16, 1, 28, 28)  # batch of 16 grayscale images
        >>> mu, logvar = encoder(x)
        """
        super().__init__()

        layers: List[nn.Module] = []
        c = in_channels
        size = input_size

        for h in conv_channels:
            layers.append(nn.Conv2d(c, h, kernel_size, stride, padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            c = h
            size = (size + 2 * padding - kernel_size) // stride + 1

        self.conv = nn.Sequential(*layers)
        
        # Automatically compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self.conv(dummy)
            self.flatten_dim = out.view(1, -1).size(1)
            self.feature_shape = out.shape[1:]  # (C,H,W)

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CNN Encoder

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, in_channels, input_size, input_size)

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            mu: Tensor of shape (batch_size, latent_dim) representing mean of latent distribution
            logvar: Tensor of shape (batch_size, latent_dim) representing log-variance of latent distribution

        Usage Example:
        --------------
        >>> encoder = CNNEncoder(input_size=28, in_channels=1, conv_channels=(32, 64), latent_dim=32)
        >>> x = torch.randn(16, 1, 28, 28)  # batch of 16 grayscale images
        >>> mu, logvar = encoder(x)
        """
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CNNDecoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        out_channels: int = 1,
        conv_channels: Tuple[int, ...] = (32, 64),
        latent_dim: int = 32,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.2,
        use_batchnorm: bool = False
    ) -> None:
        """
        CNN Decoder for VAE

        Parameters:
        -----------
        output_size: int
            Height/Width of output images (assumes square)
        out_channels: int
            Number of channels in output images (default: 1 for grayscale)
        conv_channels: Tuple[int, ...]
            Number of channels for each convolutional layer (default: (32, 64))
        latent_dim: int
            Dimensionality of latent space (default: 32)
        kernel_size: int
            Size of convolutional kernels (default: 4)
        stride: int
            Stride for convolutional layers (default: 2)
        padding: int
            Padding for convolutional layers (default: 1)
        dropout: float
            Dropout rate for convolutional layers (default: 0.2)
        use_batchnorm: bool
            Whether to use BatchNorm after convolutional layers (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> decoder = CNNDecoder(output_size=28, out_channels=1, conv_channels=(64, 32), latent_dim=32)
        >>> z = torch.randn(16, 32)  # batch of 16 latent vectors
        >>> x_hat = decoder(z)
        """
        super().__init__()

        self.init_size = output_size // (2 ** len(conv_channels))
        self.fc = nn.Linear( latent_dim, conv_channels[0] * self.init_size * self.init_size)

        layers: List[nn.Module] = []
        channels = list(conv_channels)
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size, stride, padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

        layers.append(nn.ConvTranspose2d(channels[-1], out_channels, kernel_size, stride, padding))
        self.deconv = nn.Sequential(*layers)

    def forward(
        self, 
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through CNN Decoder

        Parameters:
        -----------
        z: torch.Tensor
            Latent tensor of shape (batch_size, latent_dim)

        Returns:
        --------
        torch.Tensor
            Reconstructed tensor of shape (batch_size, out_channels, output_size, output_size)

        Usage Example:
        --------------
        >>> decoder = CNNDecoder(output_size=28, out_channels=1, conv_channels=(64, 32), latent_dim=32)
        >>> z = torch.randn(16, 32)  # batch of 16 latent vectors
        >>> x_hat = decoder(z)
        """
        h = self.fc(z)
        h = h.view(z.size(0), -1, self.init_size, self.init_size)
        return torch.sigmoid(self.deconv(h))


# =======================================
# Vector Quantizer
# =======================================
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 64,
        beta: float = 0.25
    ) -> None:
        """
        Vector Quantizer for VQ-VAE

        Parameters:
        -----------
        num_embeddings: int
            Number of discrete embeddings (default: 256)
        embedding_dim: int
            Dimensionality of each embedding vector (default: 64)
        beta: float
            Commitment loss weight (default: 0.25)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> vq = VectorQuantizer(num_embeddings=256, embedding_dim=64, beta=0.25)
        >>> z = torch.randn(16, 64)  # batch of 16 latent vectors
        >>> z_q, loss = vq(z)
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Vector Quantizer

        Parameters:
        -----------
        z: torch.Tensor
            Latent tensor of shape (batch_size, embedding_dim)
        
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            z_q: Quantized tensor of shape (batch_size, embedding_dim)
            loss: Scalar tensor representing VQ loss

        Usage Example:
        --------------
        >>> vq = VectorQuantizer(num_embeddings=256, embedding_dim=64, beta=0.25)
        >>> z = torch.randn(16, 64)  # batch of 16 latent vectors
        >>> z_q, loss = vq(z)
        """
        z_flat = z.view(-1, z.size(-1))
        distances = z_flat.pow(2).sum(1, keepdim=True) - 2 * z_flat @ self.embedding.weight.t() + self.embedding.weight.pow(2).sum(1)
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(z.shape)
        loss = F.mse_loss(z_q.detach(), z) * self.beta + F.mse_loss(z_q, z.detach())
        z_q = z + (z_q - z).detach()
        return z_q, loss


# =======================================
# Variational AutoEncoder (VAE) (works for all variants)
# =======================================
@dataclass
class VAEConfig:
    model_type: Literal["vae", "vqvae"] = "vae"
    architecture: Literal["mlp", "cnn"] = "mlp"

    input_dim: int = 784    
    hidden_dims: Tuple[int, ...] = (128, 64)
    latent_dim: int = 32         

    # for CNN
    image_channels: int = 1           
    image_size: int = 28
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1

    # VQ-VAE specific
    num_embeddings: int = 256
    embedding_dim: int = 64
    beta_vq: float = 0.25

    # Regularization
    dropout: float = 0.0
    use_batchnorm: bool = False

    # Training
    beta_kl: float = 1.0
    gamma: float = 0.5
    learning_rate: float = 1e-3
    step_size: int = 20
    weight_decay: float = 1e-5

    def save(
        self, 
        path: str
    ) -> None:
        """
        Save VAE configuration to a file

        Parameters:
        -----------
        path: str
            File path to save the configuration (e.g. "vae_config.json")
        
        Returns:
        --------
        None

        Usage Example:
        >>> cfg = VAEConfig(model_type="vae", architecture="mlp", input_dim=784, hidden_dims=(128, 64), latent_dim=32)
        >>> cfg.save("vae_config.json")
        """
        torch.save(self.__dict__, path)

    def load(
        self,
        path: str
    ) -> None:
        """
        Load VAE configuration from a file

        Parameters:
        -----------
        path: str
            File path to load the configuration from (e.g. "vae_config.json")

        Returns:
        --------
        None

        Usage Example:
        >>> cfg = VAEConfig()
        >>> cfg.load("vae_config.json")
        """
        state_dict = torch.load(path)
        for k, v in state_dict.items():
            setattr(self, k, v)


@dataclass
class VAEMetrics:
    loss: float = 0.0
    recon: float = 0.0
    kld: float = 0.0
    vq: float = 0.0

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
        >>> metrics = VAEMetrics()
        >>> batch = {"loss": torch.tensor(0.5), "recon": torch.tensor(0.3), "kld": torch.tensor(0.1), "vq": torch.tensor(0.1)}
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
        >>> metrics = VAEMetrics(loss=5.0, recon=3.0, kld=1.0, vq=1.0)
        >>> metrics.normalize(n=100)
        """
        for k in self.__dict__:
            setattr(self, k, getattr(self, k) / n)


class BaseVAE(nn.Module):
    def __init__(
        self, 
        cfg: VAEConfig, 
        device: str = "cpu"
    ):
        """
        Base VAE class that supports both vanilla VAE and VQ-VAE with MLP or CNN architectures

        Parameters:
        -----------
        cfg: VAEConfig
            Configuration dataclass containing all hyperparameters and settings for the VAE model
        device: str
            Device to run the model on (default: "cpu")

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> cfg = VAEConfig(model_type="vae", architecture="mlp", input_dim=784, hidden_dims=(128, 64), latent_dim=32)
        >>> vae = BaseVAE(cfg, device="cuda")
        >>> x = torch.randn(16, 784).to("cuda")  # batch of 16 samples
        >>> x_hat, mu, logvar, vq_loss = vae.forward(x)
        >>> vae.fit(dataloader, epochs=10)
        >>> vae.plot_reconstruction(x, n=10, n_rows=2, save_path="recon.png", show=True)
        """
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.to(device)

        # Encoder / Decoder selection
        if cfg.architecture == "mlp":
            self.encoder = MLPEncoder(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.dropout, cfg.use_batchnorm)
            self.decoder = MLPDecoder(cfg.input_dim, cfg.hidden_dims[::-1], cfg.latent_dim, cfg.dropout, cfg.use_batchnorm)
        else:
            self.encoder = CNNEncoder(cfg.image_size, cfg.image_channels, cfg.hidden_dims, cfg.latent_dim, cfg.kernel_size, cfg.stride, cfg.padding, cfg.dropout, cfg.use_batchnorm)
            self.decoder = CNNDecoder(cfg.image_size, cfg.image_channels, cfg.hidden_dims[::-1], cfg.latent_dim, cfg.kernel_size, cfg.stride, cfg.padding, cfg.dropout, cfg.use_batchnorm)

        # Optional VQ
        self.vq = VectorQuantizer(cfg.num_embeddings, cfg.embedding_dim, cfg.beta_vq) if cfg.model_type == "vqvae" else None
        
    # -------------------------------
    # Forward
    # -------------------------------
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim) for MLP or (batch_size, in_channels, image_size, image_size) for CNN
       
        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            x_hat: Reconstructed tensor of same shape as input
            mu: Mean tensor from encoder (shape depends on architecture)
            logvar: Log-variance tensor from encoder (shape depends on architecture)
            vq_loss: VQ loss tensor if using VQ-VAE, else None

        Usage Example:
        --------------
        >>> x = torch.randn(16, 784).to("cuda")  # batch of 16 samples
        >>> x_hat, mu, logvar, vq_loss = vae.forward(x)
        """
        mu, logvar = self.encoder(x)
        
        if self.cfg.model_type == "vae":
            z = reparameterize(mu, logvar)
            x_hat = self.decoder(z)
            return x_hat, mu, logvar, None

        z_q, vq_loss = self.vq(mu)
        x_hat = self.decoder(z_q)
        return x_hat, mu, None, vq_loss
    
    # -------------------------------
    # Loss
    # -------------------------------
    def compute_loss(
        self, 
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: Optional[torch.Tensor] = None,
        vq_loss: Optional[torch.Tensor] = None
    ) -> VAEMetrics:
        """
        Compute total loss for VAE, including reconstruction loss, KL divergence, and optional VQ loss

        Parameters:
        -----------
        x: torch.Tensor
            Original input tensor
        x_hat: torch.Tensor
            Reconstructed tensor from decoder
        mu: torch.Tensor
            Mean tensor from encoder (used for KL divergence)
        logvar: Optional[torch.Tensor], default=None
            Log-variance tensor from encoder (used for KL divergence)
        vq_loss: Optional[torch.Tensor], default=None
            VQ loss tensor if using VQ-VAE, else None

        Returns:
        --------
        VAEMetrics
            Dataclass containing total loss and individual components (reconstruction, KL divergence, VQ loss

        Usage Example:
        --------------
        >>> x = torch.randn(16, 784).to("cuda")  # batch of 16 samples
        >>> x_hat, mu, logvar, vq_loss = vae.forward(x)
        >>> loss_dict = vae.compute_loss(x, x_hat, mu, logvar, vq_loss)
        """
        recon = F.binary_cross_entropy(x_hat, x, reduction="mean")
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) if logvar is not None else torch.tensor(0.0, device=x.device)
        vq = vq_loss if vq_loss is not None else torch.tensor(0.0, device=x.device)
        total = recon + self.cfg.beta_kl * kld + vq
        return VAEMetrics(loss=total, recon=recon, kld=kld, vq=vq)
    
    # -------------------------------
    # Training Step
    # -------------------------------
    def train_step(
        self, 
        x: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> VAEMetrics:
        """
        Perform a single training step: forward pass, loss computation, backward pass, and optimizer step

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor for the current batch
        optimizer: torch.optim.Optimizer
            Optimizer to update model parameters

        Returns:
        --------
        VAEMetrics
            Dataclass containing loss values for the current batch

        Usage Example:
        --------------
        >>> x = torch.randn(16, 784).to("cuda")  # batch of 16 samples
        >>> optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        >>> batch_metrics = vae.train_step(x, optimizer)
        """
        optimizer.zero_grad(set_to_none=True)

        if self.cfg.architecture == "cnn":
            x = x.view(x.size(0), self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size)
        x_hat, mu, logvar, vq_loss = self(x)
        metrics = self.compute_loss(x, x_hat, mu, logvar, vq_loss)

        metrics.loss.backward()
        optimizer.step()

        return metrics

    # -------------------------------
    # Fit Loop
    # -------------------------------
    def fit(
        self, 
        dataloader: DataLoader,
        epochs: int,
        verbose: bool = True
    ) -> List[VAEMetrics]:
        """
        Train the VAE model for a specified number of epochs

        Parameters:
        -----------
        dataloader: DataLoader
            DataLoader providing training data batches
        epochs: int
            Number of epochs to train for
        verbose: bool
            Whether to print training progress after each epoch (default: True)

        Returns:
        --------
        List[VAEMetrics]
            List of VAEMetrics dataclasses containing loss values for each epoch

        Usage Example:
        --------------
        >>> dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        >>> metrics = vae.fit(dataloader, epochs=10, verbose=True)
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.step_size, gamma=self.cfg.gamma)

        metrics = []
        for epoch in range(1, epochs + 1):
            meter = VAEMetrics()

            for x, *_ in dataloader:
                x = x.to(self.device)
                if self.cfg.architecture == "mlp":
                    x = x.view(x.size(0), -1)

                batch_metrics = self.train_step(x, optimizer)
                meter.update(batch_metrics=batch_metrics.__dict__, batch_size=x.size(0))

            scheduler.step()

            meter.normalize(len(dataloader.dataset))
            metrics.append(meter)

            if verbose:
                print(f"Epoch {epoch:03d} | Loss {meter.loss:.4f} | Recon {meter.recon:.4f} | KLD {meter.kld:.4f} | VQ {meter.vq:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

        return metrics
    
    # -------------------------------
    # Save / Load
    # -------------------------------
    def save(
        self, 
        path: str, 
        print_message: bool = False
    ) -> None:
        """
        Save the VAE model state to a file

        Parameters:
        -----------
        path: str
            File path to save the model state (e.g. "models/vae.pt")
        print_message: bool
            Whether to print a confirmation message after saving (default: False)

        Returns:
        --------
        None

        Usage Example:
        >>> vae.save("models/vae.pt")
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        if print_message:
            print(f"Saved VAE model to {path}")

    def load(
        self,
        path: str,
        print_message: bool = False
    ) -> None:
        """
        Load the VAE model state from a file

        Parameters:
        -----------
        path: str
            File path to load the model state from (e.g. "models/vae.pt")
        print_message: bool
            Whether to print a confirmation message after loading (default: False)

        Returns:
        --------
        None

        Usage Example:
        >>> vae.load("models/vae.pt")
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        if print_message:
            print(f"Loaded VAE model from {path}")
        
    # -------------------------------
    # Reconstruction
    # -------------------------------
    @torch.no_grad()
    def reconstruct(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct input data by passing it through the encoder and decoder

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor to reconstruct (shape depends on architecture)

        Returns:
        --------
        torch.Tensor
            Reconstructed tensor of same shape as input

        Usage Example:
        --------------
        >>> x = torch.randn(16, 784).to("cuda")  # batch of 16 samples
        >>> x_hat = vae.reconstruct(x)
        """
        self.eval()
        x = x.to(self.device)
        
        if self.cfg.architecture == "mlp":
            x = x.view(x.size(0), -1)
        elif self.cfg.architecture == "cnn":
            x = x.view(x.size(0), self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size)

        x_hat, _, _, _ = self(x)
        return x_hat

    def plot_reconstruction(
        self, 
        x: torch.Tensor,
        n: int = 10,
        save_path: Optional[str] = None,
        cmap: str = "gray",
        show: bool = False, 
        img_size: float = 1.5
    ) -> None:
        """
        Plot original and reconstructed images side by side

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor containing images to reconstruct (shape depends on architecture)
        n: int
            Number of images to plot (default: 10)
        save_path: Optional[str], default=None
            Path to save the plot image (if None, plot will not be saved)
        cmap: str
            Colormap to use for displaying images (default: "gray")
        show: bool
            Whether to display the plot after creating it (default: False)
        img_size: float
            Base size in inches for each image (default: 1.5)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> x = torch.randn(10, 784).to("cuda")  # batch of 10 samples
        >>> vae.plot_reconstruction(x, n=10, save_path="recon.png", show=True)
        """
        x = x[:n]
        x_hat = self.reconstruct(x)

        fig, axs = plt.subplots(2, n, figsize=(img_size*n, img_size*2))
        axs = axs.flatten() if 2 * n > 1 else [axs]

        for i in range(n):
            img_original = x[i].cpu()
            img_recon = x_hat[i].cpu()

            # Show original
            if self.cfg.architecture == "cnn":
                img = img_original.view(self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size).permute(1, 2, 0).squeeze()
            elif self.cfg.architecture == "mlp":
                side = int(math.sqrt(self.cfg.input_dim))
                img = img_original.view(side, side)
            axs[i].imshow(img.cpu().numpy(), cmap=cmap)
            axs[i].axis("off")
            axs[i].set_title("Orig" if i < n else "Recon")

            # Overlay reconstruction if rows = 2
            if self.cfg.architecture == "cnn":
                img = img_recon.view(self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size).permute(1, 2, 0).squeeze()
            elif self.cfg.architecture == "mlp":
                side = int(math.sqrt(self.cfg.input_dim))
                img = img_recon.view(side, side)
            axs[i + n].imshow(img.cpu().numpy(), cmap=cmap)
            axs[i + n].axis("off")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved reconstruction plot to {save_path}")
        if show:
            plt.show()

    # -------------------------------
    # Sampling
    # -------------------------------
    @torch.no_grad()
    def sample(
        self,
        n: int
    ) -> torch.Tensor:
        """
        Generate new samples by sampling from the latent space and passing through the decoder

        Parameters:
        -----------
        n: int
            Number of samples to generate

        Returns:
        --------
        torch.Tensor
            Generated samples tensor of shape (n, input_dim) for MLP or (n, out_channels, image_size, image_size) for CNN

        Usage Example:
        --------------
        >>> samples = vae.sample(n=10)
        """
        self.eval()
        z_dim = self.cfg.embedding_dim if self.cfg.model_type == "vqvae" else self.cfg.latent_dim
        z = torch.randn(n, z_dim, device=self.device)
        samples = self.decoder(z)
        if self.cfg.architecture == "cnn":
            return samples.view(n, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size)
        return samples

    def plot_samples(self, 
        n: int = 10,
        n_rows: int = 2,
        save_path: Optional[str] = None,
        cmap: str = "gray",
        show: bool = False, 
        img_size: float = 1.5
    ) -> None:
        """
        Plot generated samples in a grid

        Parameters:
        -----------
        n: int
            Number of samples to generate and plot (default: 10)
        n_rows: int
            Number of rows in the plot grid (default: 2)
        save_path: Optional[str], default=None
            Path to save the plot image (if None, plot will not be saved)
        cmap: str
            Colormap to use for displaying images (default: "gray")
        show: bool
            Whether to display the plot after creating it (default: False)
        img_size: float
            Base size in inches for each image (default: 1.5)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> vae.plot_samples(n=10, n_rows=2, save_path="samples.png", show=True)
        """
        samples = self.sample(n)
        n_cols = math.ceil(n / n_rows)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(img_size*n_cols, img_size*n_rows))
        axs = axs.flatten() if n_rows * n_cols > 1 else [axs]

        for i in range(n):
            img_sample = samples[i]
            if self.cfg.architecture == "cnn":
                img = img_sample.view(self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size).permute(1, 2, 0).squeeze()
            elif self.cfg.architecture == "mlp":
                side = int(math.sqrt(self.cfg.input_dim))
                img = img_sample.view(side, side)
            axs[i].imshow(img.cpu().numpy(), cmap=cmap)
            axs[i].axis("off")

        plt.suptitle("Generated Samples")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved sample plot to {save_path}")
        if show:
            plt.show()

    
# === FILE: NRT/NRT_VAEs/test.py ===