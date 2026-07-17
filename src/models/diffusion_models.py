# =======================================
# Library Imports
# =======================================
from dataclasses import dataclass
import math
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Literal


# =======================================
# Diffusion Model Configuration
# =======================================
@dataclass
class DiffusionConfig:
    # ======================
    # Model
    # ======================
    model_type: Literal["cnn", "res_unet"] = "res_unet"
    loss: Literal["mse", "l1"] = "mse"

    num_classes: Optional[int] = None # Act as a boolean flag for conditional generation (if None, unconditional)
    cond_drop_prob: float = 0.1
    guidance_scale: float = 0.9

    image_size: int = 32
    image_channels: int = 3

    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)

    time_emb_dim: int = 128
    time_width_coef: int = 4

    # ======================
    # Convolutional tuning
    # ======================
    use_attention: bool = True
    attention_resolutions: Tuple[int, ...] = (8,)  # to apply attention to ResUnet : should be among the image_size / 2**i
    num_heads: int = 4

    dropout: float = 0.0
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    use_batch_norm: bool = False # for classic CNNs
    num_groups: int = 8
    eps_groupnorm: float = 1e-5

    down_kernel_size: int = 4
    down_stride: int = 2
    down_padding: int = 1
    down_num_res_blocks: int = 1
    up_kernel_size: int = 4
    up_stride: int = 2
    up_padding: int = 1
    up_num_res_blocks: int = 1

    # ======================
    # Diffusion
    # ======================
    timesteps: int = 1000
    beta_schedule: Literal["linear", "cosine"] = "linear"

    beta_start: float = 1e-4
    beta_end: float = 0.02
    s: float = 0.008 # small offset for cosine schedule

    # ======================
    # Training
    # ======================
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    batch_size: int = 128
    use_torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"

    # ======================
    # Sampling
    # ======================
    use_ddim: bool = False
    ddim_steps: int = 50
    use_ema: bool = False
    ema_decay: float = 0.9999

    # ======================
    # Latent Diffusion
    # ======================
    use_latent_diffusion: bool = False
    latent_dim: int = 16
    latent_hidden_dim: int = 64
    latent_kernel_size: int = 4
    latent_stride: int = 2
    latent_padding: int = 1
    latent_scale_factor: float = 0.18215


# =======================================
# Noise Scheduler
# =======================================
class NoiseScheduler:
    def __init__(self, 
        timesteps: int, 
        beta_schedule: str = "linear", 
        beta_start: float = 1e-4, 
        beta_end: float = 0.02, 
        s: float = 0.008, 
        device="cpu"
    ) -> None:
        """
        Noise scheduler for diffusion process, supporting linear and cosine beta schedules
        
        Parameters:
        -----------
        timesteps: int
            Number of diffusion steps
        beta_schedule: str
            Type of beta schedule ("linear" or "cosine")
        beta_start: float
            Starting value of beta for linear schedule (default: 1e-4)
        beta_end: float
            Ending value of beta for linear schedule (default: 0.02)
        s: float
            Small offset for cosine schedule (default: 0.008)
        device: str
            Device to store the tensors (default: "cpu")

        Returns:
        --------
        None
        
        Usage Example:
        --------------
        >>> scheduler = NoiseScheduler(timesteps=1000, beta_schedule="linear", device="cuda")
        >>> x0 = torch.randn(16, 3, 32, 32).to("cuda")  # Original images
        >>> t = torch.randint(0, 1000, (16,), device="cuda ")  # Random timesteps
        >>> noise = torch.randn_like(x0)  # Noise to be added
        >>> x_t = scheduler.q_sample(x0, t, noise)
        """
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == "cosine":
            self.betas = self.cosine_schedule(timesteps, s)
        else:
            raise ValueError("Unknown schedule")
        self.betas = self.betas.to(device)

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def cosine_schedule(self, 
        T: int,
        s: float = 0.008
    ) -> torch.Tensor:
        """
        Cosine schedule for beta values 

        Parameters: 
        -----------
        T: int
            Total number of timesteps
        s: float
            Small offset to avoid singularities (default: 0.008)

        Returns:
        --------
        torch.Tensor
            Tensor of beta values for each timestep

        Usage Example:
        --------------
        >>> scheduler = NoiseScheduler(timesteps=1000, beta_schedule="cosine", device="cuda")
        >>> betas = scheduler.cosine_schedule(1000, s=0.008)   
        """
        steps = torch.arange(T + 1, dtype=torch.float32)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi * 0.5) ** 2
        alpha_bar = f / f[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clamp(betas, 1e-5, 0.999)

    def q_sample(self, 
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """ 
        Sample from the forward diffusion process q(x_t | x_0) using the reparameterization trick

        Parameters:
        -----------
        x0: torch.Tensor
            Original images (x_0)   
        t: torch.Tensor
            Timesteps for each image in the batch
        noise: torch.Tensor
            Noise to be added to the images

        Returns:
        --------
        torch.Tensor
            Noisy images (x_t) at timestep t

        Usage Example:
        --------------
        >>> scheduler = NoiseScheduler(timesteps=1000, beta_schedule="linear", device="cuda")
        >>> x0 = torch.randn(16, 3, 32, 32).to("cuda")  # Original images
        >>> t = torch.randint(0, 1000, (16,), device="cuda")  # Random timesteps
        >>> noise = torch.randn_like(x0)  # Noise to be added
        >>> x_t = scheduler.q_sample(x0, t, noise)
        """
        return self.sqrt_alpha_bar[t].view(-1,1,1,1) * x0 + self.sqrt_one_minus_alpha_bar[t].view(-1,1,1,1) * noise


# =======================================
# Time Embedding
# =======================================
class TimeEmbedding(nn.Module):
    def __init__(self, 
        dim: int,
        time_width_coef: int = 4
    ) -> None:
        """
        Sinusoidal time embedding for diffusion models

        Parameters:
        -----------
        dim: int
            Dimension of the time embedding
        time_width_coef: int
            Width coefficient for the MLP (default: 4)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> time_emb = TimeEmbedding(dim=128, time_width_coef=4)
        >>> t = torch.randint(0, 1000, (16,), device="cuda")  # Random timesteps
        >>> t_emb = time_emb(t)  # Time embeddings for the timesteps
        """
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * time_width_coef),
            nn.ReLU(),
            nn.Linear(dim * time_width_coef, dim)
        )

    def forward(self, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for time embedding

        Parameters:
        -----------
        t: torch.Tensor
            Timesteps for which to compute the embeddings

        Returns:
        --------
        torch.Tensor
            Time embeddings for the input timesteps

        Usage Example:
        --------------
        >>> time_emb = TimeEmbedding(dim=128, time_width_coef=4)
        >>> t = torch.randint(0, 1000, (16,), device="cuda")  # Random timesteps
        >>> t_emb = time_emb(t)  # Time embeddings for the timesteps
        """
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / (half - 1))).to(t.device)

        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
    

# =======================================
# Convolutional Neural Network (CNN) for Diffusion Model (for faster and simpler implementation)
# =======================================
class CNN(nn.Module):
    def __init__(self, 
        input_channels: int,
        output_channels: int, 
        hidden_dims: List[int],
        time_emb_dim: int,
        time_width_coef: int = 4,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True
    ) -> None:
        """
        Simple CNN architecture for diffusion model, with time embedding added to each hidden layer (FiLM-style)

        Parameters:
        -----------
        input_channels: int
            Number of input channels (e.g. 3 for RGB images)
        output_channels: int
            Number of output channels (e.g. 3 for RGB images)
        hidden_dims: List[int]
            List of hidden dimensions for each convolutional layer
        time_emb_dim: int
            Dimension of the time embedding
        time_width_coef: int
            Width coefficient for the MLP in time embedding (default: 4)
        kernel_size: int
            Kernel size for convolutional layers (default: 3)
        stride: int
            Stride for convolutional layers (default: 1)
        padding: int
            Padding for convolutional layers (default: 1)
        use_batch_norm: bool
            Whether to use batch normalization after each convolutional layer (default: True)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> cnn = CNN(input_channels=3, output_channels=3, hidden_dims=[64, 128, 256], time_emb_dim=128, kernel_size=3, stride=1, padding=1, use_batch_norm=True)
        >>> x = torch.randn(16, 3, 32, 32).to("cuda")  # Input images
        >>> t = torch.randint(0, 1000, (16,), device="cuda")  # Random timesteps
        >>> t_emb = torch.randn(16, 128).to("cuda")  # Time embeddings
        >>> output = cnn(x, t_emb)  # Forward pass
        """
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None
        self.acts = nn.ModuleList()
        in_c = input_channels

        for h_dim in hidden_dims:
            self.convs.append(nn.Conv2d(in_c, h_dim, kernel_size, stride, padding))
            if use_batch_norm:
                self.bns.append(nn.BatchNorm2d(h_dim))
            self.acts.append(nn.ReLU())
            in_c = h_dim

        self.final_conv = nn.Conv2d(in_c, output_channels, kernel_size=1)
        self.time_emb = TimeEmbedding(time_emb_dim, time_width_coef)
        self.time_projs = nn.ModuleList([nn.Linear(time_emb_dim, h_dim) for h_dim in hidden_dims]) # time embedding projections for each hidden dim (FiLM-style addition)

    def forward(self, 
        x: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the CNN model

        Parameters:
        -----------
        x: torch.Tensor
            Input images (x_t) at timestep t
        t_emb: torch.Tensor
            Time embeddings for the input timesteps

        Returns:
        --------
        torch.Tensor
            Output predictions (predicted noise) for the input images

        Usage Example:  
        --------------
        >>> cnn = CNN(input_channels=3, output_channels=3, hidden_dims=[64, 128, 256], time_emb_dim=128, kernel_size=3, stride=1, padding=1, use_batch_norm=True)
        >>> x = torch.randn(16, 3, 32, 32).to("cuda")  # Input images
        >>> t_emb = torch.randn(16, 128).to("cuda")  # Time embeddings
        >>> output = cnn(x, t_emb)  # Forward pass
        """ 
        h = x
        if t_emb is not None:
            t_emb = t_emb.to(device=h.device)
            if t_emb.dim() == 1:
                t_emb = self.time_emb(t_emb)
            elif t_emb.dim() != 2:
                raise ValueError(f"Expected timestep indices or embeddings with rank 1 or 2, got shape {tuple(t_emb.shape)}")

        for idx, conv in enumerate(self.convs):
            h = conv(h)
            te = self.time_projs[idx](t_emb).unsqueeze(-1).unsqueeze(-1) # project time embedding to channels and add (broadcast over spatial dims)
            h = h + te
            if self.bns is not None:
                h = self.bns[idx](h)
            h = self.acts[idx](h)

        return self.final_conv(h)
    

# =======================================
# Residual Block for U-Net
# =======================================
class ResBlock(nn.Module):
    def __init__(self, 
        in_c: int,
        out_c: int,
        time_dim: int,
        dropout: float = 0.0,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_groups: int = 8, 
        eps_groupnorm: float = 1e-5
    ) -> None:
        """
        Residual block for U-Net architecture, with time embedding added to the hidden layer (FiLM-style)
        In a U-Net, this block is used in both the encoder and decoder paths, 
        to process the feature maps while incorporating time information in between the convolutional layers.

        Parameters:
        -----------
        in_c: int
            Number of input channels
        out_c: int
            Number of output channels
        time_dim: int
            Dimension of the time embedding
        dropout: float
            Dropout rate (default: 0.0)
        kernel_size: int
            Kernel size for convolutional layers (default: 3)
        stride: int
            Stride for convolutional layers (default: 1)
        padding: int
            Padding for convolutional layers (default: 1)
        num_groups: int
            Number of groups for GroupNorm (default: 8)
        eps_groupnorm: float
            Epsilon value for GroupNorm to avoid division by zero (default: 1e-5)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> res_block = ResBlock(in_c=64, out_c=128, time_dim=128, dropout=0.1, kernel_size=3, stride=1, padding=1, num_groups=8)
        >>> x = torch.randn(16, 64, 32, 32).to("cuda")  # Input feature maps
        >>> t_emb = torch.randn(16, 128).to("cuda")  # Time embeddings
        >>> output = res_block(x, t_emb)  # Forward pass
        """
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_c, eps=eps_groupnorm)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)

        self.time_proj = nn.Linear(time_dim, out_c)

        self.norm2 = nn.GroupNorm(num_groups, out_c, eps=eps_groupnorm)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)

        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, 
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the residual block

        Parameters:
        -----------
        x: torch.Tensor
            Input feature maps
        t: torch.Tensor
            Time embeddings for the input timesteps

        Returns:
        --------
        torch.Tensor
            Output feature maps after processing through the residual block

        Usage Example:
        --------------
        >>> res_block = ResBlock(in_c=64, out_c=128, time_dim=128, dropout=0.1, kernel_size=3, stride=1, padding=1, num_groups=8)
        >>> x = torch.randn(16, 64, 32, 32).to("cuda")  # Input feature maps
        >>> t_emb = torch.randn(16, 128).to("cuda")  # Time embeddings
        >>> output = res_block(x, t_emb)  # Forward pass
        """
        h = self.conv1(self.act1(self.norm1(x)))

        t_emb = self.time_proj(t)[:, :, None, None]
        h = h + t_emb

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        return h + self.skip(x)


# =======================================
# Attention Block
# =======================================
class AttentionBlock(nn.Module):
    def __init__(self, 
        channels: int,
        num_heads: int = 4,
        num_groups: int = 8, 
        eps_groupnorm: float = 1e-5
    ) -> None:
        """
        Attention block for U-Net architecture, allowing the model to focus on different parts of the feature maps while processing them.

        Parameters:
        -----------
        channels: int
            Number of input channels (should match the number of channels in the feature maps)
        num_heads: int
            Number of attention heads (default: 4)
        num_groups: int
            Number of groups for GroupNorm (default: 8)
        eps_groupnorm: float
            Epsilon value for GroupNorm to avoid division by zero (default: 1e-5)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> attn_block = AttentionBlock(channels=128, num_heads=4, num_groups=8)
        >>> x = torch.randn(16, 128, 32, 32).to("cuda")  # Input feature maps
        >>> output = attn_block(x)  # Forward pass
        """
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, eps=eps_groupnorm)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the attention block

        Parameters:
        -----------
        x: torch.Tensor
            Input feature maps

        Returns:
        --------
        torch.Tensor
            Output feature maps after processing through the attention block

        Usage Example:
        --------------
        >>> attn_block = AttentionBlock(channels=128, num_heads=4, num_groups=8)
        >>> x = torch.randn(16, 128, 32, 32).to("cuda")  # Input feature maps
        >>> output = attn_block(x)  # Forward pass
        """
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)

        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)

        return x + h


# =======================================
# Down/Up Sampling Convolutional Block
# =======================================
class DownBlock(nn.Module):
    def __init__(self, 
        in_c: int, 
        out_c: int, 
        time_dim: int,
        use_attn: bool = False,
        dropout: float = 0.0, 
        kernel_size: int = 3, 
        stride: int = 1,
        padding: int = 1,
        num_groups: int = 8,
        eps_groupnorm: float = 1e-5,
        num_heads: int = 4,
        num_res_blocks: int = 1,
        down_kernel_size: int = 4, 
        down_stride: int = 2,
        down_padding: int = 1
    ) -> None:
        """
        Downsampling block for U-Net architecture, consisting of a residual block followed by an optional attention block and a downsampling convolution.

        Parameters:
        -----------
        in_c: int
            Number of input channels
        out_c: int
            Number of output channels
        time_dim: int
            Dimension of the time embedding
        use_attn: bool
            Whether to use attention block (default: False)
        dropout: float
            Dropout rate (default: 0.0)
        kernel_size: int
            Kernel size for convolutional layers (default: 3)
        stride: int
            Stride for convolutional layers (default: 1)
        padding: int
            Padding for convolutional layers (default: 1)
        num_groups: int
            Number of groups for GroupNorm (default: 8)
        eps_groupnorm: float
            Epsilon value for GroupNorm to avoid division by zero (default: 1e-5)
        num_heads: int
            Number of attention heads (default: 4)
        num_res_blocks: int
            Number of residual blocks in the downsampling block (default: 1)
        down_kernel_size: int
            Kernel size for downsampling convolution (default: 4)
        down_stride: int
            Stride for downsampling convolution (default: 2)
        down_padding: int
            Padding for downsampling convolution (default: 1)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> down_block = DownBlock(in_c=64, out_c=128, time_dim=128, use_attn=True, dropout=0.1, kernel_size=3, stride=1, padding=1, num_groups=8, num_heads=4, down_kernel_size=4, down_stride=2, down_padding=1)
        >>> x = torch.randn(16, 64, 32, 32).to("cuda")  # Input feature maps
        >>> t_emb = torch.randn(16, 128).to("cuda")  # Time embeddings
        >>> output, skip = down_block(x, t_emb)  # Forward pass
        """
        super().__init__()

        self.res_blocks = nn.ModuleList()
        current_c = in_c
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(current_c, out_c, time_dim, dropout=dropout, kernel_size=kernel_size, stride=stride, padding=padding, num_groups=num_groups, eps_groupnorm=eps_groupnorm))
            current_c = out_c
        
        self.attn = AttentionBlock(out_c, num_heads=num_heads, num_groups=num_groups, eps_groupnorm=eps_groupnorm) if use_attn else nn.Identity()
        
        self.downsample = nn.Conv2d(out_c, out_c, kernel_size=down_kernel_size, stride=down_stride, padding=down_padding)

    def forward(self, 
        x: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the downsampling block

        Parameters:
        -----------
        x: torch.Tensor
            Input feature maps
        t: torch.Tensor
            Time embeddings for the input timesteps

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Output feature maps after downsampling and the skip connection feature maps

        Usage Example:
        --------------
        >>> down_block = DownBlock(in_c=64, out_c=128, time_dim=128, use_attn=True, dropout=0.1, kernel_size=3, stride=1, padding=1, num_groups=8, num_heads=4, down_kernel_size=4, down_stride=2, down_padding=1)
        >>> x = torch.randn(16, 64, 32, 32).to("cuda")  # Input feature maps
        >>> t_emb = torch.randn(16, 128).to("cuda")  # Time embeddings
        >>> output, skip = down_block(x, t_emb)  # Forward pass
        """
        for block in self.res_blocks:
            x = block(x, t)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, 
        in_c: int, 
        skip_c: int,
        out_c: int, 
        time_dim: int,
        use_attn: bool = False, 
        dropout: float = 0.0, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1, 
        num_groups: int = 8, 
        eps_groupnorm: float = 1e-5,
        num_heads: int = 4, 
        num_res_blocks: int = 1,
        up_kernel_size: int = 4, 
        up_stride: int = 2, 
        up_padding: int = 1
    ) -> None:
        """
        Upsampling block for U-Net architecture, consisting of a transposed convolution for upsampling, followed by a residual block and an optional attention block.

        Parameters:
        -----------
        in_c: int
            Number of input channels
        skip_c: int
            Number of channels from the skip connection (from the corresponding downsampling block)
        out_c: int
            Number of output channels
        time_dim: int
            Dimension of the time embedding
        use_attn: bool
            Whether to use attention block (default: False)
        dropout: float
            Dropout rate (default: 0.0)
        kernel_size: int
            Kernel size for convolutional layers (default: 3)
        stride: int
            Stride for convolutional layers (default: 1)
        padding: int
            Padding for convolutional layers (default: 1)
        num_groups: int
            Number of groups for GroupNorm (default: 8)
        eps_groupnorm: float
            Epsilon value for GroupNorm to avoid division by zero (default: 1e-5)
        num_heads: int
            Number of attention heads (default: 4)
        num_res_blocks: int
            Number of residual blocks in the upsampling block (default: 1)
        up_kernel_size: int
            Kernel size for transposed convolution (default: 4)
        up_stride: int
            Stride for transposed convolution (default: 2)
        up_padding: int
            Padding for transposed convolution (default: 1)
        
        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> up_block = UpBlock(in_c=128, out_c=64, time_dim=128, use_attn=True, dropout=0.1, kernel_size=3, stride=1, padding=1, num_groups=8, num_heads=4, up_kernel_size=4, up_stride=2, up_padding=1)
        >>> x = torch.randn(16, 128, 16, 16).to("cuda")  # Input feature maps
        >>> skip = torch.randn(16, 64, 32, 32).to("cuda")  # Skip connection feature maps
        >>> t_emb = torch.randn(16, 128).to("cuda")  # Time embeddings
        >>> output = up_block(x, skip, t_emb)  # Forward pass
        """
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=up_kernel_size, stride=up_stride, padding=up_padding)
       
        self.res_blocks = nn.ModuleList()
        current_c = out_c + skip_c
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(current_c, out_c, time_dim, dropout=dropout, kernel_size=kernel_size, stride=stride, padding=padding, num_groups=num_groups, eps_groupnorm=eps_groupnorm))
            current_c = out_c
        
        self.attn = AttentionBlock(out_c, num_heads=num_heads, num_groups=num_groups, eps_groupnorm=eps_groupnorm) if use_attn else nn.Identity()

    def forward(self, 
        x: torch.Tensor, 
        skip: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the upsampling block

        Parameters:
        -----------
        x: torch.Tensor
            Input feature maps from the previous layer
        skip: torch.Tensor
            Skip connection feature maps from the corresponding downsampling layer
        t: torch.Tensor
            Time embeddings for the input timesteps

        Returns:
        --------
        torch.Tensor
            Output feature maps after upsampling and processing through the residual and attention blocks

        Usage Example:
        --------------
        >>> up_block = UpBlock(in_c=128, out_c=64, time_dim=128, use_attn=True, dropout=0.1, kernel_size=3, stride=1, padding=1, num_groups=8, num_heads=4, up_kernel_size=4, up_stride=2, up_padding=1)
        >>> x = torch.randn(16, 128, 16, 16).to("cuda")  # Input feature maps
        >>> skip = torch.randn(16, 64, 32, 32).to("cuda")  # Skip connection feature maps
        >>> t_emb = torch.randn(16, 128).to("cuda")  # Time embeddings
        >>> output = up_block(x, skip, t_emb)  # Forward pass
        """
        x = self.upsample(x)

        if x.shape[-2:] != skip.shape[-2:]: # Safety if dimensions differs
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        for block in self.res_blocks:
            x = block(x, t)
        x = self.attn(x)
        return x


# =======================================
# U-Net Architecture for Diffusion Model
# =======================================
class UNet(nn.Module):
    def __init__(self, 
        time_emb_dim: int, 
        image_channels: int, 
        image_size: int,
        time_width_coef: int = 4,
        base_channels: int = 64, 
        channel_mults: Tuple[int, ...] = (1, 2, 4), 
        dropout: float = 0.0, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1, 
        num_groups: int = 8, 
        eps_groupnorm: float = 1e-5,
        use_attention: bool = True, 
        num_heads: int = 4, 
        attention_resolutions: Tuple[int, ...] = (8,), 
        down_kernel_size: int = 4, 
        down_stride: int = 2, 
        down_padding: int = 1,
        down_num_res_blocks: int = 1,
        up_kernel_size: int = 4, 
        up_stride: int = 2, 
        up_padding: int = 1,
        up_num_res_blocks: int = 1
    ) -> None:
        """
        U-Net architecture for diffusion model, consisting of : 
        - an encoder (downsampling path) with residual blocks and optional attention blocks
        - a bottleneck (middle) with residual blocks and an attention block
        - a decoder (upsampling path) with residual blocks and optional attention blocks
        This architecture allows the model to capture multi-scale features and incorporate time information at each level of the network.

        Parameters:
        -----------
        time_emb_dim: int
            Dimension of the time embedding
        image_channels: int
            Number of channels in the input images (e.g. 3 for RGB)
        image_size: int
            Size of the input images (assumed to be square)
        time_width_coef: int
            Width coefficient for the time embedding MLP (default: 4)
        base_channels: int
            Base number of channels for the U-Net (default: 64)
        channel_mults: Tuple[int, ...]
            Multipliers for the number of channels at each level of the U-Net (default: (1, 2, 4))
        dropout: float
            Dropout rate (default: 0.0)
        kernel_size: int
            Kernel size for convolutional layers (default: 3)
        stride: int
            Stride for convolutional layers (default: 1)
        padding: int
            Padding for convolutional layers (default: 1)
        num_groups: int
            Number of groups for GroupNorm (default: 8)
        eps_groupnorm: float
            Epsilon value for GroupNorm to avoid division by zero (default: 1e-5)
        use_attention: bool
            Whether to use attention blocks in the U-Net (default: True)
        num_heads: int
            Number of attention heads (default: 4)
        attention_resolutions: Tuple[int, ...]
            Resolutions at which to apply attention (default: (8,))
        down_kernel_size: int
            Kernel size for downsampling convolution within down blocks (default: 4)
        down_stride: int
            Stride for downsampling convolution within down blocks (default: 2)
        down_padding: int
            Padding for downsampling convolution within down blocks (default: 1)
        down_num_res_blocks: int
            Number of residual blocks in the downsampling path, per block (default: 1)
        up_kernel_size: int
            Kernel size for upsampling transposed convolution within up blocks (default: 4)
        up_stride: int
            Stride for upsampling transposed convolution within up blocks (default: 2)
        up_padding: int
            Padding for upsampling transposed convolution within up blocks (default: 1)
        up_num_res_blocks: int
            Number of residual blocks in the upsampling path per block (default: 1)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> unet = UNet(time_emb_dim=128, image_channels=3, image_size=32, base_channels=64, channel_mults=(1, 2, 4), dropout=0.1, kernel_size=3, stride=1, padding=1, num_groups=8, num_res_blocks_down=1, num_res_blocks_up=1, use_attention=True, num_heads=4, attention_resolutions=(8,))
        >>> x = torch.randn(16, 3, 32, 32).to("cuda")  # Input images
        >>> t = torch.randint(0, 1000, (16,), device="cuda")  # Random timesteps
        >>> cond = torch.randn(16, 128).to("cuda")  # Optional conditioning information
        >>> output = unet(x, t, cond)  # Forward pass
        """
        super().__init__()

        self.time_emb = TimeEmbedding(time_emb_dim, time_width_coef)
        channels = [base_channels * m for m in channel_mults]

        self.skip_channels = []
        
        self.init_conv = nn.Conv2d(image_channels, channels[0], kernel_size=kernel_size, stride=stride, padding=padding)

        # Encoder / Downsampling
        self.downs = nn.ModuleList()
        in_channel = channels[0]
        for i, out_c in enumerate(channels):
            use_attn = (image_size // (2 ** i)) in attention_resolutions and use_attention

            self.downs.append(DownBlock(in_channel, out_c, time_emb_dim, use_attn=use_attn, dropout=dropout, kernel_size=kernel_size, stride=stride, padding=padding, num_groups=num_groups, eps_groupnorm=eps_groupnorm, num_heads=num_heads, num_res_blocks=down_num_res_blocks, down_kernel_size=down_kernel_size, down_stride=down_stride, down_padding=down_padding))
            self.skip_channels.append(out_c)
            in_channel = out_c

        # Bottleneck / Middle
        self.mid1 = nn.ModuleList([
            ResBlock(in_channel, in_channel, time_emb_dim, dropout=dropout, kernel_size=kernel_size, stride=stride, padding=padding, num_groups=num_groups, eps_groupnorm=eps_groupnorm)
            for _ in range(down_num_res_blocks)
        ])
        self.mid_attn = AttentionBlock(in_channel, num_heads=num_heads, num_groups=num_groups, eps_groupnorm=eps_groupnorm)
        self.mid2 = nn.ModuleList([
            ResBlock(in_channel, in_channel, time_emb_dim, dropout=dropout, kernel_size=kernel_size, stride=stride, padding=padding, num_groups=num_groups, eps_groupnorm=eps_groupnorm)
            for _ in range(up_num_res_blocks)
        ])

        # Decoder / Upsampling
        self.ups = nn.ModuleList()
        for i, out_c in reversed(list(enumerate(channels))):
            skip_c = self.skip_channels.pop()
            use_attn = (image_size // (2 ** i)) in attention_resolutions and use_attention
            self.ups.append(UpBlock(in_channel, skip_c, out_c, time_emb_dim, use_attn=use_attn, dropout=dropout, kernel_size=kernel_size, stride=stride, padding=padding, num_groups=num_groups, eps_groupnorm=eps_groupnorm, num_heads=num_heads, num_res_blocks=up_num_res_blocks, up_kernel_size=up_kernel_size, up_stride=up_stride, up_padding=up_padding))
            in_channel = out_c

        self.final = nn.Conv2d(in_channel, image_channels, 1)

    def forward(self, 
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the U-Net model

        Parameters:
        -----------
        x: torch.Tensor
            Input images (x_t) at timestep t
        t: torch.Tensor
            Timesteps for each image in the batch
        cond: Optional[torch.Tensor]
            Optional conditioning information (e.g. class embeddings) to be added to the time embeddings

        Returns:
        --------
        torch.Tensor
            Output predictions (predicted noise) for the input images

        Usage Example:
        --------------
        >>> unet = UNet(time_emb_dim=128, image_channels=3, image_size=32, base_channels=64, channel_mults=(1, 2, 4), dropout=0.1, kernel_size=3, stride=1, padding=1, num_groups=8, num_res_blocks_down=1, num_res_blocks_up=1, use_attention=True, num_heads=4, attention_resolutions=(8,))
        >>> x = torch.randn(16, 3, 32, 32).to("cuda")  # Input images
        >>> t = torch.randint(0, 1000, (16,), device="cuda")  # Random timesteps
        >>> cond = torch.randn(16, 128).to("cuda")  # Optional conditioning information
        >>> output = unet(x, t, cond)  # Forward pass
        """
        t_emb = self.time_emb(t)
        if cond is not None:
            t_emb = t_emb + cond
        x = self.init_conv(x)

        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        for mid in self.mid1:
            x = mid(x, t_emb)
        x = self.mid_attn(x)
        for mid in self.mid2:
            x = mid(x, t_emb)

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb)

        return self.final(x)
    

# =======================================
# AutoEncoder for Latent Diffusion
# =======================================
class LatentAutoEncoder(nn.Module):
    def __init__(self, 
        in_c: int, 
        latent_dim: int, 
        hidden_dim: int = 32, 
        kernel_size: int = 4, 
        stride: int = 2, 
        padding: int = 1, 
        scale_factor: float = 0.18215
    ) -> None:
        """
        AutoEncoder for Latent Diffusion, consisting of an encoder and a decoder. 
        The encoder compresses the input images into a latent representation, 
        while the decoder reconstructs the images from the latent representation.

        Parameters:
        -----------
        in_c: int
            Number of input channels (e.g. 3 for RGB images)
        latent_dim: int
            Dimension of the latent representation
        hidden_dim: int
            Number of hidden channels in the encoder and decoder (default: 32)
        kernel_size: int
            Kernel size for convolutional layers (default: 4)
        stride: int
            Stride for convolutional layers (default: 2)
        padding: int
            Padding for convolutional layers (default: 1)
        scale_factor: float
            Scale factor for the latent representation (default: 0.18215)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> ae = LatentAutoEncoder(in_c=3, latent_dim=4, hidden_dim=32, kernel_size=4, stride=2, padding=1, scale_factor=0.18215)
        >>> x = torch.randn(16, 3, 32, 32).to("cuda")  # Input images
        >>> latent = ae.encode(x)  # Encode to latent representation
        >>> reconstructed = ae.decode(latent)  # Decode back to images
        """
        super().__init__()
        self.scale = scale_factor

        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size, stride, padding)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim, kernel_size, stride, padding),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_c, kernel_size, stride, padding),
            nn.Tanh()
        )
         
    def encode(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode the input images into a latent representation

        Parameters:
        -----------
        x: torch.Tensor
            Input images (x_t) at timestep t

        Returns:
        --------
        torch.Tensor
            Latent representation of the input images

        Usage Example:
        --------------
        >>> ae = LatentAutoEncoder(in_c=3, latent_dim=4, hidden_dim=32, kernel_size=4, stride=2, padding=1, scale_factor=0.18215)
        >>> x = torch.randn(16, 3, 32, 32).to("cuda")  # Input images
        >>> latent = ae.encode(x)  # Encode to latent representation
        """
        z = self.encoder(x)
        return z * self.scale

    def decode(self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode the latent representation back into images

        Parameters:
        -----------
        z: torch.Tensor
            Latent representation of the images

        Returns:
        --------
        torch.Tensor
            Reconstructed images from the latent representation

        Usage Example:
        --------------
        >>> ae = LatentAutoEncoder(in_c=3, latent_dim=4, hidden_dim=32, kernel_size=4, stride=2, padding=1, scale_factor=0.18215)
        >>> latent = torch.randn(16, 4, 8, 8).to("cuda")  # Latent representation
        >>> reconstructed = ae.decode(latent)  # Decode back to images
        """
        return self.decoder(z / self.scale)
    

# =======================================
# Exponential Moving Average (EMA) for Model Parameters
# =======================================
class EMA:
    def __init__(self, 
        model: nn.Module,
        decay: float = 0.9999
    ) -> None:
        """
        Exponential Moving Average (EMA) for model parameters, 
        used to maintain a smoothed version of the model weights during training

        Parameters:
        -----------
        model: nn.Module
            The model whose parameters will be tracked with EMA
        decay: float
            Decay rate for the EMA (default: 0.9999)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> model = UNet(time_emb_dim=128, image_channels=3, image_size=32)
        >>> ema = EMA(model, decay=0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self) -> None:
        """
        Update the EMA parameters based on the current model parameters

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> model = UNet(time_emb_dim=128, image_channels=3, image_size=32)
        >>> ema = EMA(model, decay=0.9999)
        >>> # During training loop
        >>> for data in dataloader:
        >>>     # ... training steps ...
        >>>     ema.update()  # Update EMA after each training step
        """
        for name, param in self.model.state_dict().items():
            self.shadow[name].mul_(self.decay)
            self.shadow[name].add_(param, alpha=1 - self.decay)

    def apply_shadow(self) -> None:
        """
        Apply the EMA parameters to the model, replacing the current model parameters with the EMA parameters

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> model = UNet(time_emb_dim=128, image_channels=3, image_size=32)
        >>> ema = EMA(model, decay=0.9999)
        >>> # After training is complete
        >>> ema.apply_shadow()  # Apply EMA parameters to the model for evaluation or inference
        """
        self.backup = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow)

    def restore(self) -> None:
        """
        Restore the original model parameters, replacing the EMA parameters with the original model parameters

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> model = UNet(time_emb_dim=128, image_channels=3, image_size=32)
        >>> ema = EMA(model, decay=0.9999)
        >>> # After applying EMA parameters for evaluation
        >>> ema.restore()  # Restore original model parameters for further training or evaluation
        """
        self.model.load_state_dict(self.backup)


# =======================================
# Diffusion Model Class
# =======================================
class DiffusionModel(nn.Module):
    def __init__(self, 
        cfg: DiffusionConfig, 
        device: str = "cpu"
    ) -> None:
        """ 
        Diffusion model class that encapsulates the U-Net architecture, noise scheduler, optimizer, and training loop for diffusion-based generative modeling
        It also supports optional latent diffusion using an autoencoder and conditional generation with class embeddings.
        It's fully configurable through the DiffusionConfig class, allowing for easy experimentation with different architectures and training settings.

        Parameters:
        -----------
        cfg: DiffusionConfig
            Configuration object containing hyperparameters and settings for the diffusion model
        device: str
            Device to run the model on (default: "cpu")

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> cfg = DiffusionConfig()
        >>> model = DiffusionModel(cfg, device="cpu")
        >>> model.fit(dataloader)  # Train the model using a dataloader
        >>> samples = model.sample(num_samples=16)  # Generate samples from the trained model
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        # Noise scheduler for the diffusion process
        self.scheduler = NoiseScheduler(
            timesteps=cfg.timesteps,
            beta_schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end, 
            s=cfg.s,
            device=self.device
        )

        # Latent autoencoder for latent diffusion (optional) 
        self.ae = None
        if self.cfg.use_latent_diffusion:
            self.ae = LatentAutoEncoder(
                in_c=cfg.image_channels,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.latent_hidden_dim,
                kernel_size=cfg.latent_kernel_size,
                stride=cfg.latent_stride,
                padding=cfg.latent_padding, 
                scale_factor=cfg.latent_scale_factor
            ).to(self.device)

            for p in self.ae.parameters():
                p.requires_grad = False

            dummy = torch.zeros(1, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size).to(self.device)
            with torch.no_grad():
                latent = self.ae.encode(dummy)
            self.latent_c, self.latent_h, self.latent_w = latent.shape[1:]

        # Neural network model (U-Net or CNN) for predicting noise in the diffusion process
        if self.cfg.model_type == "cnn":
            self.model = CNN(
                input_channels=cfg.image_channels,
                output_channels=cfg.image_channels,
                hidden_dims=[cfg.base_channels * m for m in cfg.channel_mults],
                time_emb_dim=cfg.time_emb_dim,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                padding=cfg.padding,
                use_batch_norm=cfg.use_batch_norm
            ).to(self.device)
        elif self.cfg.model_type in ["conv_unet", "res_unet"]:
            self.model = UNet(
                time_emb_dim=cfg.time_emb_dim,
                image_channels=self.latent_c if self.cfg.use_latent_diffusion else cfg.image_channels,
                image_size=self.latent_h if self.cfg.use_latent_diffusion else cfg.image_size,
                time_width_coef=cfg.time_width_coef,
                base_channels=cfg.base_channels,
                channel_mults=cfg.channel_mults,
                dropout=cfg.dropout,
                kernel_size=cfg.kernel_size,
                stride=cfg.stride,
                padding=cfg.padding,
                num_groups=cfg.num_groups,
                eps_groupnorm=cfg.eps_groupnorm,
                use_attention=cfg.use_attention, 
                num_heads=cfg.num_heads,
                attention_resolutions=cfg.attention_resolutions, 
                down_kernel_size=cfg.down_kernel_size,
                down_stride=cfg.down_stride,
                down_padding=cfg.down_padding,  
                down_num_res_blocks=cfg.down_num_res_blocks,
                up_kernel_size=cfg.up_kernel_size,
                up_stride=cfg.up_stride,
                up_padding=cfg.up_padding,
                up_num_res_blocks=cfg.up_num_res_blocks
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.cfg.model_type}")

        if cfg.use_torch_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode=cfg.compile_mode)

        # Conditional embedding for class labels (optional) if num_classes is specified in the configuration
        self.cond_embed = None
        if self.cfg.num_classes is not None:
            self.cond_embed = nn.Embedding(self.cfg.num_classes, self.cfg.time_emb_dim).to(self.device)

        params = list(self.model.parameters())
        if self.cond_embed is not None:
            params += list(self.cond_embed.parameters())

        # Optimizer for training the model
        self.opt = torch.optim.Adam(params, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)

        # Exponential Moving Average (EMA) for model parameters (optional)
        self.ema = EMA(self.model, decay=cfg.ema_decay) if cfg.use_ema else None

    # ======================
    # Forward process (diffusion)
    # ======================
    def forward_diffusion(self,
        x0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process that adds noise to the input images x0 at timestep t, according to the noise schedule defined in the NoiseScheduler

        Parameters:
        -----------
        x0: torch.Tensor
            Original input images (x_0) at timestep 0
        t: torch.Tensor
            Timesteps for each image in the batch, indicating how much noise to add

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            Noisy images (x_t) at timestep t and the noise added to the original images

        Usage Example:
        --------------
        >>> model = DiffusionModel(cfg, device="cpu")
        >>> x0 = torch.randn(16, 3, 32, 32).to("cpu")  # Original images
        >>> t = torch.randint(0, cfg.timesteps, (16,), device="cpu")  # Random timesteps
        >>> xt, noise = model.forward_diffusion(x0, t)  # Forward diffusion process
        """
        noise = torch.randn_like(x0)
        xt = self.scheduler.q_sample(x0, t, noise)
        return xt, noise

    # ======================
    # Loss
    # ======================
    def diffusion_loss(self, 
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the diffusion loss between the predicted noise and the actual noise added
        for a batch of input images x0 and optional conditioning information y

        Parameters:
        -----------
        x0: torch.Tensor
            Original input images (x_0) at timestep 0
        y: Optional[torch.Tensor]
            Optional conditioning information (e.g. class labels) for conditional generation

        Returns:
        --------
        torch.Tensor
            Loss value computed as either L1 or MSE loss between predicted and actual noise

        Usage Example:
        --------------
        >>> model = DiffusionModel(cfg, device="cpu")
        >>> x0 = torch.randn(16, 3, 32, 32).to("cpu")  # Original images
        >>> y = torch.randint(0, cfg.num_classes, (16,), device="cpu")  # Optional class labels
        >>> loss = model.diffusion_loss(x0, y)  # Compute diffusion loss
        """
        B = x0.size(0)
        t = torch.randint(0, self.cfg.timesteps, (B,), device=self.device)

        xt, noise = self.forward_diffusion(x0, t)

        cond = None
        if y is not None:
            cond = self.cond_embed(y) 

        if self.cfg.cond_drop_prob > 0 and cond is not None:
            drop_mask = torch.rand(B, device=self.device) < self.cfg.cond_drop_prob
            cond = cond.clone()
            cond[drop_mask] = 0.0 # unconditional for dropped samples

        if self.cfg.num_classes is not None and cond is not None:
            pred = self.model(xt, t, cond)
        else:
            pred = self.model(xt, t)

        if self.cfg.loss == "l1":
            return F.l1_loss(pred, noise)
        return F.mse_loss(pred, noise) # default to MSE loss

    # ======================
    # Train step
    # ======================
    def train_step(self, 
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> float:
        """
        Perform a single training step for the diffusion model, including forward pass, loss computation, backpropagation, and optimizer step

        Parameters:
        -----------
        x: torch.Tensor
            Input images (x_0) at timestep 0
        y: Optional[torch.Tensor]
            Optional conditioning information (e.g. class labels) for conditional generation

        Returns:
        --------
        float
            Loss value for the current training step

        Usage Example:
        --------------
        >>> model = DiffusionModel(cfg, device="cpu")
        >>> x = torch.randn(16, 3, 32, 32).to("cpu")  # Input images
        >>> y = torch.randint(0, cfg.num_classes, (16,), device="cpu")  # Optional class labels
        >>> loss = model.train_step(x, y)  # Perform a training step
        """
        x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)

        if self.cfg.use_latent_diffusion:
            with torch.no_grad():
                x = self.ae.encode(x)
        if self.cfg.num_classes is not None and y is not None:
            loss = self.diffusion_loss(x, y)
        else:
            loss = self.diffusion_loss(x)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.ema is not None:
            self.ema.update()

        return loss.item()

    # ======================
    # Training loop
    # ======================
    def fit(self, 
        dataloader: DataLoader, 
        epochs: int
    ) -> None:
        """
        Train the diffusion model for a specified number of epochs using the provided dataloader

        Parameters:
        -----------
        dataloader: DataLoader
            PyTorch DataLoader providing batches of training data
        epochs: int
            Number of epochs to train the model

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> model = DiffusionModel(cfg, device="cpu")
        >>> model.fit(dataloader)  # Train the model using a dataloader
        """
        for epoch in range(epochs):
            total_loss = 0

            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        x, y = batch
                    else:
                        x, y = batch[0], None
                else:
                    x, y = batch, None

                total_loss += self.train_step(x, y)

            print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")

    # ======================
    # Sampling (reverse process, denoising)
    # ======================
    @torch.no_grad()
    def sample(self, 
        n: int,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples from the trained diffusion model using the reverse diffusion process (denoising), optionally conditioned on class embeddings

        Parameters:
        -----------
        n: int
            Number of samples to generate
        cond: Optional[torch.Tensor]
            Optional conditioning information (e.g. class labels) for conditional generation

        Returns:
        --------
        torch.Tensor
            Generated samples (images) from the diffusion model

        Usage Example:
        --------------
        >>> model = DiffusionModel(cfg, device="cpu")
        >>> samples = model.sample(num_samples=16)  # Generate samples from the trained model
        """
        if self.ema is not None:
            self.ema.apply_shadow()

        if self.cfg.use_latent_diffusion:
            x = torch.randn(n, self.latent_c, self.latent_h, self.latent_w).to(self.device)
        else:
            x = torch.randn(n, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size).to(self.device)

        steps = self.cfg.ddim_steps if self.cfg.use_ddim else self.cfg.timesteps
        indices = torch.linspace(0, self.cfg.timesteps - 1, steps).long()

        for i in reversed(indices):
            t = torch.full((n,), i, device=self.device)

            if self.cfg.guidance_scale > 0 and self.cfg.num_classes is not None and cond is not None:
                cond_emb = self.cond_embed(cond) if cond is not None else None
                eps_cond = self.model(x, t, cond_emb)
                eps_uncond = self.model(x, t, None)
                eps = eps_uncond + self.cfg.guidance_scale * (eps_cond - eps_uncond)
            else: 
                eps = self.model(x, t)

            if self.cfg.use_ddim: # DDIM sampling
                alpha_bar_prev = self.scheduler.alpha_bar[indices[indices < i].max()] if i > 0 else torch.tensor(1.0).to(self.device)
                sigma = 0.0 # deterministic sampling
                x0_pred = (x - self.scheduler.sqrt_one_minus_alpha_bar[i] * eps) / self.scheduler.sqrt_alpha_bar[i]
                dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
                if sigma > 0:
                    noise = torch.randn_like(x)
                    x += sigma * noise

            else: # DDPM sampling
                x = (1 / self.scheduler.sqrt_alpha_bar[i]) * (x - ((1 - self.scheduler.sqrt_alpha_bar[i]) / self.scheduler.sqrt_one_minus_alpha_bar[i]) * eps)
                if i > 0:
                    noise = torch.randn_like(x)
                    beta = self.scheduler.betas[i].to(self.device)
                    x += torch.sqrt(beta) * noise

        if self.ema is not None:
            self.ema.restore()

        if self.cfg.use_latent_diffusion:
            x = self.ae.decode(x)

        return x.clamp(-1, 1).cpu()
    
    def save(
        self, 
        path: str, 
        print_message: bool = False
    ) -> None:
        """
        Save the DiffusionModel's state_dict to a file

        Parameters:
        -----------
        path: str
            File path to save the model state_dict
        print_message: bool
            Whether to print a confirmation message after saving (default: False)

        Returns:
        --------
        None

        Usage Example:
        --------------
        >>> model.save("models/diffusion_model.pth")  
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        
        payload = {
            "cfg": self.cfg,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "ema_state_dict": self.ema.shadow if self.ema is not None else None,
            "ae_state_dict": self.ae.state_dict() if self.ae is not None else None,
            "cond_embed_state_dict": self.cond_embed.state_dict() if self.cond_embed is not None else None
        }
        torch.save(payload, path)
        if print_message:
            print(f"Model saved to {path}")

    def load(
        self,
        path: str,
        print_message: bool = False
    ) -> None:
        """
        Load the DiffusionModel's state from a file

        Parameters:
        -----------
        path: str
            File path to load the model state from (e.g. "models/diffusion_model.pth")
        print_message: bool
            Whether to print a confirmation message after loading (default: False)

        Returns:
        --------
        None

        Usage Example:
        >>> model.load("models/diffusion_model.pth")
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        
        state_dict = torch.load(path, map_location=self.device)
        self.cfg = state_dict["cfg"]
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.opt.load_state_dict(state_dict["optimizer_state_dict"])
        if self.ema is not None and state_dict["ema_state_dict"] is not None:
            self.ema.shadow = state_dict["ema_state_dict"]
        if self.ae is not None and state_dict["ae_state_dict"] is not None:
            self.ae.load_state_dict(state_dict["ae_state_dict"])
        if self.cond_embed is not None and state_dict["cond_embed_state_dict"] is not None:
            self.cond_embed.load_state_dict(state_dict["cond_embed_state_dict"])
        if print_message:
            print(f"Model loaded from {path}")


# === FILE: NRT/NRT_diffusion_models/test.py ===