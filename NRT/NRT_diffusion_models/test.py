# ===========================
# Path setup
# ===========================
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ===========================
# Imports
# ===========================
from src.data.load import load_cifar10
from src.data.utils import plot_images
from src.models.diffusion_models import (
    DiffusionConfig, 
    NoiseScheduler, 
    TimeEmbedding, 
    CNN, 
    ResBlock, 
    AttentionBlock,
    DownBlock, 
    UpBlock,
    UNet, 
    LatentAutoEncoder,
    EMA,
    DiffusionModel
)


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MNIST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mnist")
os.makedirs(MNIST_OUTPUT_DIR, exist_ok=True)
CIFAR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cifar10")
os.makedirs(CIFAR_OUTPUT_DIR, exist_ok=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def print_section(title):
    print("\n\n")
    print("=" * 50)
    print(title)
    print("=" * 50)

def print_subsection(title):
    print("-" * 50)
    print(title)
    print("-" * 50)

def clear_data_dir():
    os.system(f"rm -rf {DATA_DIR}/*")
    os.system(f"rm -rf {DATA_DIR}")


# ===========================
# Test Noise Scheduler
# ===========================
def test_noise_scheduler():
    print_section("Test Noise Scheduler")
    config = DiffusionConfig()
    scheduler = NoiseScheduler(
        timesteps=config.timesteps, 
        beta_schedule=config.beta_schedule,
        beta_start=config.beta_start,
        beta_end=config.beta_end, 
        s=config.s, 
        device="cpu"
    )
    print(f"Noise Scheduler: {scheduler}")
    print(f"Betas: {scheduler.betas[:10]} --> {scheduler.betas[-10:]}")
    print(f"Alphas: {scheduler.alphas[:10]} --> {scheduler.alphas[-10:]}")
    print(f"Alphas cumprod: {scheduler.alpha_bar[:10]} --> {scheduler.alpha_bar[-10:]}")
    alpha_bar_prev = torch.cat([torch.tensor([1.0]), scheduler.alpha_bar[:-1]])
    print(f"Alphas cumprod prev: {alpha_bar_prev[:10]} --> {alpha_bar_prev[-10:]}")
    posterior_variance = scheduler.betas * (1 - alpha_bar_prev) / (1 - scheduler.alpha_bar)
    print(f"Posterior variance: {posterior_variance[:10]} --> {posterior_variance[-10:]}")


# ===========================
# Test Time Embedding
# ===========================
def test_time_embedding():
    print_section("Test Time Embedding")
    embedding_dim = 128
    time_embedding = TimeEmbedding(dim=embedding_dim, time_width_coef=4)
    print(f"Time Embedding: {time_embedding}")
    t = torch.arange(0, 1000).unsqueeze(1)  # shape (1000, 1)
    embeddings = time_embedding(t)  # shape (1000, embedding_dim)
    print(f"Embeddings shape: {embeddings.shape}")
    plt.figure(figsize=(10, 5))
    plt.imshow(embeddings.squeeze().detach().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Time Embeddings")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Timestep")
    plt.savefig(os.path.join(OUTPUT_DIR, "time_embeddings.png"))
    plt.close()


# ===========================
# Test CNN
# ===========================
def test_cnn():
    print_section("Test CNN")
    model = CNN(
        input_channels=1,
        output_channels=1,
        hidden_dims=[32, 64, 128],
        time_emb_dim=128, 
        kernel_size=3,  
        stride=1,
        padding=1,
        use_batch_norm=True
    )
    print(f"CNN: {model}")
    x = torch.randn(4, 1, 28, 28)  # batch of 4 MNIST images
    t_emb = torch.randn(4, 128)  # batch of 4 time embeddings
    output = model(x, t_emb)
    print(f"Output shape: {output.shape}")


# ===========================
# Test ResBlock / AttentionBlock / DownBlock / UpBlock / UNet
# ===========================
def test_unet():
    print_section("Test UNet and all its components")
    print_subsection("Test ResBlock")
    res_block = ResBlock(
        in_c=32, 
        out_c=64,
        time_dim=128,
        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1, 
        num_groups=8
    )
    print(f"ResBlock: {res_block}")
    x = torch.randn(4, 32, 28, 28)  # batch of 4 feature maps
    t_emb = torch.randn(4, 128)
    output = res_block(x, t_emb)
    print(f"ResBlock output shape: {output.shape}") # expected: (4, 64, 28, 28)

    print_subsection("Test AttentionBlock")
    attn_block = AttentionBlock(
        channels=64,
        num_heads=4,
        num_groups=8
    )
    print(f"AttentionBlock: {attn_block}")
    x = torch.randn(4, 64, 28, 28)
    output = attn_block(x)
    print(f"AttentionBlock output shape: {output.shape}") # expected: (4, 64, 28, 28)

    print_subsection("Test DownBlock")
    down_block = DownBlock(
        in_c=64,
        out_c=128,
        time_dim=128,
        use_attn=True,
        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        num_heads=4,
        num_res_blocks=1, 
        down_kernel_size=4,
        down_stride=2,
        down_padding=1
    )
    print(f"DownBlock: {down_block}")
    x = torch.randn(4, 64, 28, 28) # input from previous DownBlock or initial input (let's say input size is 28x28)
    t_emb = torch.randn(4, 128)
    output = down_block(x, t_emb)
    x_out, skip = output # unpack the tuple returned by DownBlock
    print(f"DownBlock output shape: {x_out.shape}") # expected: (4, 128, 14, 14) (let's say bottleneck size is 14x14)
    print(f"DownBlock skip connection shape: {skip.shape}") # expected: (4, 128, 28, 28) (skip connection for UpBlock)

    print_subsection("Test UpBlock")
    up_block = UpBlock(
        in_c=128,
        skip_c=128,
        out_c=64,
        time_dim=128,
        use_attn=True,
        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        num_heads=4,
        num_res_blocks=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1
    )
    print(f"UpBlock: {up_block}")
    x = torch.randn(4, 128, 14, 14) # input from previous UpBlock or bottleneck (bottleneck size is 14x14)
    skip = torch.randn(4, 128, 28, 28) # skip connection from DownBlock
    t_emb = torch.randn(4, 128)
    output = up_block(x, skip, t_emb)  
    print(f"UpBlock output shape: {output.shape}") # expected: (4, 64, 28, 28)

    print_subsection("Test UNet (without conditional input)")
    unet = UNet(
        time_emb_dim=128,
        image_channels=1,
        image_size=28,
        time_width_coef=4,
        base_channels=32,
        channel_mults=[1, 2],
        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        num_res_blocks_down=1,
        num_res_blocks_up=1,
        use_attention=True,
        num_heads=4,
        attention_resolutions=(8,),
        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1    
    )
    print(f"UNet: {unet}")
    x = torch.randn(4, 1, 28, 28) # batch of 4 MNIST images
    t_emb = torch.randint(0, 1000, (4,))  # random timesteps for each image
    output = unet(x, t_emb)
    print(f"UNet output shape: {output.shape}") # expected: (4, 1, 28, 28)
    
    print_subsection("Test UNet (with conditional input)")
    unet_cond = UNet(
        time_emb_dim=128,
        image_channels=1,
        image_size=28,
        time_width_coef=4,
        base_channels=32,
        channel_mults=[1, 2],
        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        num_res_blocks_down=1,
        num_res_blocks_up=1,
        use_attention=True,
        num_heads=4,
        attention_resolutions=(8,),
        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1
    )
    print(f"UNet with conditional input: {unet_cond}")
    x = torch.randn(4, 1, 28, 28) # batch of 4 MNIST images
    t_emb = torch.randint(0, 1000, (4,))  # random timesteps for each image
    cond_embed = nn.Embedding(10, 128)  # embedding for 10 classes (MNIST digits)
    y = torch.randint(0, 10, (4,))  # random labels for the batch
    cond = cond_embed(y)  # get the conditional embeddings
    output = unet_cond(x, t_emb, cond)
    print(f"UNet with conditional input output shape: {output.shape}") # expected: (4, 1, 28, 28)


# ===========================
# Test LatentAutoEncoder
# ===========================
def test_latent_autoencoder():
    print_section("Test LatentAutoEncoder")
    latent_autoencoder = LatentAutoEncoder(
        in_c=1,
        latent_dim=16,
        hidden_dim=32,
        kernel_size=4,
        stride=2,
        padding=1,
        scale_factor=0.18215
    )
    print(f"LatentAutoEncoder: {latent_autoencoder}")
    x = torch.randn(4, 1, 28, 28) # batch of 4 MNIST images
    z = latent_autoencoder.encode(x)
    print(f"Encoded latent shape: {z.shape}") # expected: (4, 16, 7, 7)
    x_recon = latent_autoencoder.decode(z)
    print(f"Reconstructed image shape: {x_recon.shape}") # expected: (4, 1, 28, 28)


# ===========================
# Test EMA
# ===========================
def test_ema():
    print_section("Test EMA")
    model = nn.Linear(10, 1)
    ema = EMA(model, decay=0.99)
    print(f"EMA: {ema}")
    x = torch.randn(4, 10)
    y = model(x)
    print(f"Model output: {y}")
    ema.update()
    print(f"EMA updated.")


# ===========================
# Test DiffusionModel on MNIST with CNN
# ===========================


# ===========================
# Test DiffusionModel on MNIST with ResUNet
# ===========================


# ===========================
# Test DiffusionModel on MNIST with Latent Diffusion DDPM
# ===========================


# ===========================
# Test DiffusionModel on CIFAR-10 with ResUNet DDPM
# ===========================


# ==========================
# Test DiffusionModel on CIFAR-10 with ResUNet DDIM
# ==========================


# =========================
# Test DiffusionModel on CIFAR-10 with Conditional ResUNet DDIM Ema
# =========================


if __name__ == "__main__":
    test_noise_scheduler()
    test_time_embedding()
    test_cnn()
    test_unet()
    test_latent_autoencoder()
    test_ema()
    clear_data_dir()