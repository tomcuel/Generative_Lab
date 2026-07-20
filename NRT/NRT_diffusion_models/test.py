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
from src.data.load import (
    load_mnist,
    load_cifar10
)
from src.data.utils import (
    set_seed, 
    plot_images
)
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


RANDOM_SEED = 42

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
def test_diffusion_model_mnist_cnn():
    """
    Test the DiffusionModel on MNIST dataset using a simple CNN architecture.

    This function initializes the DiffusionModel with a CNN backbone, 
    trains it on the MNIST dataset to see that everything is working as expected, 
    and generates sample images after training (quick training for demonstration purposes, so no assertions on quality).
    """
    print_section("Test DiffusionModel on MNIST with CNN")
    print_subsection("Load MNIST dataset")
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=64, train=True, flatten=False)
    # only keep a small portion 
    subset_size = 1000
    mnist_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_loader.dataset, range(subset_size)), batch_size=64, shuffle=True)

    # Plot real data distribution
    first_numbers = next(iter(mnist_loader))[0][:5]  # Get first 5 samples of the first batch
    print(first_numbers.shape)
    plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR, "mnist_real.png"), title="Real MNIST Samples")

    print_subsection("Initialize DiffusionModel with CNN")
    config = DiffusionConfig(
        # model 
        model_type="cnn",
        loss="mse",

        num_classes=None,  # No conditional input for this test

        image_size=28,
        image_channels=1,

        base_channels=16,
        channel_mults=[1, 2],

        time_emb_dim=64,
        time_width_coef=4,

        # convolutional tuning
        use_attention=False,

        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        use_batch_norm=False,

        # diffusion
        timesteps=50,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        s=0.008,

        # training
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0,
        batch_size=64,

        # sampling
        use_ddim=False,
        use_ema=False,

        # latent diffusion
        use_latent_diffusion=False
    )
    print("DiffusionModel Configuration (selected params):")
    selected_params = {
        "model_type": config.model_type,
        "loss": config.loss,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "image_channels": config.image_channels,
        "base_channels": config.base_channels,
        "channel_mults": config.channel_mults,
        "time_emb_dim": config.time_emb_dim,
        "time_width_coef": config.time_width_coef,
        "use_attention": config.use_attention,
        "dropout": config.dropout,
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "padding": config.padding,
        "use_batch_norm": config.use_batch_norm,
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "s": config.s,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "use_ddim": config.use_ddim,
        "use_ema": config.use_ema,
        "use_latent_diffusion": config.use_latent_diffusion,
    }
    for key, value in selected_params.items():
        print(f"{key}: {value}")
    print()

    # Initialize the DiffusionModel
    diffusion_model = DiffusionModel(config, device="cpu")
    print("DiffusionModel Architecture:")
    print(diffusion_model)

    # Train the model on MNIST
    print_subsection("Train DiffusionModel on MNIST")
    epochs = 2 # for demonstration purposes, see if all is working fine
    diffusion_model.fit(mnist_loader, epochs)

    # Sample images from the trained model
    print_subsection("Sample images from trained DiffusionModel")
    n_samples = 5
    sampled_images = diffusion_model.sample(n_samples)
    plot_images(sampled_images, n_samples, save_path=os.path.join(OUTPUT_DIR, "test_diffusion_model_mnist_cnn.png"), title="Sampled MNIST Images")


# ===========================
# Test DiffusionModel on MNIST with ResUNet (without attention) + l1 loss
# ===========================
def test_diffusion_model_mnist_resunet():
    """
    Test the DiffusionModel on MNIST dataset using a ResUNet architecture without attention and with l1 loss.

    This function initializes the DiffusionModel with a ResUNet backbone, 
    trains it on the MNIST dataset to see that everything is working as expected, 
    and generates sample images after training (quick training for demonstration purposes, so no assertions on quality).
    """
    print_section("Test DiffusionModel on MNIST with ResUNet")
    print_subsection("Load MNIST dataset")
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=64, train=True, flatten=False)
    # only keep a small portion 
    subset_size = 1000
    mnist_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_loader.dataset, range(subset_size)), batch_size=64, shuffle=True)

    # Plot real data distribution
    first_numbers = next(iter(mnist_loader))[0][:5]  # Get first 5 samples of the first batch
    print(first_numbers.shape)
    plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR, "mnist_real.png"), title="Real MNIST Samples")

    print_subsection("Initialize DiffusionModel with ResUNet")
    config = DiffusionConfig(
        # model
        model_type="res_unet",
        loss="l1",

        num_classes=None,  # No conditional input for this test

        image_size=28,
        image_channels=1,

        base_channels=32,
        channel_mults=[1, 2],

        time_emb_dim=128,
        time_width_coef=4,

        # convolutional tuning
        use_attention=False,

        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        eps_groupnorm=1e-5,

        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        down_num_res_blocks=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1,
        up_num_res_blocks=1,

        # diffusion
        timesteps=50,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        s=0.008,

        # training
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0, 
        batch_size=64,

        # sampling
        use_ddim=False,
        use_ema=False,

        # latent diffusion
        use_latent_diffusion=False
    )
    print("DiffusionModel Configuration (selected params):")
    selected_params = {
        "model_type": config.model_type,
        "loss": config.loss,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "image_channels": config.image_channels,
        "base_channels": config.base_channels,
        "channel_mults": config.channel_mults,
        "time_emb_dim": config.time_emb_dim,
        "time_width_coef": config.time_width_coef,
        "use_attention": config.use_attention,
        "dropout": config.dropout, 
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "padding": config.padding,
        "num_groups": config.num_groups, 
        "eps_groupnorm": config.eps_groupnorm,
        "down_kernel_size": config.down_kernel_size, 
        "down_stride": config.down_stride, 
        "down_padding": config.down_padding, 
        "down_num_res_blocks": config.down_num_res_blocks, 
        "up_kernel_size": config.up_kernel_size, 
        "up_stride": config.up_stride, 
        "up_padding": config.up_padding, 
        "up_num_res_blocks": config.up_num_res_blocks, 
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "s": config.s,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "use_ddim": config.use_ddim,
        "use_ema": config.use_ema,
        "use_latent_diffusion": config.use_latent_diffusion,
    }
    for key, value in selected_params.items():
        print(f"{key}: {value}")
    print()

    # Initialize the DiffusionModel
    diffusion_model = DiffusionModel(config, device="cpu")
    print("DiffusionModel Architecture:")
    print(diffusion_model)

    # Train the model on MNIST
    print_subsection("Train DiffusionModel on MNIST")
    epochs = 2 # for demonstration purposes, see if all is working fine
    diffusion_model.fit(mnist_loader, epochs)

    # Sample images from the trained model
    print_subsection("Sample images from trained DiffusionModel")
    n_samples = 5
    sampled_images = diffusion_model.sample(n_samples)
    plot_images(sampled_images, n_samples, save_path=os.path.join(OUTPUT_DIR, "test_diffusion_model_mnist_resunet.png"), title="Sampled MNIST Images")


# ===========================
# Test DiffusionModel on MNIST with ResUNet (with attention) + DDIM sampling
# ===========================
def test_diffusion_model_mnist_resunet_attention_ddim():
    """
    Test the DiffusionModel on MNIST dataset using a ResUNet architecture with attention and DDIM sampling.

    This function initializes the DiffusionModel with a ResUNet backbone, 
    trains it on the MNIST dataset to see that everything is working as expected, 
    and generates sample images after training (quick training for demonstration purposes, so no assertions on quality).
    """
    print_section("Test DiffusionModel on MNIST with ResUNet + Attention + DDIM")
    print_subsection("Load MNIST dataset")
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=64, train=True, flatten=False)
    # only keep a small portion 
    subset_size = 1000
    mnist_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_loader.dataset, range(subset_size)), batch_size=64, shuffle=True)

    # Plot real data distribution
    first_numbers = next(iter(mnist_loader))[0][:5]  # Get first 5 samples of the first batch
    print(first_numbers.shape)
    plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR, "mnist_real.png"), title="Real MNIST Samples")

    print_subsection("Initialize DiffusionModel with ResUNet + Attention + DDIM")
    config = DiffusionConfig(
        # model
        model_type="res_unet",
        loss="l1",

        num_classes=None,  # No conditional input for this test

        image_size=28,
        image_channels=1,

        base_channels=32,
        channel_mults=[1, 2],

        time_emb_dim=128,
        time_width_coef=4,

        # convolutional tuning
        use_attention=True,
        attention_resolutions=(14,), # should be among the image_size / 2**i
        num_heads=4,

        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        eps_groupnorm=1e-5,

        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        down_num_res_blocks=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1,
        up_num_res_blocks=1,

        # diffusion
        timesteps=50,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        s=0.008,

        # training
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0, 
        batch_size=64,

        # sampling
        use_ddim=True,
        ddim_steps=25,
        use_ema=False,

        # latent diffusion
        use_latent_diffusion=False
    )
    print("DiffusionModel Configuration (selected params):")
    selected_params = {
        "model_type": config.model_type,
        "loss": config.loss,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "image_channels": config.image_channels,
        "base_channels": config.base_channels,
        "channel_mults": config.channel_mults,
        "time_emb_dim": config.time_emb_dim,
        "time_width_coef": config.time_width_coef,
        "use_attention": config.use_attention,
        "attention_resolutions": config.attention_resolutions,
        "num_heads": config.num_heads,
        "dropout": config.dropout, 
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "padding": config.padding,
        "num_groups": config.num_groups, 
        "eps_groupnorm": config.eps_groupnorm,
        "down_kernel_size": config.down_kernel_size, 
        "down_stride": config.down_stride, 
        "down_padding": config.down_padding, 
        "down_num_res_blocks": config.down_num_res_blocks, 
        "up_kernel_size": config.up_kernel_size, 
        "up_stride": config.up_stride, 
        "up_padding": config.up_padding, 
        "up_num_res_blocks": config.up_num_res_blocks, 
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "s": config.s,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "use_ddim": config.use_ddim,
        "ddim_steps": config.ddim_steps,
        "use_ema": config.use_ema,
        "use_latent_diffusion": config.use_latent_diffusion,
    }
    for key, value in selected_params.items():
        print(f"{key}: {value}")
    print()

    # Initialize the DiffusionModel
    diffusion_model = DiffusionModel(config, device="cpu")
    print("DiffusionModel Architecture:")
    print(diffusion_model)

    # Train the model on MNIST
    print_subsection("Train DiffusionModel on MNIST")
    epochs = 2 # for demonstration purposes, see if all is working fine
    diffusion_model.fit(mnist_loader, epochs)

    # Sample images from the trained model
    print_subsection("Sample images from trained DiffusionModel")
    n_samples = 5
    sampled_images = diffusion_model.sample(n_samples)
    plot_images(sampled_images, n_samples, save_path=os.path.join(OUTPUT_DIR, "test_diffusion_model_mnist_resunet_attention_ddim.png"), title="Sampled MNIST Images")


# ===========================
# Test DiffusionModel on MNIST with ResUNet + EMA sampling
# ===========================
def test_diffusion_model_mnist_resunet_ema():
    """
    Test the DiffusionModel on MNIST dataset using a ResUNet architecture with EMA sampling.

    This function initializes the DiffusionModel with a ResUNet backbone, 
    trains it on the MNIST dataset to see that everything is working as expected, 
    and generates sample images after training (quick training for demonstration purposes, so no assertions on quality).
    """
    print_section("Test DiffusionModel on MNIST with ResUNet + Attention + EMA")
    print_subsection("Load MNIST dataset")
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=64, train=True, flatten=False)
    # only keep a small portion 
    subset_size = 1000
    mnist_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_loader.dataset, range(subset_size)), batch_size=64, shuffle=True)

    # Plot real data distribution
    first_numbers = next(iter(mnist_loader))[0][:5]  # Get first 5 samples of the first batch
    print(first_numbers.shape)
    plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR, "mnist_real.png"), title="Real MNIST Samples")

    print_subsection("Initialize DiffusionModel with ResUNet")
    config = DiffusionConfig(
        # model
        model_type="res_unet",
        loss="mse",

        num_classes=None,  # No conditional input for this test

        image_size=28,
        image_channels=1,

        base_channels=32,
        channel_mults=[1, 2],

        time_emb_dim=128,
        time_width_coef=4,

        # convolutional tuning
        use_attention=False,

        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        eps_groupnorm=1e-5,

        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        down_num_res_blocks=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1,
        up_num_res_blocks=1,

        # diffusion
        timesteps=50,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        s=0.008,

        # training
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0, 
        batch_size=64,

        # sampling
        use_ddim=False,
        use_ema=True,
        ema_decay=0.9999,

        # latent diffusion
        use_latent_diffusion=False
    )
    print("DiffusionModel Configuration (selected params):")
    selected_params = {
        "model_type": config.model_type,
        "loss": config.loss,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "image_channels": config.image_channels,
        "base_channels": config.base_channels,
        "channel_mults": config.channel_mults,
        "time_emb_dim": config.time_emb_dim,
        "time_width_coef": config.time_width_coef,
        "use_attention": config.use_attention,
        "dropout": config.dropout, 
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "padding": config.padding,
        "num_groups": config.num_groups, 
        "eps_groupnorm": config.eps_groupnorm,
        "down_kernel_size": config.down_kernel_size, 
        "down_stride": config.down_stride, 
        "down_padding": config.down_padding, 
        "down_num_res_blocks": config.down_num_res_blocks, 
        "up_kernel_size": config.up_kernel_size, 
        "up_stride": config.up_stride, 
        "up_padding": config.up_padding, 
        "up_num_res_blocks": config.up_num_res_blocks, 
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "s": config.s,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "use_ddim": config.use_ddim,
        "use_ema": config.use_ema,
        "ema_decay": config.ema_decay,
        "use_latent_diffusion": config.use_latent_diffusion,
    }
    for key, value in selected_params.items():
        print(f"{key}: {value}")
    print()

    # Initialize the DiffusionModel
    diffusion_model = DiffusionModel(config, device="cpu")
    print("DiffusionModel Architecture:")
    print(diffusion_model)

    # Train the model on MNIST
    print_subsection("Train DiffusionModel on MNIST")
    epochs = 2 # for demonstration purposes, see if all is working fine
    diffusion_model.fit(mnist_loader, epochs)

    # Sample images from the trained model
    print_subsection("Sample images from trained DiffusionModel")
    n_samples = 5
    sampled_images = diffusion_model.sample(n_samples)
    plot_images(sampled_images, n_samples, save_path=os.path.join(OUTPUT_DIR, "test_diffusion_model_mnist_resunet_ema.png"), title="Sampled MNIST Images")


# ===========================
# Test DiffusionModel on MNIST with ResUnet + conditional input (class labels)
# ===========================
def test_diffusion_model_mnist_resunet_conditional():
    """
    Test the DiffusionModel on MNIST dataset using a ResUNet architecture with conditional input (class labels).

    This function initializes the DiffusionModel with a ResUNet backbone, 
    trains it on the MNIST dataset to see that everything is working as expected, 
    and generates sample images after training (quick training for demonstration purposes, so no assertions on quality).
    """
    print_section("Test DiffusionModel on MNIST with ResUNet + Conditional Input")
    print_subsection("Load MNIST dataset")
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=64, train=True, flatten=False)
    # only keep a small portion 
    subset_size = 1000
    mnist_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_loader.dataset, range(subset_size)), batch_size=64, shuffle=True)

    # Plot real data distribution
    first_numbers = next(iter(mnist_loader))[0][:5]  # Get first 5 samples of the first batch
    print(first_numbers.shape)
    plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR, "mnist_real.png"), title="Real MNIST Samples")

    print_subsection("Initialize DiffusionModel with ResUNet + Latent Diffusion")
    config = DiffusionConfig(
        # model
        model_type="res_unet",
        loss="l1",

        num_classes=10,  # Conditional input for MNIST (10 classes)
        cond_drop_prob=0.1,  # Dropout probability for conditional input
        guidance_scale=0.9,  # Guidance scale for conditional generation

        image_size=28,
        image_channels=1,

        base_channels=32,
        channel_mults=[1, 2],

        time_emb_dim=128,
        time_width_coef=4,

        # convolutional tuning
        use_attention=False,

        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        eps_groupnorm=1e-5,

        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        down_num_res_blocks=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1,
        up_num_res_blocks=1,

        # diffusion
        timesteps=50,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        s=0.008,

        # training
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0, 
        batch_size=64,

        # sampling
        use_ddim=False,
        use_ema=False,

        # latent diffusion
        use_latent_diffusion=False 
    )
    print("DiffusionModel Configuration (selected params):")
    selected_params = {
        "model_type": config.model_type,
        "loss": config.loss,
        "num_classes": config.num_classes,
        "cond_drop_prob": config.cond_drop_prob,
        "guidance_scale": config.guidance_scale,
        "image_size": config.image_size,
        "image_channels": config.image_channels,
        "base_channels": config.base_channels,
        "channel_mults": config.channel_mults,
        "time_emb_dim": config.time_emb_dim,
        "time_width_coef": config.time_width_coef,
        "use_attention": config.use_attention,
        "dropout": config.dropout, 
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "padding": config.padding,
        "num_groups": config.num_groups, 
        "eps_groupnorm": config.eps_groupnorm,
        "down_kernel_size": config.down_kernel_size, 
        "down_stride": config.down_stride, 
        "down_padding": config.down_padding, 
        "down_num_res_blocks": config.down_num_res_blocks, 
        "up_kernel_size": config.up_kernel_size, 
        "up_stride": config.up_stride, 
        "up_padding": config.up_padding, 
        "up_num_res_blocks": config.up_num_res_blocks, 
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "s": config.s,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "use_ddim": config.use_ddim,
        "use_ema": config.use_ema,
        "use_latent_diffusion": config.use_latent_diffusion,
    }
    for key, value in selected_params.items():
        print(f"{key}: {value}")
    print()

    # Initialize the DiffusionModel
    diffusion_model = DiffusionModel(config, device="cpu")
    print("DiffusionModel Architecture:")
    print(diffusion_model)

    # Train the model on MNIST
    print_subsection("Train DiffusionModel on MNIST")
    epochs = 2 # for demonstration purposes, see if all is working fine
    diffusion_model.fit(mnist_loader, epochs)

    # Sample images from the trained model
    print_subsection("Sample images from trained DiffusionModel")
    n_samples = 5
    sample_numbers = torch.randint(0, 10, (n_samples,))  # Randomly sample class labels for conditional generation
    print(f"Sampled class labels for conditional generation: {sample_numbers}")
    sampled_images = diffusion_model.sample(n_samples, cond=sample_numbers)
    plot_images(sampled_images, n_samples, save_path=os.path.join(OUTPUT_DIR, "test_diffusion_model_mnist_resunet_conditional.png"), title="Sampled MNIST Images")


# ===========================
# Test DiffusionModel on MNIST with Latent Diffusion on ResUNet
# ===========================
def test_diffusion_model_mnist_latent_diffusion():
    """
    Test the DiffusionModel on MNIST dataset using a ResUNet architecture with Latent Diffusion.

    This function initializes the DiffusionModel with a ResUNet backbone and a Latent Autoencoder, 
    trains it on the MNIST dataset to see that everything is working as expected, 
    and generates sample images after training (quick training for demonstration purposes, so no assertions on quality).
    """
    print_section("Test DiffusionModel on MNIST with ResUNet + Latent Diffusion")
    print_subsection("Load MNIST dataset")
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=64, train=True, flatten=False)
    # only keep a small portion 
    subset_size = 1000
    mnist_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(mnist_loader.dataset, range(subset_size)), batch_size=64, shuffle=True)

    # Plot real data distribution
    first_numbers = next(iter(mnist_loader))[0][:5]  # Get first 5 samples of the first batch
    print(first_numbers.shape)
    plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR, "mnist_real.png"), title="Real MNIST Samples")

    print_subsection("Initialize DiffusionModel with ResUNet + Latent Diffusion")
    config = DiffusionConfig(
        # model
        model_type="res_unet",
        loss="l1",

        num_classes=None,  # No conditional input for this test

        image_size=28,
        image_channels=1,

        base_channels=64,
        channel_mults=[1, 2],

        time_emb_dim=128,
        time_width_coef=4,

        # convolutional tuning
        use_attention=False,

        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        eps_groupnorm=1e-5,

        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        down_num_res_blocks=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1,
        up_num_res_blocks=1,

        # diffusion
        timesteps=50,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        s=0.008,

        # training
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0, 
        batch_size=64,

        # sampling
        use_ddim=False,
        use_ema=False,

        # latent diffusion
        use_latent_diffusion=True, 
        latent_dim=16, # double conv into (16, 7, 7) latent space
        latent_hidden_dim=64,
        latent_kernel_size=4,
        latent_stride=2,
        latent_padding=1,
        latent_scale_factor=0.18215
    )
    print("DiffusionModel Configuration (selected params):")
    selected_params = {
        "model_type": config.model_type,
        "loss": config.loss,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "image_channels": config.image_channels,
        "base_channels": config.base_channels,
        "channel_mults": config.channel_mults,
        "time_emb_dim": config.time_emb_dim,
        "time_width_coef": config.time_width_coef,
        "use_attention": config.use_attention,
        "dropout": config.dropout, 
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "padding": config.padding,
        "num_groups": config.num_groups, 
        "eps_groupnorm": config.eps_groupnorm,
        "down_kernel_size": config.down_kernel_size, 
        "down_stride": config.down_stride, 
        "down_padding": config.down_padding, 
        "down_num_res_blocks": config.down_num_res_blocks, 
        "up_kernel_size": config.up_kernel_size, 
        "up_stride": config.up_stride, 
        "up_padding": config.up_padding, 
        "up_num_res_blocks": config.up_num_res_blocks, 
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "s": config.s,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "use_ddim": config.use_ddim,
        "use_ema": config.use_ema,
        "use_latent_diffusion": config.use_latent_diffusion,
        "latent_dim": config.latent_dim,
        "latent_hidden_dim": config.latent_hidden_dim,
        "latent_kernel_size": config.latent_kernel_size,
        "latent_stride": config.latent_stride,
        "latent_padding": config.latent_padding,
        "latent_scale_factor": config.latent_scale_factor
    }
    for key, value in selected_params.items():
        print(f"{key}: {value}")
    print()

    # Initialize the DiffusionModel
    diffusion_model = DiffusionModel(config, device="cpu")
    print("DiffusionModel Architecture:")
    print(diffusion_model)

    # Train the model on MNIST
    print_subsection("Train DiffusionModel on MNIST")
    epochs = 2 # for demonstration purposes, see if all is working fine
    diffusion_model.fit(mnist_loader, epochs)

    # Sample images from the trained model
    print_subsection("Sample images from trained DiffusionModel")
    n_samples = 5
    sampled_images = diffusion_model.sample(n_samples)
    plot_images(sampled_images, n_samples, save_path=os.path.join(OUTPUT_DIR, "test_diffusion_model_mnist_latent_diffusion.png"), title="Sampled MNIST Images")


# ===========================
# Test DiffusionModel on CIFAR-10 with ResUNet
# ===========================
def test_diffusion_model_cifar10_resunet():
    """
    Test the DiffusionModel on CIFAR-10 dataset using a ResUNet architecture.

    This function initializes the DiffusionModel with a ResUNet backbone, 
    trains it on the CIFAR-10 dataset to see that everything is working as expected, 
    and generates sample images after training (quick training for demonstration purposes, so no assertions on quality).
    """
    print_section("Test DiffusionModel on CIFAR-10 with ResUNet")
    print_subsection("Load CIFAR-10 dataset")
    print("Loading CIFAR-10 dataset...")
    cifar_loader = load_cifar10(batch_size=64, downsample=None, grayscale=False, normalize=True, flatten=False, train=True, subset_size=10000)

    # Plot real data distribution
    first_images = next(iter(cifar_loader))[0][:5]  # Get first 5 samples of the first batch
    print(first_images.shape) # torch.Size([5, 3, 32, 32])
    first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR, "cifar10_real.png"), title="Real CIFAR-10 Samples")

    print_subsection("Initialize DiffusionModel with ResUNet for CIFAR-10")
    config = DiffusionConfig(
        # model
        model_type="res_unet",
        loss="l1",

        num_classes=None,  # No conditional input for this test

        image_size=28,
        image_channels=3,

        base_channels=64,
        channel_mults=[1, 2],

        time_emb_dim=128,
        time_width_coef=4,

        # convolutional tuning
        use_attention=True,
        attention_resolutions=(14,), # should be among the image_size / 2**i
        num_heads=4,

        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        eps_groupnorm=1e-5,

        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        down_num_res_blocks=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1,
        up_num_res_blocks=1,

        # diffusion
        timesteps=50,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        s=0.008,

        # training
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0, 
        batch_size=64,

        # sampling
        use_ddim=False,
        use_ema=False,

        # latent diffusion
        use_latent_diffusion=False
    )
    print("DiffusionModel Configuration (selected params):")
    selected_params = {
        "model_type": config.model_type,
        "loss": config.loss,
        "num_classes": config.num_classes,
        "image_size": config.image_size,
        "image_channels": config.image_channels,
        "base_channels": config.base_channels,
        "channel_mults": config.channel_mults,
        "time_emb_dim": config.time_emb_dim,
        "time_width_coef": config.time_width_coef,
        "use_attention": config.use_attention,
        "attention_resolutions": config.attention_resolutions,
        "num_heads": config.num_heads,
        "dropout": config.dropout, 
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "padding": config.padding,
        "num_groups": config.num_groups, 
        "eps_groupnorm": config.eps_groupnorm,
        "down_kernel_size": config.down_kernel_size, 
        "down_stride": config.down_stride, 
        "down_padding": config.down_padding, 
        "down_num_res_blocks": config.down_num_res_blocks, 
        "up_kernel_size": config.up_kernel_size, 
        "up_stride": config.up_stride, 
        "up_padding": config.up_padding, 
        "up_num_res_blocks": config.up_num_res_blocks, 
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "s": config.s,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "use_ddim": config.use_ddim,
        "use_ema": config.use_ema,
        "use_latent_diffusion": config.use_latent_diffusion
    }
    for key, value in selected_params.items():
        print(f"{key}: {value}")
    print()

    # Initialize the DiffusionModel
    diffusion_model = DiffusionModel(config, device="cpu")
    print("DiffusionModel Architecture:")
    print(diffusion_model)

    # Train the model on MNIST
    print_subsection("Train DiffusionModel on CIFAR-10")
    epochs = 2 # for demonstration purposes, see if all is working fine
    diffusion_model.fit(cifar_loader, epochs)

    # Sample images from the trained model
    print_subsection("Sample images from trained DiffusionModel")
    n_samples = 5
    sampled_images = diffusion_model.sample(n_samples)
    sampled_images = (sampled_images + 1) / 2 # Convert from [-1,1] to [0,1]
    plot_images(sampled_images, n_samples, save_path=os.path.join(OUTPUT_DIR, "test_diffusion_model_cifar10_resunet.png"), title="Sampled CIFAR-10 Images")


# ==========================
# Test DiffusionModel on CIFAR-10 with Latent Diffusion + DDIM + EMA + Conditional Input
# ==========================
def test_diffusion_model_cifar10_latent_diffusion_ddim_ema_conditional():
    """
    Test the DiffusionModel on CIFAR-10 dataset using a ResUNet architecture with Latent Diffusion, DDIM sampling, EMA, and Conditional Input.

    This function initializes the DiffusionModel with a ResUNet backbone and a Latent Autoencoder, 
    trains it on the CIFAR-10 dataset to see that everything is working as expected, 
    and generates sample images after training (quick training for demonstration purposes, so no assertions on quality).
    """
    print_section("Test DiffusionModel on CIFAR-10 with ResUNet + Latent Diffusion + DDIM + EMA + Conditional Input")
    print_subsection("Load CIFAR-10 dataset")
    print("Loading CIFAR-10 dataset...")
    cifar_loader = load_cifar10(batch_size=64, downsample=None, grayscale=False, normalize=True, flatten=False, train=True, subset_size=10000)

    # Plot real data distribution
    first_images = next(iter(cifar_loader))[0][:5]  # Get first 5 samples of the first batch
    print(first_images.shape) # torch.Size([5, 3, 32, 32])
    first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR, "cifar10_real.png"), title="Real CIFAR-10 Samples")

    print_subsection("Initialize DiffusionModel with Latent Diffusion + DDIM + EMA + Conditional Input")
    config = DiffusionConfig(
        # model
        model_type="res_unet",
        loss="l1",

        num_classes=10,  # Conditional input for CIFAR-10 (10 classes)
        cond_drop_prob=0.1,
        guidance_scale=0.9,

        image_size=28,
        image_channels=3,

        base_channels=64,
        channel_mults=[1, 2],

        time_emb_dim=128,
        time_width_coef=4,

        # convolutional tuning
        use_attention=True,
        attention_resolutions=(14,), # should be among the image_size / 2**i
        num_heads=4,

        dropout=0.1,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=8,
        eps_groupnorm=1e-5,

        down_kernel_size=4,
        down_stride=2,
        down_padding=1,
        down_num_res_blocks=1,
        up_kernel_size=4,
        up_stride=2,
        up_padding=1,
        up_num_res_blocks=1,

        # diffusion
        timesteps=50,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        s=0.008,

        # training
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0.0, 
        batch_size=64,

        # sampling
        use_ddim=True,
        ddim_steps=25,
        use_ema=True,
        ema_decay=0.9999,

        # latent diffusion
        use_latent_diffusion=True, 
        latent_dim=16, # double conv into (16, 7, 7) latent space
        latent_hidden_dim=64,
        latent_kernel_size=4,
        latent_stride=2,
        latent_padding=1,
        latent_scale_factor=0.18215
    )
    print("DiffusionModel Configuration (selected params):")
    selected_params = {
        "model_type": config.model_type,
        "loss": config.loss,
        "num_classes": config.num_classes,
        "cond_drop_prob": config.cond_drop_prob,
        "guidance_scale": config.guidance_scale,
        "image_size": config.image_size,
        "image_channels": config.image_channels,
        "base_channels": config.base_channels,
        "channel_mults": config.channel_mults,
        "time_emb_dim": config.time_emb_dim,
        "time_width_coef": config.time_width_coef,
        "use_attention": config.use_attention,
        "attention_resolutions": config.attention_resolutions,
        "num_heads": config.num_heads,
        "dropout": config.dropout, 
        "kernel_size": config.kernel_size,
        "stride": config.stride,
        "padding": config.padding,
        "num_groups": config.num_groups, 
        "eps_groupnorm": config.eps_groupnorm,
        "down_kernel_size": config.down_kernel_size, 
        "down_stride": config.down_stride, 
        "down_padding": config.down_padding, 
        "down_num_res_blocks": config.down_num_res_blocks, 
        "up_kernel_size": config.up_kernel_size, 
        "up_stride": config.up_stride, 
        "up_padding": config.up_padding, 
        "up_num_res_blocks": config.up_num_res_blocks, 
        "timesteps": config.timesteps,
        "beta_schedule": config.beta_schedule,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "s": config.s,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "use_ddim": config.use_ddim,
        "ddim_steps": config.ddim_steps,
        "use_ema": config.use_ema,
        "ema_decay": config.ema_decay,
        "use_latent_diffusion": config.use_latent_diffusion,
        "latent_dim": config.latent_dim,
        "latent_hidden_dim": config.latent_hidden_dim,
        "latent_kernel_size": config.latent_kernel_size,
        "latent_stride": config.latent_stride,
        "latent_padding": config.latent_padding,
        "latent_scale_factor": config.latent_scale_factor
    }
    for key, value in selected_params.items():
        print(f"{key}: {value}")
    print()

    print_subsection("Initialize the DiffusionModel")
    diffusion_model = DiffusionModel(config, device="cpu")
    print("DiffusionModel Architecture:")
    print(diffusion_model)

    print_subsection("Train DiffusionModel on CIFAR-10")
    epochs = 2 # for demonstration purposes, see if all is working fine
    diffusion_model.fit(cifar_loader, epochs)

    print_subsection("Sample images from trained DiffusionModel")
    n_samples = 5
    sample_labels = torch.randint(0, 10, (n_samples,))  # Randomly sample class labels for conditional generation
    print(f"Sampled class labels for conditional generation: {sample_labels}")
    sampled_images = diffusion_model.sample(n_samples, cond=sample_labels)
    sampled_images = (sampled_images + 1) / 2 # Convert from [-1,1] to [0,1]
    plot_images(sampled_images, n_samples, save_path=os.path.join(OUTPUT_DIR, "test_diffusion_model_cifar10_latent_diffusion_ddim_ema_conditional.png"), title="Sampled CIFAR-10 Images")


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    test_noise_scheduler()
    test_time_embedding()
    test_cnn()
    test_unet()
    test_latent_autoencoder()
    test_ema()
    test_diffusion_model_mnist_cnn()
    test_diffusion_model_mnist_resunet()
    test_diffusion_model_mnist_resunet_attention_ddim()
    test_diffusion_model_mnist_resunet_ema()
    test_diffusion_model_mnist_resunet_conditional()
    test_diffusion_model_mnist_latent_diffusion()
    test_diffusion_model_cifar10_resunet()
    test_diffusion_model_cifar10_latent_diffusion_ddim_ema_conditional()
    clear_data_dir()