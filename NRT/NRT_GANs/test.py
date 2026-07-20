# ===========================
# Path setup
# ===========================
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from torch.utils.data import DataLoader


# ===========================
# Imports
# ===========================
from src.data.load import (
    load_blobs,
    load_mnist,
    load_cifar10
)
from src.data.utils import (
    set_seed,
    plot_blob_distribution,
    plot_images
)
from src.models.GANs import (
    MLPGenerator, 
    MLPDiscriminator,
    DCGANGenerator,
    DCGANDiscriminator,
    CGANGenerator,
    CGANDiscriminator,
    GANConfig,
    GAN
)


RANDOM_SEED = 42

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BLOB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "blobs")
os.makedirs(BLOB_OUTPUT_DIR, exist_ok=True)
MNIST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mnist")
os.makedirs(MNIST_OUTPUT_DIR, exist_ok=True)
CIFAR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cifar10")
os.makedirs(CIFAR_OUTPUT_DIR, exist_ok=True)


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def print_section(title):
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
# Testing Generator and Discriminator
# ===========================
def test_mlp_gen_disc():
    """
    Test MLP Generator and Discriminator on MNIST dataset

    This function initializes an MLP Generator and Discriminator, 
    and tests their forward passes.
    """
    # MLP Generator and Discriminator
    mlp_gen = MLPGenerator(
        latent_dim=100,
        output_dim=16*16,
        hidden_dims=[128, 256],
        dropout=0.2, 
        batch_norm=True
    )
    mlp_disc = MLPDiscriminator(
        input_dim=16*16,
        hidden_dims=[256, 128],
        dropout=0.2, 
        spectral_norm_on=True
    )

    # Test Generator
    z = torch.randn(128, 100)  # Batch of latent vectors
    gen_samples = mlp_gen(z)
    print("MLP Generator output shape:", gen_samples.shape)  # Should be (128, 256)
    print("Sample generated data (first 5 samples):")
    for i in range(5):
        print(gen_samples[i].detach().numpy())

    # Test Discriminator
    x = torch.randn(128, 16*16)  # Batch of real/fake samples
    disc_output = mlp_disc(x)
    print("MLP Discriminator output shape:", disc_output.shape)  # Should be (128, 1)
    print("Sample discriminator output (first 5 samples):")
    for i in range(5):
        print(disc_output[i].detach().numpy())
    

def test_dcgan_gen_disc():
    """
    Test DCGAN Generator and Discriminator on MNIST dataset

    This function initializes a DCGAN Generator and Discriminator,
    and tests their forward passes.
    """
    # DCGAN Generator and Discriminator
    dcgan_gen = DCGANGenerator(
        image_size=16,
        image_channels=1,
        conv_channels=[64, 128, 256],
        latent_dim=100,
        kernel_size=4,
        stride=2,
        padding=1,
        dropout=0.2, 
        batch_norm=True
    )
    dcgan_disc = DCGANDiscriminator(
        image_size=16,
        image_channels=1,
        conv_channels=[64, 128, 256],
        kernel_size=4,
        stride=2,
        padding=1,
        dropout=0.2, 
        spectral_norm_on=True
    )

    # Test Generator
    z = torch.randn(128, 100)  # Batch of latent vectors
    gen_samples = dcgan_gen(z)
    print("DCGAN Generator output shape:", gen_samples.shape)  # Should be (128, 1, 16, 16)
    print("Sample generated data (first sample):")
    print(gen_samples[0].detach().numpy())

    # Test Discriminator
    x = torch.randn(128, 1, 16, 16)  # Batch of real/fake samples
    disc_output = dcgan_disc(x)
    print("DCGAN Discriminator output shape:", disc_output.shape)  # Should be (128, 1)
    print("Sample discriminator output (first 5 samples):")
    for i in range(5):
        print(disc_output[i].detach().numpy())


def test_cgan_gen_disc():
    """
    Test CGAN Generator and Discriminator on MNIST dataset

    This function initializes a CGAN Generator and Discriminator,
    and tests their forward passes.
    """
    # CGAN Generator and Discriminator
    cgan_gen = CGANGenerator(
        latent_dim=100,
        hidden_dim=[128, 256],
        num_classes=10,
        output_dim=16*16,
        dropout=0.2,
        batch_norm=True
    )
    cgan_disc = CGANDiscriminator(
        input_dim=16*16,
        hidden_dim=[256, 128],
        num_classes=10,
        dropout=0.2, 
        spectral_norm_on=True
    )

    # Test Generator
    z = torch.randn(128, 100)  # Batch of latent vectors
    labels = torch.randint(0, 10, (128,))  # Batch of random class labels
    gen_samples = cgan_gen(z, labels)
    print("CGAN Generator output shape:", gen_samples.shape)  # Should be (128, 256)
    print("Sample generated data (first 5 samples):")
    for i in range(5):
        print(gen_samples[i].detach().numpy())

    # Test Discriminator
    x = torch.randn(128, 16*16)  # Batch of real/fake samples
    labels = torch.randint(0, 10, (128,))  # Batch of random class labels
    disc_output = cgan_disc(x, labels)
    print("CGAN Discriminator output shape:", disc_output.shape)  # Should be (128, 1)
    print("Sample discriminator output (first 5 samples):")
    for i in range(5):
        print(disc_output[i].detach().numpy())


# ===========================
# Testing GANs models and configurations to ensure they are properly working 
# ===========================
def test_mlp_gan_blobs():
    """
    Test MLP GAN on Blobs dataset with default loss function

    This function initializes an MLP GAN with default configuration,
    and tests its forward pass and loss computation, 
    and then verify the sampling (with ema)
    """
    print("Loading Blobs dataset...")
    space_dim = 2
    cent = [(-0.4, 0), (-0.285, -0.285), (-0.285, 0.285), (0., -0.4), (0., 0.4), (0.285, -0.285), (0.285, 0.285), (0.4, 0)]
    blob_tensor = load_blobs(n_samples=1600, n_features=space_dim, centers=cent, cluster_std=0.04, random_state=42)
    print("Blobs dataset shape:", blob_tensor.shape) # Should be (1600, 2)

    # Change the range of the blob data to be between -1 and 1 for better GAN training stability
    print("old range of blob data: min =", blob_tensor.min().item(), "max =", blob_tensor.max().item())
    blob_min = blob_tensor.min(dim=0, keepdim=True)[0]
    blob_max = blob_tensor.max(dim=0, keepdim=True)[0]
    blob_tensor = 2 * (blob_tensor - blob_min) / (blob_max - blob_min + 1e-8) - 1
    print("new range of blob data: min =", blob_tensor.min().item(), "max =", blob_tensor.max().item())

    # Plot real data distribution
    plot_blob_distribution(blob_tensor.numpy(), save_path=os.path.join(BLOB_OUTPUT_DIR,"blobs_real.png"))    

    # Prepare DataLoader
    blob_dataset = torch.utils.data.TensorDataset(blob_tensor)
    blob_loader = DataLoader(blob_dataset, batch_size=128, shuffle=True)
    print("Test batch content: ", next(iter(blob_loader))[0][:5])  # Print first 5 samples of the first batch
    
    # Define GAN configuration
    config = GANConfig(
        architecture="GAN", 
        loss="LeastSquare",
        latent_dim=16,
        input_dim=space_dim,
        hidden_dims=[128, 128],
        lsgan_lambda=0.5,
        dropout=None,
        batch_norm=False,
        spectral_norm_on=True,
        learning_rate=2e-4,
        step_size=50,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,
        is_ema=True,
        ema_decay=0.995
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process 
    blob_loader = DataLoader(blob_tensor, batch_size=128, shuffle=True)
    epochs = 300
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(BLOB_OUTPUT_DIR, "mlp_gan_blobs")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training MLP GAN on Blobs dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(blob_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(1000).numpy()
        plot_blob_distribution(blob_tensor.numpy(), samples, save_path=os.path.join(TEST_OUTPUT_DIR,f"blobs_gen_rep{rep+1}.png"))
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("MLP GAN Loss History on Blobs dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_mlp_wasserstein_gan_blobs():
    """
    Test MLP GAN on Blobs dataset with wasserstein loss function and gradient penalty

    This function initializes an MLP GAN with wasserstein loss and gradient penalty configuration,
    and tests its forward pass and loss computation,    
    and then verify the sampling (without ema)
    """
    print("Loading Blobs dataset...")
    space_dim = 2
    cent = [(-0.4, 0), (-0.285, -0.285), (-0.285, 0.285), (0., -0.4), (0., 0.4), (0.285, -0.285), (0.285, 0.285), (0.4, 0)]
    blob_tensor = load_blobs(n_samples=1600, n_features=space_dim, centers=cent, cluster_std=0.04, random_state=42)
    print("Blobs dataset shape:", blob_tensor.shape) # Should be (1600, 2)  

    # Change the range of the blob data to be between -1 and 1 for better GAN training stability
    print("old range of blob data: min =", blob_tensor.min().item(), "max =", blob_tensor.max().item())
    blob_min = blob_tensor.min(dim=0, keepdim=True)[0]
    blob_max = blob_tensor.max(dim=0, keepdim=True)[0]
    blob_tensor = 2 * (blob_tensor - blob_min) / (blob_max - blob_min + 1e-8) - 1
    print("new range of blob data: min =", blob_tensor.min().item(), "max =", blob_tensor.max().item())

    # Plot real data distribution
    # plot_blob_distribution(blob_tensor.numpy(), save_path=os.path.join(BLOB_OUTPUT_DIR,"blobs_real.png"))

    # Prepare DataLoader
    blob_dataset = torch.utils.data.TensorDataset(blob_tensor)
    blob_loader = DataLoader(blob_dataset, batch_size=128, shuffle=True)
    print("Test batch content: ", next(iter(blob_loader))[0][:5])  # Print first 5 samples of the first batch

    # Define GAN configuration
    config = GANConfig(
        architecture="GAN", 
        loss="Wasserstein",
        latent_dim=16,
        input_dim=space_dim,
        hidden_dims=[128, 128],
        weight_clip=0.01,
        gradient_penalty_lambda=10.0,
        n_critic=5,
        dropout=None,
        batch_norm=True,
        spectral_norm_on=True,
        learning_rate=2e-4,
        step_size=50,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,
        is_ema=False,
        ema_decay=0.995
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    blob_loader = DataLoader(blob_tensor, batch_size=128, shuffle=True)
    epochs = 300
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(BLOB_OUTPUT_DIR, "mlp_wgan_blobs")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training MLP GAN with Wasserstein loss and gradient penalty on Blobs dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(blob_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(1000).numpy()
        plot_blob_distribution(blob_tensor.numpy(), samples, save_path=os.path.join(TEST_OUTPUT_DIR,f"blobs_gen_rep{rep+1}.png"))
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("MLP Wasserstein GAN Loss History on Blobs dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_unrolled_gan_blobs():
    """ 
    Test Unrolled GAN on Blobs dataset with default loss function

    This function initializes an Unrolled GAN with default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (with ema)
    """    
    print("Testing Unrolled GAN on Blobs dataset ...")
    space_dim = 2
    cent = [(-0.4, 0), (-0.285, -0.285), (-0.285, 0.285), (0., -0.4), (0., 0.4), (0.285, -0.285), (0.285, 0.285), (0.4, 0)]
    blob_tensor = load_blobs(n_samples=1600, n_features=space_dim, centers=cent, cluster_std=0.04, random_state=42)
    print("Blobs dataset shape:", blob_tensor.shape) # Should be (1600, 2)

    # Change the range of the blob data to be between -1 and 1 for better GAN training stability
    print("old range of blob data: min =", blob_tensor.min().item(), "max =", blob_tensor.max().item())
    blob_min = blob_tensor.min(dim=0, keepdim=True)[0]
    blob_max = blob_tensor.max(dim=0, keepdim=True)[0]
    blob_tensor = 2 * (blob_tensor - blob_min) / (blob_max - blob_min + 1e-8) - 1
    print("new range of blob data: min =", blob_tensor.min().item(), "max =", blob_tensor.max().item())

    # Plot real data distribution
    # plot_blob_distribution(blob_tensor.numpy(), save_path=os.path.join(BLOB_OUTPUT_DIR,"blobs_real.png"))

    # Prepare DataLoader
    blob_dataset = torch.utils.data.TensorDataset(blob_tensor)
    blob_loader = DataLoader(blob_dataset, batch_size=128, shuffle=True)    
    print("Test batch content: ", next(iter(blob_loader))[0][:5])  # Print first 5 samples of the first batch

    # Define GAN configuration
    config = GANConfig(
        architecture="MLP_UnrolledGAN",
        loss="Default",
        latent_dim=16,
        input_dim=space_dim,
        hidden_dims=[128, 128],
        unrolled_steps=5,
        dropout=None,
        batch_norm=True,
        spectral_norm_on=True,
        learning_rate=2e-4,
        step_size=50,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,
        is_ema=True,
        ema_decay=0.995
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    blob_loader = DataLoader(blob_tensor, batch_size=128, shuffle=True)
    epochs = 300
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(BLOB_OUTPUT_DIR, "mlp_unrolled_gan_blobs")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training Unrolled MLP GAN on Blobs dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(blob_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(1000).numpy()
        plot_blob_distribution(blob_tensor.numpy(), samples, save_path=os.path.join(TEST_OUTPUT_DIR,f"blobs_gen_rep{rep+1}.png"))
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("Unrolled MLP GAN Loss History on Blobs dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_mlp_gan_mnist():
    """
    Test MLP GAN on MNIST dataset with default loss function

    This function initializes an MLP GAN with default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (with ema)
    """
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=128, train=True)

    # Plot real data distribution
    first_numbers = next(iter(mnist_loader))[0][:5]  # Get the first 5 samples of the first batch
    print(first_numbers.shape)  # torch.Size([5, 784]) for 28x28 downsampled images
    plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR,"mnist_real.png"), title="Image 1 from MNIST dataset")
    
    # Define GAN configuration
    config = GANConfig(
        architecture="GAN",
        loss="Default",
        latent_dim=64,
        input_dim=784,
        hidden_dims=[256, 256],
        dropout=0.0,
        batch_norm=False,
        spectral_norm_on=True,
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,
        is_ema=True,
        ema_decay=0.995
    )
    print("GAN configuration:")
    print(config)
    print() 

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(MNIST_OUTPUT_DIR, "mlp_gan_mnist")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training MLP GAN on MNIST dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(mnist_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"mnist_gen_rep{rep+1}.png"), title=f"Generated Images from MLP GAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("MLP GAN Loss History on MNIST dataset")  
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_mlp_wasserstein_gan_mnist():
    """
    Test MLP GAN on MNIST dataset with wasserstein loss function and gradient penalty

    This function initializes an MLP GAN with wasserstein loss and gradient penalty configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (without ema)
    """
    print("Testing MLP GAN with Wasserstein loss and gradient penalty on MNIST dataset ...")
    mnist_loader = load_mnist(batch_size=128, train=True)
    
    # Plot real data distribution
    # first_numbers = next(iter(mnist_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_numbers.shape)  # torch.Size([5, 784]) for 28x28 downsampled images
    # plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR,"mnist_real.png"), title="Image 1 from MNIST dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="GAN",
        loss="Wasserstein",
        latent_dim=64,
        input_dim=784,
        hidden_dims=[256, 256],
        weight_clip=0.01,
        gradient_penalty_lambda=10.0,
        n_critic=5,
        dropout=0.0,
        batch_norm=False,
        spectral_norm_on=True,
        learning_rate=1e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.0,
        beta2=0.9,
        is_ema=False,
        ema_decay=0.995
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(MNIST_OUTPUT_DIR, "mlp_wgan_mnist")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training MLP GAN with Wasserstein loss and gradient penalty on MNIST dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(mnist_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"mnist_gen_rep{rep+1}.png"), title=f"Generated Images from MLP Wasserstein GAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("MLP Wasserstein GAN Loss History on MNIST dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_unrolled_gan_mnist():
    """
    Test Unrolled GAN on MNIST dataset with default loss function

    This function initializes an Unrolled GAN with default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (with ema)
    """
    print("Testing Unrolled GAN on MNIST dataset ...")
    mnist_loader = load_mnist(batch_size=128, train=True)

    # Plot real data distribution
    # first_numbers = next(iter(mnist_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_numbers.shape)  # torch.Size([5, 784]) for 28x28 downsampled images
    # plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR,"mnist_real.png"), title="Image 1 from MNIST dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="MLP_UnrolledGAN",
        loss="Default",
        latent_dim=64,
        input_dim=784,
        hidden_dims=[256, 256],
        unrolled_steps=5,
        dropout=0.0,
        batch_norm=False,
        spectral_norm_on=True,
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,
        is_ema=True,
        ema_decay=0.995
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(MNIST_OUTPUT_DIR, "mlp_unrolled_gan_mnist")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training Unrolled MLP GAN on MNIST dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(mnist_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"mnist_gen_rep{rep+1}.png"), title=f"Generated Images from Unrolled MLP GAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("Unrolled MLP GAN Loss History on MNIST dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_cgan_gan_mnist():
    """
    Test CGAN on MNIST dataset with default loss function

    This function initializes a CGAN with default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (with ema)
    """
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=128, train=True)  

    # Plot real data distribution
    # first_numbers = next(iter(mnist_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_numbers.shape)  # torch.Size([5, 784]) for 28x28 downsampled images
    # plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR,"mnist_real.png"), title="Image 1 from MNIST dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="CGAN",
        loss="Default",
        latent_dim=64,
        input_dim=784,
        hidden_dims=[256, 256],
        num_classes=10,
        dropout=0.0,
        batch_norm=False,
        spectral_norm_on=True,
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=True,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(MNIST_OUTPUT_DIR, "cgan_mnist")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training CGAN on MNIST dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(mnist_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5, labels=torch.tensor([0,1,2,3,4])).numpy()  # Sample one image for each of the first 5 classes
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"mnist_gen_rep{rep+1}.png"), title=f"Generated Images from CGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("CGAN Loss History on MNIST dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_dcgan_mnist():
    """
    Test DCGAN on MNIST dataset with default loss function
    
    This function initializes a DCGAN with default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (with ema)
    """
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=128, downsample=(16, 16), normalize=True, flatten=False, train=True)

    # Plot real data distribution
    # first_numbers = next(iter(mnist_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_numbers.shape)  # torch.Size([5, 1, 16, 16]) for 16x16 downsampled images
    # plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR,"mnist_real.png"), title="Image 1 from MNIST dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="DCGAN",
        loss="Default",
        latent_dim=32,
        hidden_dims=[128, 64],
        image_size=16,
        image_channels=1,
        kernel_size=4,
        stride=2,
        padding=1,
        dropout=0.0,
        batch_norm=True,
        spectral_norm_on=False, # too slow otherswise for this test
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=False,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 6
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(MNIST_OUTPUT_DIR, "dcgan_mnist")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training DCGAN on MNIST dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(mnist_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"mnist_gen_rep{rep+1}.png"), title=f"Generated Images from DCGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()    
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("DCGAN Loss History on MNIST dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_dcgan_wasserstein_mnist():
    """
    Test DCGAN on MNIST dataset with wasserstein loss function and gradient penalty

    This function initializes a DCGAN with wasserstein loss and gradient penalty configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (without ema)
    """
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=128, downsample=(16, 16), normalize=True, flatten=False, train=True)

    # Plot real data distribution
    # first_numbers = next(iter(mnist_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_numbers.shape)  # torch.Size([5, 256]) for 16x16 downsampled images
    # plot_images(first_numbers, 5, save_path=os.path.join(MNIST_OUTPUT_DIR,"mnist_real.png"), title="Image 1 from MNIST dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="DCGAN",
        loss="Wasserstein",
        latent_dim=32,
        hidden_dims=[128, 64],
        image_size=16,
        image_channels=1,
        kernel_size=4,
        stride=2,
        padding=1,
        weight_clip=0.01,
        gradient_penalty_lambda=10.0,
        n_critic=5,
        dropout=0.0,
        batch_norm=True,
        spectral_norm_on=False, # too slow otherswise for this test
        learning_rate=1e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.0,
        beta2=0.9,
        is_ema=False,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 1
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(MNIST_OUTPUT_DIR, "dcgan_wgan_mnist")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training DCGAN with Wasserstein loss and gradient penalty on MNIST dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(mnist_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"mnist_gen_rep{rep+1}.png"), title=f"Generated Images from DCGAN Wasserstein GAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("DCGAN with Wasserstein loss and gradient penalty Loss History on MNIST dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_cgan_cifar10():
    """
    Test CGAN on CifAR-10 dataset with default loss function

    This function initializes a CGAN with default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (with ema)
    """
    print("Testing CGAN on CifAR-10 dataset ...")
    cifar_loader = load_cifar10(batch_size=128, downsample=(16, 16), normalize=True, flatten=True, train=True)

    # Plot real data distribution
    first_images = next(iter(cifar_loader))[0][:5]  # Get the first 5 samples of the first batch
    print(first_images.shape)  # torch.Size([5, 768]) for 16x16 downsampled images with 3 channels
    # reshape to (5, 3, 16, 16) for plotting
    first_images = first_images.view(-1, 3, 16, 16)
    first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    print(first_images.shape)  # torch.Size([5, 3, 16, 16])
    plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR,"cifar10_real.png"), title="Image 1 from Cifar-10 dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="CGAN",
        loss="Default",
        latent_dim=64,
        input_dim=768,
        hidden_dims=[256, 256],
        num_classes=10,
        dropout=0.0,
        batch_norm=False,
        spectral_norm_on=True,
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=True,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 5
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(CIFAR_OUTPUT_DIR, "cgan_cifar10")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training CGAN on Cifar-10 dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(cifar_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5, labels=torch.tensor([0,1,2,3,4])).numpy()  # Sample one image for each of the first 5 classes
        samples = (samples + 1) / 2 # Convert from [-1,1] to [0,1]
        samples = samples.reshape(-1, 3, 16, 16)  # Reshape to (5, 3, 16, 16) for plotting
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"cifar10_gen_rep{rep+1}.png"), title=f"Generated Images from CGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("CGAN Loss History on Cifar-10 dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_dcgan_cifar10():
    """
    Test DCGAN on CifAR-10 dataset with default loss function

    This function initializes a DCGAN with default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (without ema)
    """
    print("Testing DCGAN on Cifar-10 dataset ...")
    cifar_loader = load_cifar10(batch_size=128, downsample=(16, 16), normalize=True, flatten=False, train=True)

    # Plot real data distribution
    # first_images = next(iter(cifar_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_images.shape)  # torch.Size([5, 3, 16, 16]) for 16x16 downsampled images with 3 channels
    # first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    # plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR,"cifar10_real.png"), title="Image 1 from Cifar-10 dataset")
     
    # Define GAN configuration
    config = GANConfig(
        architecture="DCGAN",
        loss="Default",
        latent_dim=64,
        hidden_dims=[256, 128, 64],
        image_size=16,
        image_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        dropout=0.0,
        batch_norm=True,
        spectral_norm_on=False, # too slow otherswise for this test
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=False,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)
    print()

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 5
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(CIFAR_OUTPUT_DIR, "dcgan_cifar10")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training DCGAN on Cifar-10 dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(cifar_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        samples = (samples + 1) / 2 # Convert from [-1,1] to [0,1]
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"cifar10_gen_rep{rep+1}.png"), title=f"Generated Images from DCGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("DCGAN Loss History on Cifar-10 dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_dcgan_cifar10_2():
    """
    Test DCGAN on downsampled CifAR-10 dataset with default loss function

    This function initializes a DCGAN with small architecture and default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (without ema)
    Note: This test is similar to the previous one but with a smaller architecture and more epochs to test going deeper in training and see if the model can learn to generate better samples with more training, even with a smaller architecture.
    This is also to test the stability of the training process over more epochs.
    """
    print("Testing DCGAN on Cifar-10 dataset with more epochs and smaller architecture ...")
    cifar_loader = load_cifar10(batch_size=128, downsample=(16, 16), normalize=True, flatten=False, train=True, subset_size=2000)

    # Plot real data distribution
    # first_images = next(iter(cifar_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_images.shape)  # torch.Size([5, 3, 16, 16]) for 16x16 downsampled images with 3 channels
    # first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    # plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR,"cifar10_real.png"), title="Image 1 from Cifar-10 dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="DCGAN",
        loss="Default",
        latent_dim=64,
        hidden_dims=[128, 64],
        image_size=16,
        image_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        dropout=0.0,
        batch_norm=True,
        spectral_norm_on=False, # too slow otherswise for this test
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=False,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(CIFAR_OUTPUT_DIR, "dcgan_cifar10_2")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training DCGAN on Cifar-10 dataset with more epochs and smaller architecture - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(cifar_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        samples = (samples + 1) / 2 # Convert from [-1,1] to [0,1]
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"cifar10_gen_rep{rep+1}.png"), title=f"Generated Images from DCGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("DCGAN Loss History on Cifar-10 dataset with more epochs and smaller architecture")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_dcgan_cifar10_3():
    """
    Test DCGAN on downsized CifAR-10 dataset with default loss function and a slightly bigger architecture than the previous test

    This function initializes a DCGAN with a slightly bigger architecture than the previous test and default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (without ema)
    Note: This test is similar to the previous one but with a slightly bigger architecture and 32x32 images to test if the model can learn to generate better samples with a slightly bigger architecture and more details in the images, even with more epochs.
    The subset still making it roughly 10x faster than the first one 
    """
    print("Testing DCGAN on Cifar-10 dataset with more epochs and slightly bigger architecture ...")
    cifar_loader = load_cifar10(batch_size=128, downsample=None, normalize=True, flatten=False, train=True, subset_size=2000)

    # Plot real data distribution
    first_images = next(iter(cifar_loader))[0][:5]  # Get the first 5 samples of the first batch
    print(first_images.shape)  # torch.Size([5, 3, 32, 32]) for 32x32 images with 3 channels
    first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR,"cifar10_real_32_32.png"), title="Image 1 from Cifar-10 dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="DCGAN",
        loss="Default",
        latent_dim=64,
        hidden_dims=[256, 128, 64],
        image_size=32,
        image_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        noise_coef=0.03,
        dropout=0.0,
        batch_norm=True,
        spectral_norm_on=False, # too slow otherswise for this test
        learning_rate=1e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=False,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(CIFAR_OUTPUT_DIR, "dcgan_cifar10_3")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training DCGAN on Cifar-10 dataset with more epochs and slightly bigger architecture - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(cifar_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        samples = (samples + 1) / 2 # Convert from [-1,1] to [0,1]
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"cifar10_gen_rep{rep+1}.png"), title=f"Generated Images from DCGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("DCGAN Loss History on Cifar-10 dataset with more epochs and slightly bigger architecture")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_dcgan_cifar10_4():
    """
    Test DCGAN on downsized CifAR-10 dataset, trying to have result there 

    This function initializes a DCGAN with a smaller architecture than the previous test and default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (without ema)
    Note: This test is similar to the previous one but with a smaller architecture and 32
    """
    print("Testing DCGAN on Cifar-10 dataset with less downsampling and a slightly smaller architecture ...")
    cifar_loader = load_cifar10(batch_size=64, downsample=None, normalize=True, flatten=False, train=True, subset_size=10000)

    # Plot real data distribution
    # first_images = next(iter(cifar_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_images.shape)  # torch.Size([5, 3, 32, 32]) for 32x32 images with 3 channels
    # first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    # plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR,"cifar10_real_32_32.png"), title="Image 1 from Cifar-10 dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="DCGAN",
        loss="Default",
        latent_dim=64,
        hidden_dims=[128, 64],
        image_size=32,
        image_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        noise_coef=0.02,
        dropout=0.0,
        batch_norm=True,
        spectral_norm_on=False, # too slow otherswise for this test
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=False,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)
    print("GAN Architecture:")
    print(gan)

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(CIFAR_OUTPUT_DIR, "dcgan_cifar10_4")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training DCGAN on Cifar-10 dataset with less downsampling and slightly smaller architecture - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(cifar_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(10).cpu().numpy()
        samples = (samples + 1) / 2 # Convert from [-1,1] to [0,1]
        plot_images(samples, 10, save_path=os.path.join(TEST_OUTPUT_DIR,f"cifar10_gen_rep{rep+1}.png"), title=f"Generated Images from DCGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("DCGAN Loss History on Cifar-10 dataset with less downsampling and slightly smaller architecture")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


def test_stylegan_cifar10():
    """
    Quick test to check if StyleGAN architecture can be initialized and trained on Cifar-10 dataset with default configuration, and if it can generate samples without errors. 
    This is not meant to produce good results but just to check if the code runs without errors and the training process is stable for a few epochs.

    This function initializes a StyleGAN with default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (without ema)
    """
    print("Testing StyleGAN on Cifar-10 dataset ...")
    cifar_loader = load_cifar10(batch_size=128, downsample=(16, 16), normalize=True, flatten=False, train=True, subset_size=2000)

    # Plot real data distribution
    # first_images = next(iter(cifar_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_images.shape)  # torch.Size([5, 3, 16, 16]) for 16x16 downsampled images with 3 channels
    # first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    # plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR,"cifar10_real.png"), title="Image 1 from Cifar-10 dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="StyleGAN",
        loss="Default",
        latent_dim=32,
        hidden_dims=[128, 64],
        image_channels=3,
        style_dim=32,
        image_size=16,
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=False,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)    
    print("GAN Architecture:")
    print(gan)

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 1
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(CIFAR_OUTPUT_DIR, "stylegan_cifar10")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training StyleGAN on Cifar-10 dataset - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(cifar_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        samples = (samples + 1) / 2 # Convert from [-1,1] to [0,1]
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"cifar10_gen_rep{rep+1}.png"), title=f"Generated Images from StyleGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("StyleGAN Loss History on Cifar-10 dataset")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()
    

def test_stylegan_cifar10_2():
    """
    Test StyleGAN on Cifar-10 dataset with least square loss function and a slightly bigger architecture than the previous test
    to try having some reasonable results, even with more epochs, and to test the stability of the training process over more epochs.

    This function initializes a StyleGAN with a slightly bigger architecture than the previous test and default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (with ema)
    """
    print("Testing StyleGAN on Cifar-10 dataset with more epochs and slightly bigger architecture ...")
    cifar_loader = load_cifar10(batch_size=128, downsample=None, normalize=True, flatten=False, train=True, subset_size=10000)

    # Plot real data distribution
    # first_images = next(iter(cifar_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_images.shape)  # torch.Size([5, 3, 32, 32]) for 32x32 images with 3 channels
    # first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    # plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR,"cifar10_real.png"), title="Image 1 from Cifar-10 dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="StyleGAN",
        loss="LeastSquare",
        latent_dim=128,
        hidden_dims=[256, 128, 64],
        image_channels=3,
        style_dim=128,
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=True,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)    
    print("GAN Architecture:")
    print(gan)

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(CIFAR_OUTPUT_DIR, "stylegan_cifar10_2")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training StyleGAN on Cifar-10 dataset with more epochs and slightly bigger architecture - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(cifar_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        samples = (samples + 1) / 2 # Convert from [-1,1] to [0,1]
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"cifar10_gen_rep{rep+1}.png"), title=f"Generated Images from StyleGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("StyleGAN Loss History on Cifar-10 dataset with more epochs and slightly bigger architecture")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()



def test_stylegan_cifar10_3():
    """
    Test StyleGAN on Cifar-10 dataset with least square loss function and a slightly bigger architecture than the previous test
    to try having some reasonable results, even with more epochs, and to test the stability of the training process over more epochs.

    This function initializes a StyleGAN with a slightly bigger architecture than the previous test and default configuration,
    and tests its forward pass and loss computation,
    and then verify the sampling (with ema)
    """
    print("Testing StyleGAN on Cifar-10 dataset with more epochs and slightly bigger architecture ...")
    cifar_loader = load_cifar10(batch_size=128, downsample=None, normalize=True, flatten=False, train=True, subset_size=10000)

    # Plot real data distribution
    # first_images = next(iter(cifar_loader))[0][:5]  # Get the first 5 samples of the first batch
    # print(first_images.shape)  # torch.Size([5, 3, 32, 32]) for 32x32 images with 3 channels
    # first_images = (first_images + 1) / 2 # Convert from [-1,1] to [0,1]
    # plot_images(first_images, 5, save_path=os.path.join(CIFAR_OUTPUT_DIR,"cifar10_real.png"), title="Image 1 from Cifar-10 dataset")

    # Define GAN configuration
    config = GANConfig(
        architecture="StyleGAN",
        loss="LeastSquare",
        latent_dim=32,
        hidden_dims=[32, 32, 16],
        image_channels=3,
        kernel_size=4,
        stride=2,
        padding=1,
        style_dim=64,
        kernel_size_style_gen=3,
        stride_style_gen=1,
        padding_style_gen=1,
        noise_weight=0.05,
        mixing_prob=0.9,   
        learning_rate=2e-4,
        step_size=20,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,    
        is_ema=True,
        ema_decay=0.999
    )
    print("GAN configuration:")
    print(config)
    print()

    # Initialize GAN
    device = "cpu"
    gan = GAN(config, device=device)    
    print("GAN Architecture:")
    print(gan)

    # Train the model for a few epochs then plot the generated samples, then repeat the process
    epochs = 10
    nb_reps = 5
    TEST_OUTPUT_DIR = os.path.join(CIFAR_OUTPUT_DIR, "stylegan_cifar10_3")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    G_losses = []
    D_losses = []
    for rep in range(nb_reps):
        print(f"Training StyleGAN on Cifar-10 dataset with more epochs and slightly bigger architecture - Rep {rep+1}/{nb_reps} - Epochs: {epochs*rep} to {epochs*(rep+1)}")
        history = gan.fit(cifar_loader, epochs=epochs, verbose=False)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])
        samples = gan.sample(5).numpy()
        samples = (samples + 1) / 2 # Convert from [-1,1] to [0,1]
        plot_images(samples, 5, save_path=os.path.join(TEST_OUTPUT_DIR,f"cifar10_gen_rep{rep+1}.png"), title=f"Generated Images from StyleGAN - Rep {rep+1}")
        print()

    # Plot loss history
    plt.figure()
    plt.plot(G_losses, label="Generator Loss", color='red')
    plt.plot(D_losses, label="Discriminator Loss", color='blue')
    plt.legend()
    plt.title("StyleGAN Loss History on Cifar-10 dataset with more epochs and slightly bigger architecture")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(TEST_OUTPUT_DIR,"loss_history.png"))
    plt.close()


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    print_section("Testing Generator and Discriminator architectures")

    print_subsection("Testing MLP Generator and Discriminator")
    test_mlp_gen_disc()
    print_subsection("Testing DCGAN Generator and Discriminator")
    test_dcgan_gen_disc()
    print_subsection("Testing CGAN Generator and Discriminator")
    test_cgan_gen_disc()


    print_section("Testing GAN models and configurations")
    print_subsection("Testing MLP GAN on Blobs dataset with some parameter tweaking")
    test_mlp_gan_blobs()
    test_mlp_wasserstein_gan_blobs()
    test_unrolled_gan_blobs()

    print_subsection("Testing MLP GAN on MNIST dataset")
    test_mlp_gan_mnist()
    test_mlp_wasserstein_gan_mnist()
    test_unrolled_gan_mnist()

    print_subsection("Testing CGAN on MNIST dataset")
    test_cgan_gan_mnist()

    print_subsection("Testing DCGAN on MNIST dataset")
    test_dcgan_mnist()
    test_dcgan_wasserstein_mnist()

    print_subsection("Testing CGAN on CifAR-10 dataset")
    test_cgan_cifar10()

    print_subsection("Testing DCGAN on CifAR-10 dataset")
    test_dcgan_cifar10()
    test_dcgan_cifar10_2()
    test_dcgan_cifar10_3()
    test_dcgan_cifar10_4()
    
    print_subsection("Testing StyleGAN on CifAR-10 dataset")
    test_stylegan_cifar10()
    test_stylegan_cifar10_2()
    test_stylegan_cifar10_3()

    print("All tests completed successfully!")
    clear_data_dir()