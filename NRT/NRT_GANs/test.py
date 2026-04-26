# ===========================
# Path setup
# ===========================
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from torch.utils.data import DataLoader


# ===========================
# Imports
# ===========================
import torch


from src.data.load import (
    load_blobs,
    load_cifar10
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


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BLOB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "blobs")
os.makedirs(BLOB_OUTPUT_DIR, exist_ok=True)
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
def plot_blob_distribution(real, fake=None, save_path=None):
    plt.figure()
    plt.scatter(real[:,0], real[:,1], s=10, c='blue', label='Real')
    if fake is not None:
        plt.scatter(fake[:,0], fake[:,1], s=10, c='red', label='Fake')
    plt.legend()
    plt.title(f"{'Real vs Fake ' if fake is not None else ''}Blobs distribution")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def test_mlp_gan_blobs():
    """
    Test MLP GAN on Blobs dataset with default loss function

    This function initializes an MLP GAN with default configuration,
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
        dropout=None,
        batch_norm=False,
        spectral_norm_on=True,
        gamma=0.99, 
        learning_rate=2e-4,
        step_size=50,
        weight_decay=0.0,
        beta1=0.5,
        beta2=0.999,
        is_ema=True,
        ema_decay=0.995,
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


if __name__ == "__main__":
    print_section("Testing Generator and Discriminator architectures")

    print_subsection("Testing MLP Generator and Discriminator")
    # test_mlp_gen_disc()
    print_subsection("Testing DCGAN Generator and Discriminator")
    # test_dcgan_gen_disc()
    print_subsection("Testing CGAN Generator and Discriminator")
    # test_cgan_gen_disc()


    print_section("Testing GAN models and configurations")
    print_subsection("Testing MLP GAN on Blobs dataset with some parameter tweaking")
    test_mlp_gan_blobs()

    print_subsection("Testing MLP GAN on MNIST dataset")
    print_subsection("Testing DCGAN on MNIST dataset")
    print_subsection("Testing CGAN on MNIST dataset")
    print_subsection("Testing GAN configs on CIFAR-10 dataset")

    print("All tests completed successfully!")

    # clear_data_dir()