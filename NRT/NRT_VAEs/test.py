# ===========================
# Path setup
# ===========================
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from torch.utils.data import DataLoader, TensorDataset


# ===========================
# Imports
# ===========================
from src.data.load import (
    load_blobs,
    load_mnist
)
from src.models.VAEs import (
    VAEConfig, 
    BaseVAE, 
    FastCNNVAE
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BLOB_DIR = os.path.join(OUTPUT_DIR, "blobs")
os.makedirs(BLOB_DIR, exist_ok=True)
MNIST_DIR = os.path.join(OUTPUT_DIR, "mnist")
os.makedirs(MNIST_DIR, exist_ok=True)

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
# Testing VAEs on MNIST
# ===========================
def test_MLP_BaseVAE_mnist():
    """
    Test the MLP-based VAE on the MNIST dataset with downsampling to 16x16

    This function loads the MNIST dataset, 
    initializes a VAE with an MLP architecture,
    trains it for a few epochs, save and load the model,
    and then verify the called methods for reconstruction and sampling
    """
    print_subsection("Testing MLP-based VAE on MNIST")
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=128, downsample=(16, 16), train=True)

    # Define VAE configuration
    cfg = VAEConfig(
        model_type="vae",
        architecture="mlp",   
        input_dim=256,       # 16*16 after downsampling
        hidden_dims=(128, 64),
        latent_dim=32,
        dropout=0.1,
        use_batchnorm=True,
        beta_kl=1.0,
        gamma=0.5,
        learning_rate=1e-3,
        step_size=10,
        weight_decay=1e-5
    )
    print("VAE configuration:")
    print(cfg)
    print()

    # Initialize VAE
    device = "cpu"
    vae = BaseVAE(cfg, device=device)
    print("VAE architecture:")
    print(vae)
    print()

    # Train VAE for a few epochs
    print("Training VAE...")
    metrics = vae.fit(mnist_loader, epochs=5, verbose=True)
    print()
    print("Training metrics:")
    for epoch, metric in enumerate(metrics):
        print(f"Epoch {epoch+1}: Loss={metric.loss:.4f}, Recon Loss={metric.recon:.4f}, KL Div={metric.kld:.4f}")
    print()

    # Save / Load model 
    print("Saving...")
    vae.save(os.path.join(MNIST_DIR, "mlp_vae.pth"))
    cfg.save(os.path.join(MNIST_DIR, "mlp_vae_config.json"))
    print("Model saved to mlp_vae.pth and configuration saved to mlp_vae_config.json")
    print()

    print("Loading...")
    cfg = VAEConfig()
    cfg.load(os.path.join(MNIST_DIR, "mlp_vae_config.json"))
    vae = BaseVAE(cfg, device=device)
    vae.load(os.path.join(MNIST_DIR, "mlp_vae.pth"))
    print("Model loaded from mlp_vae.pth with configuration from mlp_vae_config.json")
    print()

    # Test reconstructions and sampling
    x_test, _ = next(iter(mnist_loader))
    x_test = x_test.to(device)
    print("Test batch shape:", x_test.shape)  # Should be (128, 256)

    x_hat = vae.reconstruct(x_test)
    print("Reconstructed batch shape:", x_hat.shape)  # Should be (128, 256)

    vae.plot_image_reconstruction(x_test, n=10, save_path=os.path.join(MNIST_DIR, "mnist_reconstructions_mlp_vae.png"))
    vae.plot_image_samples(n=20, n_rows=4, save_path=os.path.join(MNIST_DIR, "mnist_samples_mlp_vae.png"))


def test_CNN_BaseVAE_mnist():
    """
    Test the CNN-based VAE on the MNIST dataset with downsampling to 16x16

    This function loads the MNIST dataset, 
    initializes a VAE with an CNN architecture,
    trains it for a few epochs, save and load the model,
    and then verify the called methods for reconstruction and sampling
    """
    print_subsection("Testing CNN-based VAE on MNIST")
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=128, downsample=(16, 16), train=True)

    # Define VAE configuration
    cfg = VAEConfig(
        model_type="vae",
        architecture="cnn",  
        input_dim=256,       # 16*16 after downsampling
        hidden_dims=(128, 64),
        latent_dim=32,
        image_channels=1,
        image_size=16,
        kernel_size=4,
        stride=2, 
        padding=1,
        dropout=0.1,
        use_batchnorm=True,
        beta_kl=1.0,
        gamma=0.5,
        learning_rate=1e-3,
        step_size=10,
        weight_decay=1e-5
    )
    print("VAE configuration:")
    print(cfg)
    print()

    # Initialize VAE
    device = "cpu"
    vae = BaseVAE(cfg, device=device)
    print("VAE architecture:")
    print(vae)
    print()

    # Train VAE for a few epochs
    print("Training VAE...")
    metrics = vae.fit(mnist_loader, epochs=5, verbose=True)
    print()
    print("Training metrics:")
    for epoch, metric in enumerate(metrics):
        print(f"Epoch {epoch+1}: Loss={metric.loss:.4f}, Recon Loss={metric.recon:.4f}, KL Div={metric.kld:.4f}")
    print()

    # Save / Load model 
    print("Saving...")
    vae.save(os.path.join(MNIST_DIR, "cnn_vae.pth"))
    cfg.save(os.path.join(MNIST_DIR, "cnn_vae_config.json"))
    print("Model saved to cnn_vae.pth and configuration saved to cnn_vae_config.json")
    print()

    print("Loading...")
    cfg = VAEConfig()
    cfg.load(os.path.join(MNIST_DIR, "cnn_vae_config.json"))
    vae = BaseVAE(cfg, device=device)
    vae.load(os.path.join(MNIST_DIR, "cnn_vae.pth"))
    print("Model loaded from cnn_vae.pth with configuration from cnn_vae_config.json")
    print()

    # Test reconstructions and sampling
    x_test, _ = next(iter(mnist_loader))
    x_test = x_test.to(device)
    print("Test batch shape:", x_test.shape)  # Should be (128, 256)

    x_hat = vae.reconstruct(x_test)
    print("Reconstructed batch shape:", x_hat.shape)  # Should be (128, 256)

    vae.plot_image_reconstruction(x_test, n=10, save_path=os.path.join(MNIST_DIR, "mnist_reconstructions_cnn_vae.png"))
    vae.plot_image_samples(n=20, n_rows=4, save_path=os.path.join(MNIST_DIR, "mnist_samples_cnn_vae.png"))


def test_VQ_VAE_mnist():
    """
    Test the VQ-VAE on the MNIST dataset with downsampling to 16x16

    This function loads the MNIST dataset, 
    initializes a VQ-VAE with a MLP architecture,
    trains it for a few epochs, save and load the model,
    and then verify the called methods for reconstruction and sampling
    """
    print_subsection("Testing VQ-VAE on MNIST")
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=128, downsample=(16, 16), train=True)

    # Define VAE configuration
    cfg = VAEConfig(
        model_type="vqvae",
        architecture="mlp",  
        input_dim=256,       # 16*16 after downsampling
        hidden_dims=(128, 64),
        latent_dim=32,
        num_embeddings=64,
        embedding_dim=32,
        beta_vq=0.25,
        dropout=0.1,
        use_batchnorm=True,
        beta_kl=1.0,
        gamma=0.5,
        learning_rate=1e-3,
        step_size=10,
        weight_decay=1e-5
    )
    print("VAE configuration:")
    print(cfg)
    print()

    # Initialize VAE
    device = "cpu"
    vae = BaseVAE(cfg, device=device)
    print("VAE architecture:")
    print(vae)
    print()

    # Train VAE for a few epochs
    print("Training VAE...")
    metrics = vae.fit(mnist_loader, epochs=5, verbose=True)
    print()
    print("Training metrics:")
    for epoch, metric in enumerate(metrics):
        print(f"Epoch {epoch+1}: Loss={metric.loss:.4f}, Recon Loss={metric.recon:.4f}, KL Div={metric.kld:.4f}, VQ Loss={metric.vq:.4f}")
    print()

    # Save / Load model
    print("Saving...")
    vae.save(os.path.join(MNIST_DIR, "vq_vae.pth"))
    cfg.save(os.path.join(MNIST_DIR, "vq_vae_config.json"))
    print("Model saved to vq_vae.pth and configuration saved to vq_vae_config.json")
    print()

    print("Loading...")
    cfg = VAEConfig()
    cfg.load(os.path.join(MNIST_DIR, "vq_vae_config.json"))
    vae = BaseVAE(cfg, device=device)
    vae.load(os.path.join(MNIST_DIR, "vq_vae.pth"))
    print("Model loaded from vq_vae.pth with configuration from vq_vae_config.json")
    print()

    # Test reconstructions and sampling
    x_test, _ = next(iter(mnist_loader))
    x_test = x_test.to(device)
    print("Test batch shape:", x_test.shape)  # Should be (128, 256)
    x_hat = vae.reconstruct(x_test)
    print("Reconstructed batch shape:", x_hat.shape)  # Should be (128, 256)
    vae.plot_image_reconstruction(x_test, n=10, save_path=os.path.join(MNIST_DIR, "mnist_reconstructions_vq_vae.png"))
    vae.plot_image_samples(n=20, n_rows=4, save_path=os.path.join(MNIST_DIR, "mnist_samples_vq_vae.png"))


def test_fast_cnn_vae_mnist():
    """
    Test the FastCNNVAE on the MNIST dataset with downsampling to 16x16

    This function loads the MNIST dataset, 
    initializes a FastCNNVAE,
    trains it for a few epochs, save and load the model,
    and then verify the called methods for reconstruction and sampling
    """
    print_subsection("Testing FastCNNVAE on MNIST")
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist_loader = load_mnist(batch_size=128, downsample=(16, 16), train=True)

    # Define VAE configuration
    cfg = VAEConfig(
        model_type="fastvae",
        architecture="cnn",  
        input_dim=256,       # 16*16 after downsampling
        hidden_dims=(16, 32),
        latent_dim=32,
        image_channels=1,
        image_size=16,
        kernel_size=3,
        stride=2, 
        padding=1,
        beta_kl=1.0,
        gamma=0.5,
        learning_rate=1e-3,
        step_size=10,
        weight_decay=1e-5
    )
    print("VAE configuration:")
    print(cfg)
    print()

    # Initialize VAE
    device = "cpu"
    vae = FastCNNVAE(cfg, device=device)
    print("VAE architecture:")
    print(vae)
    print()

    # Train VAE for a few epochs
    print("Training VAE...")
    metrics = vae.fit(mnist_loader, epochs=5, verbose=True)
    print()
    print("Training metrics:")
    for epoch, metric in enumerate(metrics):
        print(f"Epoch {epoch+1}: Loss={metric.loss:.4f}, Recon Loss={metric.recon:.4f}, KL Div={metric.kld:.4f}")
    print()

    # Save / Load model
    print("Saving...")
    vae.save(os.path.join(MNIST_DIR, "fast_cnn_vae.pth"))
    cfg.save(os.path.join(MNIST_DIR, "fast_cnn_vae_config.json"))
    print("Model saved to fast_cnn_vae.pth and configuration saved to fast_cnn_vae_config.json")
    print()

    print("Loading...")
    cfg = VAEConfig()
    cfg.load(os.path.join(MNIST_DIR, "fast_cnn_vae_config.json"))
    vae = FastCNNVAE(cfg, device=device)
    vae.load(os.path.join(MNIST_DIR, "fast_cnn_vae.pth"))
    print("Model loaded from fast_cnn_vae.pth with configuration from fast_cnn_vae_config.json")
    print()

    # Test reconstructions and sampling
    x_test, _ = next(iter(mnist_loader))
    x_test = x_test.to(device)
    print("Test batch shape:", x_test.shape)  # Should be (128, 256)
    x_hat = vae.reconstruct(x_test)
    print("Reconstructed batch shape:", x_hat.shape)  # Should be (128, 1, 16, 16)
    vae.plot_image_reconstruction(x_test, n=10, save_path=os.path.join(MNIST_DIR, "mnist_reconstructions_fast_cnn_vae.png"))
    vae.plot_image_samples(n=20, n_rows=4, save_path=os.path.join(MNIST_DIR, "mnist_samples_fast_cnn_vae.png"))


# ===========================
# Testing VAEs on Blobs
# ===========================
def test_MLP_BaseVAE_blobs():
    """
    Test the MLP-based VAE on the Blobs dataset with downsampling to 16x16

    This function loads the Blobs dataset, 
    initializes a VAE with an MLP architecture,
    trains it for a few epochs, save and load the model,
    and then verify the called methods for reconstruction and sampling
    """
    print_subsection("Testing MLP-based VAE on Blobs")
    # Load Blobs dataset
    print("Loading Blobs dataset...")
    blob_tensor = load_blobs(n_samples=1000, n_features=2, centers=3, cluster_std=1, random_state=42)
    blob_dataset = TensorDataset(blob_tensor)
    blob_loader = DataLoader(blob_dataset, batch_size=128, shuffle=True)
    print("Test batch content:", next(iter(blob_loader))[0][0])  # Should be a tensor of shape (2,) with values around the cluster centers

    # Define VAE configuration
    cfg = VAEConfig(
        model_type="vae",
        architecture="mlp",
        reconstruction_loss="mse",   
        input_dim=2,
        hidden_dims=(16, 16),
        latent_dim=2,
        dropout=0.0,
        use_batchnorm=True,
        beta_kl=0.1,
        gamma=0.5,
        learning_rate=1e-4,
        step_size=20,
        weight_decay=1e-5
    )
    print("VAE configuration:")
    print(cfg)
    print()

    # Initialize VAE
    device = "cpu"
    vae = BaseVAE(cfg, device=device)
    print("VAE architecture:")
    print(vae)
    print()

    # Train VAE for a few epochs
    print("Training VAE...")
    metrics = vae.fit(blob_loader, epochs=50, verbose=True)
    print()
    print("Training metrics:")
    for epoch, metric in enumerate(metrics):
        print(f"Epoch {epoch+1}: Loss={metric.loss:.4f}, Recon Loss={metric.recon:.4f}, KL Div={metric.kld:.4f}")
    print()

    # Save / Load model 
    print("Saving...")
    vae.save(os.path.join(BLOB_DIR, "mlp_vae.pth"))
    cfg.save(os.path.join(BLOB_DIR, "mlp_vae_config.json"))
    print("Model saved to mlp_vae.pth and configuration saved to mlp_vae_config.json")
    print()

    print("Loading...")
    cfg = VAEConfig()
    cfg.load(os.path.join(BLOB_DIR, "mlp_vae_config.json"))
    vae = BaseVAE(cfg, device=device)
    vae.load(os.path.join(BLOB_DIR, "mlp_vae.pth"))
    print("Model loaded from mlp_vae.pth with configuration from mlp_vae_config.json")
    print()

    # Test reconstructions and sampling
    x_test = next(iter(blob_loader))[0]
    x_test.to(device)  # Take a batch of 128 samples
    print("Test batch shape:", x_test.shape)  # Should be (128, 2)
    print(x_test[:5])  # Original samples (first 5)

    x_hat = vae.reconstruct(x_test)
    print("Reconstructed batch shape:", x_hat.shape)  # Should be (128, 256)
    print(x_hat[:5])   # Reconstructed samples (first 5)

    vae.plot_blobs_reconstruction(x_test, n=100, save_path=os.path.join(BLOB_DIR, "mnist_reconstructions_mlp_vae.png"))
    vae.plot_blobs_samples(n=100, save_path=os.path.join(BLOB_DIR, "mnist_samples_mlp_vae.png"))


if __name__ == "__main__":
    print_section("Testing different VAE architectures on MNIST dataset")
    test_MLP_BaseVAE_mnist()
    test_CNN_BaseVAE_mnist()
    test_VQ_VAE_mnist()
    test_fast_cnn_vae_mnist()

    print_section("Testing different VAE architectures on Blobs dataset")
    test_MLP_BaseVAE_blobs()

    clear_data_dir()