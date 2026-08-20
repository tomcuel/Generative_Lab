# ===========================
# Path Setup
# ===========================
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ===========================
# Imports
# ===========================
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.data.load import (
    load_blobs,
    load_mnist,
    load_fashion_mnist,
    load_cifar10
)
from src.data.utils import set_seed


RANDOM_SEED = 42

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BLOB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "blobs")
os.makedirs(BLOB_OUTPUT_DIR, exist_ok=True)
MNIST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mnist")
os.makedirs(MNIST_OUTPUT_DIR, exist_ok=True)
FASHION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "fashion_mnist")
os.makedirs(FASHION_OUTPUT_DIR, exist_ok=True)
CIFAR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cifar10")
os.makedirs(CIFAR_OUTPUT_DIR, exist_ok=True)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def clear_data_dir():
    os.system(f"rm -rf {DATA_DIR}/*")
    os.system(f"rm -rf {DATA_DIR}")


# ===========================
# Helper: save image grid
# ===========================
def save_image_grid(images, path, n=16):
    images = images[:n]
    n_row = int(np.sqrt(n))
    n_col = int(np.ceil(n / n_row))

    fig, axes = plt.subplots(n_row, n_col, figsize=(6,6))

    for i in range(n_row*n_col):
        ax = axes[i//n_col, i%n_col]
        if i < len(images):
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def assert_cifar10_class_names(class_names):
    expected = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    assert isinstance(class_names, list)
    assert class_names == expected

# ===========================
# NRT Tests
# ===========================
def test_blobs():
    X = load_blobs(n_samples=500)

    assert isinstance(X, torch.Tensor)
    assert X.shape == (500,2)

    X = X.numpy()
    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=10)
    plt.title("Blobs dataset")
    plt.savefig(os.path.join(BLOB_OUTPUT_DIR,"blobs.png"))
    plt.close()

    print("Blobs dataset OK")


def test_mnist():
    loader = load_mnist(batch_size=64)
    X, y = next(iter(loader))

    assert X.shape == (64,784)

    images = X.view(-1,28,28).numpy()
    save_image_grid(images, os.path.join(MNIST_OUTPUT_DIR,"mnist_samples.png"), 16)

    print("MNIST dataset OK")


def test_mnist_downsample():
    loader = load_mnist(batch_size=32, downsample=(14,14))
    X, y = next(iter(loader))

    assert X.shape == (32,196)

    images = X.view(-1,14,14).numpy()
    save_image_grid( images, os.path.join(MNIST_OUTPUT_DIR,"mnist_downsample.png"), 16)

    print("MNIST downsample OK")


def test_fashion_mnist():
    loader = load_fashion_mnist(batch_size=64)
    X, y = next(iter(loader))

    assert X.shape == (64,784)

    images = X.view(-1,28,28).numpy()
    save_image_grid(images, os.path.join(FASHION_OUTPUT_DIR,"fashion_samples.png"), 16)

    print("Fashion-MNIST dataset OK")


def test_cifar10():
    loader = load_cifar10(
        batch_size=32, 
        downsample=(16,16), 
        grayscale=True,
        normalize=False,
        flatten=True, 
        subset_size=100
    )
    X, y = next(iter(loader))

    # Verify shape
    assert X.shape == (32, 16*16)
    assert y.shape == (32,)
    assert_cifar10_class_names(loader.class_names)
    # Show a few label values
    print("Sample labels:", y[:5].numpy())

    images = X.view(-1,16,16).numpy()
    save_image_grid(images, os.path.join(CIFAR_OUTPUT_DIR,"cifar_grayscale_flattened.png"), 16)

    print("CIFAR10 Grayscale Flattened OK")

    loader = load_cifar10(
        batch_size=32, 
        downsample=(32,32), 
        flatten=True,
        grayscale=False, 
        normalize=False,
        subset_size=100
    )
    X, y = next(iter(loader))

    # Verify shape 
    assert X.shape == (32, 3*32*32)
    assert y.shape == (32,)
    assert_cifar10_class_names(loader.class_names)
    # Verify pixel value range
    assert X.min() >= 0.0 and X.max() <= 1.0
    
    images = X.view(-1,3,32,32).permute(0,2,3,1).numpy()
    save_image_grid(images, os.path.join(CIFAR_OUTPUT_DIR,"cifar_rgb_flattened.png"), 16)
    
    print("CIFAR10 RGB Flattened OK")

    loader = load_cifar10(
        batch_size=32, 
        downsample=(16,16), 
        flatten=False,
        grayscale=False, 
        normalize=True, # normalize to [-1, 1] so plots are darker if output is return back to original range for visualization
        subset_size=100
    )
    X, y = next(iter(loader))

    # Verify shape
    assert X.shape == (32, 3, 16, 16)
    assert y.shape == (32,)
    assert_cifar10_class_names(loader.class_names)
    # Verify pixel value range    assert X.min() >= -1.0 and X.max() <= 1.0
    assert X.min() >= -1.0 and X.max() <= 1.0

    images = X.permute(0,2,3,1).numpy()
    save_image_grid(images, os.path.join(CIFAR_OUTPUT_DIR,"cifar_rgb_normalized.png"), 16)

    print("CIFAR10 RGB Normalized OK")


# ===========================
# Run NRT
# ===========================
if __name__ == "__main__":
    set_seed(RANDOM_SEED)

    test_blobs()

    test_mnist()
    test_mnist_downsample()

    test_fashion_mnist()

    test_cifar10()

    clear_data_dir()
    print("\nAll dataset NRT tests passed")