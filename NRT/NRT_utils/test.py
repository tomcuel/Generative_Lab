# ===========================
# Path setup
# ===========================
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ===========================
# Imports
# ===========================
import numpy as np
import torch
from src.data.utils import (
    set_seed,
    reparameterize,
    to_numpy,
    df_to_tensor_dataset,
    make_dataloaders,
    center_crop,
    resize_image,
    transform_image,
    merge_images,
    save_image_grid
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================
# Reproducibility tests
# ===========================
def test_seed():
    set_seed(123)
    a = torch.rand(5)

    set_seed(123)
    b = torch.rand(5)

    assert torch.allclose(a, b)
    print("set_seed reproducibility OK")


# ===========================
# Tensor utilities
# ===========================
def test_reparameterization():
    mu = torch.randn(5, 10)
    logvar = torch.randn(5, 10)
    z = reparameterize(mu, logvar)  # Should sample from N(mu, sigma^2)

    assert z.shape == (5, 10)
    assert not torch.isnan(z).any()
    print("reparameterization OK")


def test_to_numpy():
    x = torch.tensor([[1, 2], [3, 4]])
    arr = to_numpy(x)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, 2)

    print("to_numpy conversion OK")


def test_tensor_dataset():
    X = np.random.randn(100, 10)
    dataset = df_to_tensor_dataset(X)

    assert len(dataset) == 100
    assert dataset.tensors[0].shape == (100, 10)

    print("df_to_tensor_dataset OK")


def test_dataloader():
    X = np.random.randn(100, 10)
    dataset = df_to_tensor_dataset(X)
    loader = make_dataloaders(dataset, batch_size=32)
    batch = next(iter(loader))

    assert batch[0].shape == (32, 10)
    print("DataLoader creation OK")


# ===========================
# Image utilities
# ===========================
def test_center_crop():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cropped = center_crop(img, 64, 64)

    assert cropped.shape == (64, 64, 3)
    print("center_crop OK")


def test_resize():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    resized = resize_image(img, (32, 32))

    assert resized.shape == (32, 32, 3)
    print("resize_image OK")


def test_transform():
    img = torch.randint(0, 255, (3, 100, 100), dtype=torch.uint8)
    transformed = transform_image(img, resize=(64, 64), crop=False)

    assert transformed.shape[0] == 64
    assert transformed.shape[1] == 64
    assert transformed.max() <= 1.0

    print("transform_image OK")


# ===========================
# Grid utilities
# ===========================
def test_merge_images():
    images = np.random.rand(16, 28, 28)
    grid = merge_images(images, 4, 4)

    assert grid.shape == (28 * 4, 28 * 4)
    print("merge_images OK")


def test_save_grid():
    images = np.random.rand(9, 28, 28)
    path = os.path.join(OUTPUT_DIR, "grid.png")
    save_image_grid(images, path, 3, 3)

    assert os.path.exists(path)
    print("save_image_grid OK")


# ===========================
# Run all tests
# ===========================
if __name__ == "__main__":
    test_seed()

    test_reparameterization()
    test_to_numpy()
    test_tensor_dataset()
    test_dataloader()

    test_center_crop()
    test_resize()
    test_transform()

    test_merge_images()
    test_save_grid()

    print("\nAll NRT utils tests passed")