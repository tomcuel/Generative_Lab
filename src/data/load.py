# ===========================
# Imports
# ===========================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from sklearn.datasets import make_blobs
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple


# ===========================
# Dataset Loaders
# ===========================
def load_blobs(
    n_samples: int = 1000, 
    n_features: int = 2,
    centers: int | list[float] = 3,
    cluster_std: float | list[float] = 1.0,
    random_state: int = 42
) -> torch.Tensor:
    """
    Generate synthetic clustering dataset

    Parameters:
    -----------
    n_samples: int
        Total number of samples to generate
    n_features: int
        Number of features for each sample
    centers: int or array-like
        Number of centers to generate, or the fixed center locations
    cluster_std: float or array-like
        Standard deviation of the clusters
    random_state: int
        Determines random number generation for dataset creation

    Returns:
    --------    
    torch.Tensor
        Generated dataset of shape (n_samples, n_features)

    Usage Example:
    --------------
    >>> X = load_blobs(n_samples=500, centers=3, cluster_std=0.5)
    >>> print(X.shape)  # Output: torch.Size([500, 2])
    >>> print(X[:5])    # Output: tensor of first 5 samples
    """
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )

    return torch.tensor(X, dtype=torch.float32)


def load_mnist(
    batch_size: int = 128,
    downsample: tuple[int, int] | None = None,
    normalize: bool = False, 
    flatten: bool = True,
    train: bool = True
) -> DataLoader:
    """
    Load the MNIST dataset with optional downsampling

    Parameters:
    -----------
    batch_size: int
        Number of samples per batch to load
    downsample: tuple[int, int] | None
        If None, images will be flattened to (28*28,)
        If a tuple (new_h, new_w), images will be resized to (new_h, new_w) and then flattened to (new_h*new_w,)
    normalize: bool
        If True, normalize pixel values to [-1, 1] range (critical for GANs/VAEs); if False, keep pixel values in [0, 1] range
    flatten: bool
        If True, images will be flattened to (H*W,) or (new_h*new_w,); if False, images will be kept in (C, H, W) format
    train: bool
        If True, load the training set; otherwise, load the test set
    
    Returns:
    --------
    DataLoader
        DataLoader for the MNIST dataset with specified transformations

    Usage Example:
    --------------
    >>> mnist_loader = load_mnist(batch_size=64, downsample=(16, 16), train=True)
    >>> for images, labels in mnist_loader:
    ...     print(images.shape)  # Output: torch.Size([64, 256])
    ...     print(labels.shape)  # Output: torch.Size([64])
    ...     break
    """
    transform_list = []
    transform_list.append(transforms.ToTensor())    

    if downsample is not None:
        transform_list.append(transforms.Resize(downsample))

    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        # Normalize((0.5,), (0.5,)) give [-1, 1]

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_fashion_mnist(
    batch_size: int = 128,
    downsample: tuple[int, int] | None = None,
    normalize: bool = False,
    flatten: bool = True,
    train: bool = True
) -> DataLoader:
    """
    Load the FashionMNIST dataset with optional downsampling

    Parameters:
    -----------
    batch_size: int
        Number of samples per batch to load
    downsample: tuple[int, int] | None
        If None, images will be flattened to (28*28,)
        If a tuple (new_h, new_w), images will be resized to (new_h, new_w) and then flattened to (new_h*new_w,)
    normalize: bool
        If True, normalize pixel values to [-1, 1] range (critical for GANs/VAEs); if False, keep pixel values in [0, 1] range
    flatten: bool
        If True, images will be flattened to (H*W,) or (new_h*new_w,); if False, images will be kept in (C, H, W) format
    train: bool
        If True, load the training set; otherwise, load the test set
    
    Returns:
    --------
    DataLoader
        DataLoader for the FashionMNIST dataset with specified transformations

    Usage Example:
    --------------
    >>> fashion_loader = load_fashion_mnist(batch_size=64, downsample=(16, 16), train=True)
    >>> for images, labels in fashion_loader:   
    ...     print(images.shape)  # Output: torch.Size([64, 256])
    ...     print(labels.shape)  # Output: torch.Size([64])
    ...     break
    """
    transform_list = []
    transform_list.append(transforms.ToTensor())

    if downsample is not None:
        transform_list.append(transforms.Resize(downsample))

    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        # Normalize((0.5,), (0.5,)) give [-1, 1]

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    dataset = datasets.FashionMNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_cifar10(
    batch_size: int = 128,
    downsample: Optional[Tuple[int, int]] = None,
    grayscale: bool = False, 
    normalize: bool = False,
    flatten: bool = True,
    train: bool = True
) -> DataLoader:
    """
    Load the CIFAR-10 dataset with optional downsampling, flattening, grayscale conversion, and normalization

    Parameters:
    -----------
    batch_size: int
        Number of samples per batch to load
    downsample: tuple[int, int] | None
        If None, images will be flattened to (H*W,) [if grayscale] or (C*H*W,) [if RGB]
        If a tuple (new_h, new_w), images will be resized to (new_h, new_w) and then flattened to (new_h*new_w,) [if grayscale] or (C*new_h*new_w,) [if RGB]
    grayscale: bool
        If True, convert images to grayscale; otherwise, keep them in RGB
    normalize: bool
        If True, normalize pixel values to [-1, 1] range (critical for GANs/VAEs); if False, keep pixel values in [0, 1] range
    flatten: bool
        If True, images will be flattened to (H*W,) or (C*H*W,); if False, images will be kept in (C, H, W) format
    train: bool
        If True, load the training set; otherwise, load the test set

    Returns:
    --------
    DataLoader
        DataLoader for the CIFAR-10 dataset with specified transformations

    Usage Example:
    --------------
    >>> cifar_loader = load_cifar10(batch_size=64, downsample=(16, 16), train=True, grayscale=True)
    >>> for images, labels in cifar_loader:
    ...     print(images.shape)  # Output: torch.Size([64, 256]) if downsample=(16,16) and grayscale=True
    ...     print(labels.shape)  # Output: torch.Size([64])
    ...     break
    """
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale())
        channels = 1
    else:
        channels = 3

    transform_list.append(transforms.ToTensor())

    if downsample is not None:
        transform_list.append(transforms.Resize(downsample))

    if normalize:
        transform_list.append(transforms.Normalize((0.5,) * channels, (0.5,) * channels))
        # Normalize((0.5,) * channels, (0.5,) * channels) give [-1, 1]

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# === FILE: NRT/NRT_load/test.py ===