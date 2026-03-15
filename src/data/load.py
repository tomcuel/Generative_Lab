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


def _flatten_or_downsample(
    x: torch.Tensor,
    downsample: tuple[int, int] | None
) -> torch.Tensor:
    """
    Helper transform function to either flatten the image or downsample it

    Parameters:
    -----------
    x: torch.Tensor
        Input image tensor of shape (C, H, W)
    downsample: tuple[int, int] | None
        If None, the image will be flattened to (C*H*W,)
        If a tuple (new_h, new_w), the image will be resized to (C, new_h, new_w) and then flattened to (C*new_h*new_w,)
    
    Returns:
    --------
    torch.Tensor
        Transformed image tensor of shape (C*H*W,) or (C*new_h*new_w,)

    Usage Example:
    --------------
    >>> img = torch.rand(3, 32, 32)  # Example image
    >>> flat_img = _flatten_or_downsample(img, None)
    >>> print(flat_img.shape)  # Output: torch.Size([3072])
    >>> downsampled_img = _flatten_or_downsample(img, (16, 16))
    >>> print(downsampled_img.shape)  # Output: torch.Size([768])
    """
    if downsample is None:
        return torch.flatten(x)
    else:
        x = F.interpolate(x.unsqueeze(0), size=downsample).squeeze(0)
        return torch.flatten(x)


def load_mnist(
    batch_size: int = 128,
    downsample: tuple[int, int] | None = None,
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: _flatten_or_downsample(x, downsample))
    ])

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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: _flatten_or_downsample(x, downsample))
    ])

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
    flatten: bool = True,
    train: bool = True,
    grayscale: bool = False, 
    normalize: bool = False
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
    flatten: bool
        If True, images will be flattened to (C*H*W,) or (C*new_h*new_w,); if False, images will be kept in (C, H
    train: bool
        If True, load the training set; otherwise, load the test set
    grayscale: bool
        If True, convert images to grayscale; otherwise, keep them in RGB
    normalize: bool
        If True, normalize pixel values to [-1, 1] range (critical for GANs/VAEs); if False, keep pixel values in [0, 1] range

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

    if normalize:
        transform_list.append(
            transforms.Normalize(
                (0.5,) * channels,
                (0.5,) * channels
            )
        )

    transform = transforms.Compose(transform_list)

    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    def collate_fn(batch): # Custom collate function to handle downsampling and flattening
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs)
        if downsample is not None:
            imgs = F.interpolate(imgs, size=downsample)
        if flatten:
            imgs = imgs.view(imgs.size(0), -1)

        return imgs, torch.tensor(labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


# === FILE: NRT/NRT_load/test.py ===