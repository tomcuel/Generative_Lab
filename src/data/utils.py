# ===========================
# Imports
# ===========================
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# ===========================
# Reproducibility
# ===========================
def set_seed(
    seed: int = 42
) -> None:
    """
    Set all major random seeds for reproducibility

    Parameters:
    -----------
    seed: int
        The seed value to set for all random number generators

    Returns:
    --------
    None

    Usage Example:
    --------------
    >>> set_seed(123)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================
# Tensor Utilities
# ===========================
def reparameterize(
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, sigma^2) using N(0, 1)

        Parameters:
        -----------
        mu: torch.Tensor
            Mean tensor of shape (batch_size, latent_dim)
        logvar: torch.Tensor
            Log-variance tensor of shape (batch_size, latent_dim)

        Returns:
        --------
        torch.Tensor
            Sampled tensor of shape (batch_size, latent_dim) drawn from N(mu, sigma^2)

        Usage Example:
        --------------
        >>> mu = torch.zeros(16, 32)  # mean tensor
        >>> logvar = torch.zeros(16, 32)  # log-variance tensor
        >>> z = reparameterize(mu, logvar)
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std  


def to_numpy(
    x: torch.Tensor | np.ndarray
) -> np.ndarray:
    """
    Convert tensor to numpy safely

    Parameters:
    -----------
    x: torch.Tensor or np.ndarray
        Input tensor or array to convert

    Returns:
    --------
    np.ndarray
        Converted numpy array

    Usage Example:
    --------------
    >>> x = torch.tensor([[1, 2], [3, 4]])
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def df_to_tensor_dataset(
    X: np.ndarray
) -> TensorDataset:
    """
    Convert numpy arrays to PyTorch TensorDataset

    Parameters
    ----------
    X : np.ndarray
        Feature data

    Returns
    -------
    TensorDataset
        PyTorch TensorDataset containing features and labels

    Usage Example
    --------------
    dataset = df_to_tensor_dataset(X_train)
    """
    X_tensor = torch.tensor(X, dtype=torch.float)
    return TensorDataset(X_tensor)


def make_dataloaders(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False
) -> DataLoader:
    """
    Create DataLoader from Dataset
    
    Parameters
    ----------
    dataset : Dataset
        PyTorch Dataset
    batch_size : int
        Batch size for DataLoader
    shuffle : bool
        Whether to shuffle the data

    Returns
    -------
    DataLoader
        PyTorch DataLoader

    Usage Example
    --------------
    dataloader = make_dataloaders(dataset, batch_size=32, shuffle=True)
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ===========================
# Image Utilities
# ===========================
def center_crop(
    img: np.ndarray,
    crop_h: int,
    crop_w: int
) -> np.ndarray:
    """
    Center crop an image to the specified height and width

    Parameters:
    img: np.ndarray
        Input image array of shape (H, W) or (H, W, C)
    crop_h: int
        Desired crop height
    crop_w: int
        Desired crop width

    Returns:
    np.ndarray
        Center cropped image array of shape (crop_h, crop_w) or (crop_h, crop_w, C)
    
    Usage Example:
    --------------
    >>> img = np.random.rand(100, 100, 3)  # Example
    >>> cropped = center_crop(img, 64, 64)
    >>> print(cropped.shape)  # Output: (64, 64, 3)
    """
    h, w = img.shape[:2]
    if crop_h > h or crop_w > w:
        raise ValueError("Crop size larger than image")

    j = int((h - crop_h) / 2)
    i = int((w - crop_w) / 2)
    return img[j:j + crop_h, i:i + crop_w]


def resize_image(
    img: np.ndarray,
    size: tuple[int, int]
) -> np.ndarray:
    """
    Resize an image to the specified size using PIL

    Parameters:
    img: np.ndarray
        Input image array of shape (H, W) or (H, W, C)
    size: tuple[int, int]
        Desired output size (new_h, new_w)

    Returns:
    np.ndarray
        Resized image array of shape (new_h, new_w) or (new_h, new_w, C)

    Usage Example:
    --------------
    >>> img = np.random.rand(100, 100, 3)  # Example
    >>> resized = resize_image(img, (64, 64))
    >>> print(resized.shape)  # Output: (64, 64, 3)
    """
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return np.array(Image.fromarray(img).resize(size, Image.BILINEAR))


def transform_image(
    img: torch.Tensor | np.ndarray,
    resize: tuple[int, int] = (28, 28),
    crop: bool = False
) -> np.ndarray:
    """
    Transform an image by optionally cropping and resizing it

    Parameters:
    img: torch.Tensor or np.ndarray
        Input image tensor or array of shape (C, H, W) or (H, W) or (H, W, C)
    resize: tuple[int, int]
        Desired output size (new_h, new_w)
    crop: bool
        Whether to center crop the image before resizing

    Returns:
    np.ndarray
        Transformed image array of shape (new_h, new_w) or (new_h, new_w, C)

    Usage Example:
    --------------
    >>> img = torch.rand(3, 100, 100)  # Example image tensor
    >>> transformed = transform_image(img, resize=(64, 64), crop=True)  
    >>> print(transformed.shape)  # Output: (64, 64, 3)
    """
    img = to_numpy(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if crop:
        h, w = resize
        img = center_crop(img, h, w)

    img = resize_image(img, resize)
    img = img.astype(np.float32) / 255.0
    return img


# ===========================
# Visualization Utilities
# ===========================
def plot_images(
    images: torch.Tensor | np.ndarray, 
    n: int = 16,
    title: str = "Images"
) -> None:
    """
    Plot a batch of images in a grid

    Parameters:
    -----------
    images: torch.Tensor or np.ndarray
        Batch of images to plot, shape (N, H, W) or (N, H, W, C)
    n: int
        Number of images to plot
    title: str
        Title for the plot

    Returns:
    --------
    None

    Usage Example:
    --------------
    >>> plot_images(images, n=16, title="Sample Images")
    """
    images = to_numpy(images)
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))

    for i in range(n):
        img = images[i]
        if img.ndim == 2:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(img)
        axes[i].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ===========================
# Grid/Merge Utilities
# ===========================
def merge_images(
    images : torch.Tensor | np.ndarray,
    n_row: int | None = None,
    n_col: int | None = None
) -> np.ndarray:
    """
    Merge a batch of images into a single grid image

    Parameters:
    -----------
    images: torch.Tensor or np.ndarray
        Batch of images to merge, shape (N, H, W) or (N, H, W, C)
    n_row: int or None
        Number of rows in the grid, if None it will be auto-calculated
    n_col: int or None
        Number of columns in the grid, if None it will be auto-calculated

    Returns:
    --------
    np.ndarray
        Merged grid image

    Usage Example:
    --------------
    >>> merged = merge_images(images, n_row=4, n_col=5)
    >>> print(merged.shape)  # Output: (H*4, W*5, C)
    """
    images = to_numpy(images)
    N = len(images)

    if images.ndim == 3:
        images = images[..., None]

    N, H, W, C = images.shape

    if n_row is None:
        n_row = int(np.sqrt(N))

    if n_col is None:
        n_col = int(np.ceil(N / n_row))

    grid = np.zeros((H * n_row, W * n_col, C))
    for idx, img in enumerate(images):
        row = idx // n_col
        col = idx % n_col
        grid[
            row * H:(row + 1) * H,
            col * W:(col + 1) * W
        ] = img

    return grid.squeeze()


def save_image_grid(
    images: torch.Tensor | np.ndarray,
    path: str,
    n_row: int | None = None,
    n_col: int | None = None
) -> None:
    """
    Save a grid of images to disk

    Parameters:
    -----------
    images: torch.Tensor or np.ndarray
        Batch of images to save, shape (N, H, W) or (N, H, W, C)
    path: str
        File path to save the image grid
    n_row: int or None
        Number of rows in the grid, if None it will be auto-calculated
    n_col: int or None
        Number of columns in the grid, if None it will be auto-calculated

    Returns:
    --------
    None

    Usage Example:
    --------------
    >>> save_image_grid(images, "output/grid.png", n_row=4, n_col=5)
    """
    grid = merge_images(images, n_row, n_col)
    grid = np.clip(grid, 0, 1)
    img = Image.fromarray((grid * 255).astype(np.uint8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# === FILE: NRT/NRT_utils/test.py ===