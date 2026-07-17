# Data
> The `data` module contains utilities for loading datasets, preprocessing data, ensuring reproducibility, and handling common image operations used throughout the generative models project (VAEs, GANs, Diffusion Models, etc.).

## Directory Structure
```text
data/
├── load.py        # Dataset loading
├── utils.py       # General utilities and image processing helpers
```

## `load.py`
Collection of dataset loaders designed to provide a consistent interface across commonly used datasets for model benchmarking.

### Synthetic Blobs
Generate clustered Gaussian data for toy experiments
```py
load_blob(
    n_samples,
    n_features,
    centers,
    cluster_std
)
```

**Parameters**
| Parameter | Description |
|-----------|-------------|
| `n_samples` | Number of samples to generate |
| `n_features` | Feature dimension |
| `centers` | Number or coordinates of cluster centers |
| `cluster_std` | Standard deviation of each cluster |

**Returns**
- PyTorch `Tensor` of shape `(n_samples, n_features)`
- Generated samples without labels (unsupervised)

### MNIST
Loads the handwritten digit dataset (greyscale images of size 28x28) with options for preprocessing
```py
load_mnist(
    batch_size,
    downsample=None,
    normalize=False,
    flatten=True,
    train=True
)
```

**Parameters**
| Parameter | Description |
|-----------|-------------|
| `batch_size` | Number of samples per batch |
| `downsample` | Optional downsampling factor (e.g., (14,14) for 14x14 images) |
| `normalize` | Whether to normalize pixel values to [0, 1] |
| `flatten` | Whether to flatten images into vectors |  
| `train` | Whether to load the training or test split |

**Returns**
- PyTorch `DataLoader` yielding batches of images (and labels if applicable)
- Generally used for training and evaluating generative models on simple image distributions

### Fashion-MNIST
Loads the Fashion-MNIST dataset with the same interface as MNIST.
Ideal for evaluating generative models on slightly more complex image distributions.
```py
load_fashion_mnist(
    batch_size,
    downsample=None,
    normalize=False,
    flatten=True,
    train=True
)
```

### CIFAR-10
Loads the CIFAR-10 natural image dataset
```py
load_cifar10(
    batch_size,
    downsample=None,
    grayscale=False,
    normalize=False,
    flatten=True,
    train=True,
    subset_size=None
)
```

**Parameters** (only the new ones are listed here)
| Parameter | Description |
|-----------|-------------|
| `grayscale` | Whether to convert images to greyscale |
| `subset_size` | Optional limit on the number of samples to load (useful for quick experiments) |

**Returns**
- PyTorch `DataLoader` yielding batches of images (and labels if applicable)
- Suitable for training and evaluating generative models on more complex, real-world image distributions.

## `utils.py`
General utility functions shared across experiments, including reproducibility, tensor manipulation, and image processing and saving utilities.

### `set_seed()` for reproducibility
Sets random seeds across common libraries to ensure deterministic experiments.
This includes:
- Python
- NumPy
- PyTorch
- CUDA (when available)
```py
set_seed(42)
```

### `reparameterize()` : reparameterization trick for sampling latent variables
Samples latent variables using the reparameterization trick employed in Variational Autoencoders.
Instead of directly sampling

\[
z \sim \mathcal{N}(\mu,\sigma^2)
\]

the function computes

\[
z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
\]

allowing gradients to propagate through the sampling operation.


### Tensor Utilities
Convenience functions for converting between common data formats.
Supported conversions include
- NumPy arrays $\leftrightarrow$ PyTorch tensors
- Tensors $\leftrightarrow$ NumPy arrays
- Arrays $\leftrightarrow$ Dataset
- Dataset $\leftrightarrow$ DataLoader
These helpers simplify preprocessing pipelines and reduce boilerplate code.

### Image Utilities
A collection of image manipulation functions frequently used in generative modeling experiments, to provide a common framework for image preprocessing and visualization.

#### Cropping
```py
center_crop(image, crop_h, crop_w)
```
Extracts the centered square region of an image.

#### Resizing
```py
resize_image(image, size)
```
Resizes images while preserving compatibility with PyTorch transforms.

#### Image Transformations
```py
transform_image(resize=(28, 28), crop=True)
```
Applies preprocessing operations such as normalization, resizing, grayscale conversion, and tensor conversion.

### Visualization
Utility plotting functions for quick qualitative evaluation.
Supported visualizations include
- Synthetic blob datasets
- Grayscale images
- RGB images
```py
plot_blob_distribution(real, fake)
plot_images(images, n=16)
```

### Image Grids
Functions for assembling generated samples into a single visualization.
Available helpers include
- `merge_images()`
- `image_grid()`
These are particularly useful for monitoring training progress of
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Diffusion Models

## Summary
| Module | Purpose |
|--------|---------|
| `load.py` | Dataset loading and preprocessing |
| `utils.py` | Reproducibility, tensor utilities, image processing, visualization |

The module provides a unified interface for preparing datasets and manipulating images, making experiments across different generative models simple and consistent.