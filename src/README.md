# Source Code (`src`)
> The `src` directory contains the complete implementation of the project. It is organized into independent modules responsible for **data processing**, **model architectures**, **pre-trained pipelines**, and **experiment management**, providing a modular and extensible framework for research in deep generative models.


# Directory Structure
```text
src/
├── data/
│   ├── load.py                # Dataset loading and preprocessing
│   └── utils.py               # Utility and image processing functions
│
├── models/
│   ├── VAEs.py                # Variational Autoencoders
│   ├── GANs.py                # Generative Adversarial Networks
│   └── diffusion_models.py    # Diffusion Models
│
├── pretrained/
│   ├── inference.py           # Pre-trained inference pipelines
│   └── fine_tuning.py         # Fine-tuning pipelines
│
└── launcher.py                # Unified training and experiment launcher
```
```
                launcher.py
                     │
        ┌────────────┼────────────┐
        │            │            │
     data/       models/    pretrained/
        │            │            │
        └───────► Training ◄──────┘
                     │
               Generated Images
```


# Module Overview
The project is divided into four independent components, each responsible for a specific stage of the generative modeling workflow.

| Module | Purpose |
|---------|---------|
| **`data/`** | Dataset loading, preprocessing, visualization, and reproducibility utilities |
| **`models/`** | Implementation of all generative model architectures |
| **`pretrained/`** | High-level inference pipelines and fine-tuning tools for pretrained models |
| **`launcher.py`** | Entry point for training, evaluation, and experimentation |


# [data](./data)
The **data** module provides a unified interface for preparing datasets used throughout the project.

Its responsibilities include:
- loading synthetic and real-world datasets
- preprocessing and normalization
- image transformations
- reproducibility utilities
- visualization helpers

Supported datasets include:
- Synthetic Gaussian Blobs
- MNIST
- Fashion-MNIST
- CIFAR-10

The module exposes PyTorch-ready `Dataset` and `DataLoader` objects that can be directly consumed by every implemented model.


# [models](./models)
The **models** module contains the complete implementation of the project's generative architectures.
Implemented families include:

| Family | Implementations |
|--------|-----------------|
| **Variational Autoencoders** | MLP-VAE, CNN-VAE, FastCNNVAE, VQ-VAE |
| **Generative Adversarial Networks** | GAN, DCGAN, CGAN, WGAN, Unrolled GAN, StyleGAN |
| **Diffusion Models** | CNN, Residual U-Net, DDPM, DDIM, Latent Diffusion |

Although these models rely on different learning paradigms, they share a common API exposing methods such as:
```python
fit()
sample()
save()
load()
```
allowing experiments to be performed with minimal code changes.


# [pretrained](./pretrained)
The **pretrained** module provides high-level interfaces for working with large pre-trained generative models.
It includes:
- loading pretrained checkpoints
- image generation
- prompt-based inference
- model fine-tuning
- integration with the training loop of `models/` implementations
- checkpoint management

The objective is to expose production-ready pipelines while keeping the underlying implementation modular, on top of comparing the architectures of the pre-trained models with my own implementations.


# `launcher.py`
`launcher.py` serves as the main entry point for experiments.
Using a single launcher makes experiments reproducible while providing a consistent interface across every implemented generative model.
It centralizes:
- configuration management
- model initialization
- dataset (and/or model) loading
- training if necessary
- evaluation if training 
- checkpoint saving
- sample generation

The source code is organized around several guiding principles:

- **Modularity**: each component can be used independently
- **Consistency**: similar APIs are shared across different model families
- **Extensibility**: new architectures can be integrated with minimal changes
- **Reproducibility**: centralized configurations and deterministic utilities ensure repeatable experiments
- **Research-oriented**: implementations prioritize readability and experimentation while remaining efficient enough for practical use

This organization makes the framework suitable both as an educational resource for understanding modern generative models and as a foundation for developing and evaluating new architectures.