# Non-Regression Tests (NRT)
> The **NRT** (Non-Regression Tests) module provides a comprehensive validation framework for the entire project. Its purpose is to ensure that new features, code refactoring, optimization, or dependency updates do not alter the expected behavior of implemented models and utilities.
>
>Unlike unit tests, which verify individual functions in isolation, non-regression tests validate complete workflows—from data loading to training, inference, serialization, and image generation—ensuring that every component of the project remains reliable over time.


# Directory Structure
```text
NRT/
├── NRT_GANs/
├── NRT_VAEs/
├── NRT_diffusion_models/
├── NRT_inference/
├── NRT_pre_trained/
├── NRT_load/
├── NRT_utils/
└── test.py
```


# Testing Philosophy
The testing framework follows three main principles:
- **Functional correctness**: every public API behaves as expected
- **Reproducibility**: identical configurations produce deterministic results
- **Regression detection**: existing features remain unchanged after code modifications

Each test suite is completely independent, allowing failures to be traced back to a specific module while avoiding interference between successive executions.


# Test Launcher
The entry point of the testing framework is
```text
python test.py
```
The launcher automatically:
1. Discovers every NRT test suite
2. Executes each suite independently
3. Clears temporary files, GPU memory, caches, and generated artifacts between tests
4. Reports failures immediately while continuing with the remaining suites when possible

This isolation guarantees reproducible executions and prevents interactions between successive tests.


# Test Suites

## [NRT_GANs](./NRT_GANs)
Validates every implemented **Generative Adversarial Network (GAN)** architecture.

### Covered architectures
- MLP-GAN
- DCGAN
- Conditional GAN (CGAN)
- Unrolled GAN
- Wasserstein GAN (WGAN)
- StyleGAN

## Tested datasets
- MNIST
- CIFAR10

### Typical validations
- Generator initialization
- Discriminator initialization
- Forward propagation
- Loss computation
- Training step
- Short training sessions
- Sample generation
- Output shape consistency
- Generated image validity
- Model serialization (save/load)

## [NRT_VAEs](./NRT_VAEs)
Validates every implemented **Variational Autoencoder (VAE)** architecture.

### Covered architectures
- MLP-VAE
- CNN-VAE
- FastCNNVAE
- VQ-VAE

### Tested datasets
- Synthetic Blobs
- MNIST
- CIFAR-10

### Typical validations
- Encoder and decoder forward pass
- Latent sampling through the reparameterization trick
- Reconstruction pipeline
- ELBO loss computation
- Vector quantization (VQ-VAE)
- Training loop
- Reconstruction quality
- Sample generation
- Output dimensions
- Model serialization (save/load)

## [NRT_diffusion_models](./NRT_diffusion_models)
Validates every implemented **Diffusion Model** architecture.

### Covered components
- Noise Scheduler
- Time Embeddings
- CNN backbone
- Residual U-Net
- Latent Autoencoder
- EMA sampling

### Tested datasets
- MNIST
- CIFAR-10

### Typical validations
- Latent diffusion
- EMA parameter updates
- Noise prediction
- Forward diffusion process
- Reverse denoising process through sampling (DDPM and DDIM)
- Conditional generation
- Training loop
- Generated image consistency
- Model serialization (save/load)

## [NRT_inference](./NRT_inference)
Validates the complete inference pipelines of all supported pre-trained models.

Rather than testing individual neural networks, this suite validates the **end-to-end generation workflow**.

Typical checks include:
- Pipeline initialization
- Prompt processing if compatible
- Command-line argument parsing
- Generated image validity
- Output reproducibility
- Saving generated images
- Saving and reloading pipelines / model parameters

Currently supported pipelines include:
- **DDPM Pipeline**
- **Stable Diffusion**

## [NRT_fine_tuning](./NRT_fine_tuning)
Validates the fine-tuning workflow combining the pre-trained model weights and my own implemented training loops (similar to my own model architectures).

Typical checks include:
- Loading pretrained checkpoints
- Model architecture compatibility
- Dataset preparation
- Parameter freezing/unfreezing
- Optimizer configuration and learning rate scheduling
- Resume training
- Inference after fine-tuning

## [NRT_launcher](./NRT_launcher)
Validate the overall launcher of the project, ensuring that the main entry point correctly orchestrates the execution of all components, from data loading, model configuration, training, inference, results saving, and logging.

## [NRT_load](./NRT_load)
Validates every dataset loader implemented in the project.

Typical checks include:
- Dataset download
- Train/test split
- Normalization
- Downsampling
- Flattening
- Grayscale conversion
- Batch generation
- DataLoader consistency
- Subset loading
- Reproducibility

## [NRT_utils](./NRT_utils)
VValidates all shared utility functions.

Covered utilities include:
- Random seed initialization
- Tensor conversions
- Dataset helpers
- DataLoader helpers
- Image preprocessing
- Image resizing
- Center cropping
- Image transformations
- Visualization utilities
- Image grids
- Image merging


# Continuous Validation
The NRT framework is designed to be executed:
- before every release
- after adding a new model
- after dependency upgrades
- after major refactoring
- during continuous integration (CI)

Because generative models often evolve rapidly, this validation layer provides confidence that improvements do not introduce subtle behavioral changes, or break existing functionalities.

# Why Non-Regression Tests Matter
The NRT framework helps guarantee that the project remains stable, reproducible, and maintainable as it grows.

It ensures:
- API compatibility across versions
- Stable training behavior
- Deterministic experiments
- Consistent inference pipelines
- Reliable checkpoint loading
- Early detection of breaking changes
- Long-term maintainability of the codebase
- Reproducibility of generated samples and experimental results