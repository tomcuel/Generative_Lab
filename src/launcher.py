# =======================================
# Library Imports
# =======================================
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path
import subprocess
import sys 
import shlex
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Literal
import yaml


# =======================================
# Important Path Setup
# =======================================
from src.data.load import (
    load_blobs,
    load_mnist,
    load_fashion_mnist,
    load_cifar10
)
from src.data.utils import (
    set_seed,
    df_to_tensor_dataset,
    make_dataloaders,
    center_crop, 
    resize_image,
    transform_image,
    plot_blob_distribution,
    plot_images,
    save_image_grid
)
from src.models.diffusion_models import (
    DiffusionConfig, 
    DiffusionModel
)
from src.models.GANs import (
    GANConfig,
    GAN
)
from src.models.VAEs import (
    VAEConfig, 
    BaseVAE, 
    FastCNNVAE
)


# ===========================
# Helpers
# ===========================
def print_section(title):
    print("\n\n" + "=" * 50)
    print(title)
    print("=" * 50)

def print_subsection(title):
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)

def clear_data_dir(data_dir):
    os.system(f"rm -rf {data_dir}/*")
    os.system(f"rm -rf {data_dir}")

def build_launch_command(args):
    command = [sys.executable, os.path.join(PROJECT_ROOT, "src", "pretrained", "launch.py")]
    for key, value in args.items():
        command.append(f"--{key}")
        if isinstance(value, (list, tuple)):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command


# =======================================
# Argument Parser
# =======================================
def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


parser = argparse.ArgumentParser(description="Generative Lab - Project Launcher")
is_nrt = False
# -----------------------------------------------------------------------
# General arguments
# -----------------------------------------------------------------------
parser.add_argument("--launch_mode",
                    type=str,
                    choices=["vae", "gan", "diffusion", "inference", "finetuning"],
                    default=None,
                    help="Mode to launch the project in (vae, gan, diffusion, inference, finetuning)")
parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Random seed for reproducible results")
parser.add_argument("--device",
                    type=str,
                    default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Device to run inference on (auto selects cuda if available)")
parser.add_argument("--name",
                    type=str,
                    default="default_experiment",
                    help="Name of the experiment (used for saving outputs, configurations, models, etc.)")

# Dataset
parser.add_argument("--dataset",
                    type=str,
                    choices=["cifar10", "fashion_mnist", "imagefolder", "mnist"],
                    default="none",
                    help="Dataset to use for training (cifar10, fashion_mnist, imagefolder, mnist)")
parser.add_argument("--downsample_size",
                    type=int,
                    nargs="+",
                    default=None,
                    help="Downsample size for the images (ex : 16x16 for CIFAR10 instead of 32x32)")
parser.add_argument("--grayscale",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=True,
                    help="If set, will convert images to grayscale")
parser.add_argument("--normalize",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will normalize images to [-1, 1]")
parser.add_argument("--flatten",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=True,
                    help="If set, will flatten images to 1D vectors")
parser.add_argument("--subset_size",
                    type=int,
                    default=None,
                    help="Subset size for training (only for mnist and cifar10)")

# Model (VAE, GAN, Diffusion) specific 
parser.add_argument("--is_training",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will run the training loop")
parser.add_argument("--n_sample",
                    type=int,
                    default=16,
                    help="Number of samples to generate during training for visualization")

# Saving
parser.add_argument("--show_architecture",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will print the architecture of the model")
parser.add_argument("--save_model",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will save the model parameters for future reuse without re-downloading the whole thing")
parser.add_argument("--model_name",
                    type=str,
                    default=None,
                    help="Name of the model to save/load weights for inference of those 'from scratch' models (vae, gan, diffusion) or the pretrained models (inference, finetuning)")

# Diffusion Scheduler
parser.add_argument("--timesteps",
                    type=int,
                    default=1000,
                    help="Number of diffusion timesteps")
parser.add_argument("--beta_schedule",
                    type=str,
                    default="linear",
                    choices=["linear", "cosine"],
                    help="Noise schedule used by the diffusion process")
parser.add_argument("--beta_start",
                    type=float,
                    default=1e-4,
                    help="Start value of the beta schedule")
parser.add_argument("--beta_end",
                    type=float,
                    default=0.02,
                    help="End value of the beta schedule")
parser.add_argument("--cosine_s",
                    type=float,
                    default=0.008,
                    help="Small offset used in the cosine beta schedule")

# Training
parser.add_argument("--training_batch_size",
                    type=int,
                    default=16,
                    help="Batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=1,
                    help="Number of epochs for training")
parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-5,
                    help="Learning rate for training")
parser.add_argument("--step_size",
                    type=int,
                    default=20,
                    help="Step size for learning rate scheduler in training")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-2,
                    help="Weight decay for training")

# Inference and Fine-tuning 
# Sampling 
parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help="Number of images to generate in a batch = per prompt for baseline")
parser.add_argument("--height",
                    type=int,
                    default=256,
                    help="Height of the generated images")
parser.add_argument("--width",
                    type=int,
                    default=256,
                    help="Width of the generated images")
parser.add_argument("--num_inference_steps",
                    type=int,
                    default=30,
                    help="Number of inference steps for image generation")
parser.add_argument("--guidance_scale",
                    type=float,
                    default=7.5,
                    help="Guidance scale for image generation (default: 7.5, no guidance)")

# -----------------------------------------------------------------------
# VAEs arguments
# -----------------------------------------------------------------------
parser.add_argument("--vae_config", 
                    type=str, 
                    default=None, 
                    help="Name to the VAE configuration file (e.g. vae_config for data/configs/vae_config.yaml). If not provided, CLI arguments will be used to create the VAEConfig.")
parser.add_argument("--vae_model_type", 
                    type=str, 
                    default="vae", 
                    choices=["vae", "vqvae", "fastvae"], 
                    help="Type of VAE model to use")
parser.add_argument("--vae_architecture", 
                    type=str, 
                    default="mlp", 
                    choices=["mlp", "cnn"], 
                    help="Architecture of the VAE model")
parser.add_argument("--vae_reconstruction_loss",
                    type=str,
                    default="bce",
                    choices=["mse", "bce"],
                    help="Reconstruction loss for the VAE model")
parser.add_argument("--vae_input_dim",
                    type=int,
                    default=784,
                    help="Input dimension for MLP VAE (flattened image size)")
parser.add_argument("--vae_hidden_dims",
                    type=int,
                    nargs="+",
                    default=[128, 64],
                    help="Hidden dimensions for MLP VAE (list of integers)")
parser.add_argument("--vae_latent_dim",
                    type=int,
                    default=32,
                    help="Latent dimension for the VAE model")
# for CNN
parser.add_argument("--vae_image_channels",
                    type=int,
                    default=1,
                    help="Number of image channels for CNN VAE")
parser.add_argument("--vae_image_size",
                    type=int,
                    default=28,
                    help="Image size for CNN VAE (assumes square images)")
parser.add_argument("--vae_kernel_size",
                    type=int,
                    default=4,
                    help="Kernel size for CNN VAE")
parser.add_argument("--vae_stride",
                    type=int,
                    default=2,
                    help="Stride for CNN VAE")
parser.add_argument("--vae_padding",
                    type=int,
                    default=1,
                    help="Padding for CNN VAE")
# VQ-VAE specific
parser.add_argument("--vae_num_embeddings",
                    type=int,
                    default=256,
                    help="Number of embeddings for VQ-VAE")
parser.add_argument("--vae_embedding_dim",
                    type=int,
                    default=64,
                    help="Embedding dimension for VQ-VAE")
parser.add_argument("--vae_beta_vq",
                    type=float,
                    default=0.25,
                    help="Beta parameter for VQ-VAE")
# Regularization
parser.add_argument("--vae_dropout",
                    type=float,
                    default=0.0,
                    help="Dropout rate for the VAE model")
parser.add_argument("--vae_use_batchnorm",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will use batch normalization in the VAE model")   
# Training
parser.add_argument("--vae_beta_kl",
                    type=float,
                    default=1.0,
                    help="Beta parameter for KL divergence in VAE training")
parser.add_argument("--vae_gamma",
                    type=float,
                    default=0.5,
                    help="Gamma parameter for VAE training")

# -----------------------------------------------------------------------
# GANs arguments
# -----------------------------------------------------------------------
parser.add_argument("--gan_config", 
                    type=str, 
                    default=None, 
                    help="Name of the GAN configuration file (e.g. gan_config for data/configs/gan_config.yaml). If not provided, CLI arguments will be used to create the GANConfig.")
parser.add_argument("--gan_architecture", 
                    type=str, 
                    default="GAN", 
                    choices=["GAN", "CGAN", "DCGAN", "MLP_UnrolledGAN", "DC_UnrolledGAN", "StyleGAN"], 
                    help="Architecture of the GAN model")
parser.add_argument("--gan_loss",
                    type=str,   
                    default="Default",
                    choices=["Default", "Wasserstein", "LeastSquare"],
                    help="Loss function for the GAN model")
parser.add_argument("--gan_latent_dim",
                    type=int,
                    default=32,
                    help="Latent dimension for the GAN model")
# For MLPs
parser.add_argument("--gan_input_dim",
                    type=int,
                    default=784,
                    help="Input dimension for MLP GAN (flattened image size)")
parser.add_argument("--gan_hidden_dims",
                    type=int,
                    nargs="+",
                    default=[128, 64],
                    help="Hidden dimensions for MLP GAN (list of integers)")
parser.add_argument("--gan_image_size",
                    type=int,
                    default=28,
                    help="Image size for DCGAN (assumes square images)")
parser.add_argument("--gan_image_channels",
                    type=int,
                    default=1,
                    help="Number of image channels for DCGAN and StyleGAN")
parser.add_argument("--gan_kernel_size",
                    type=int,
                    default=4,
                    help="Kernel size for DCGAN and StyleGAN discriminator")
parser.add_argument("--gan_stride",
                    type=int,
                    default=2,
                    help="Stride for DCGAN and StyleGAN discriminator")
parser.add_argument("--gan_padding",
                    type=int,
                    default=1,
                    help="Padding for DCGAN and StyleGAN discriminator")
parser.add_argument("--gan_noise_coef",
                    type=float,
                    default=0.03,
                    help="Noise coefficient for DCGAN and StyleGAN")
# For CGANs
parser.add_argument("--gan_num_classes",
                    type=int,
                    default=10,
                    help="Number of classes for CGANs (e.g., 10 for MNIST, 100 for CIFAR-100, etc.)")
# For Unrolled GANs
parser.add_argument("--gan_unrolled_steps",
                    type=int,
                    default=5,
                    help="Number of unrolled steps for Unrolled GANs")
# For WGANs
parser.add_argument("--gan_weight_clip",
                    type=float,
                    default=0.01,
                    help="Weight clipping value for WGANs")
parser.add_argument("--gan_gradient_penalty_lambda",
                    type=float,
                    default=10.0,
                    help="Gradient penalty lambda for WGAN-GP")
parser.add_argument("--gan_n_critic",
                    type=int,
                    default=5,
                    help="Number of critic updates per generator update for WGANs")
# For LSGANs
parser.add_argument("--gan_lsgan_lambda",
                    type=float,
                    default=0.5,
                    help="Lambda parameter for LSGANs")
# For StyleGANs
parser.add_argument("--gan_style_dim",
                    type=int,
                    default=64,
                    help="Style dimension for StyleGANs")
parser.add_argument("--gan_kernel_size_style_gen",
                    type=int,
                    default=3,
                    help="Kernel size for StyleGAN generator")
parser.add_argument("--gan_stride_style_gen",
                    type=int,
                    default=1,
                    help="Stride for StyleGAN generator")
parser.add_argument("--gan_padding_style_gen",
                    type=int,
                    default=1,
                    help="Padding for StyleGAN generator")
parser.add_argument("--gan_noise_weight",
                    type=float,
                    default=0.05,
                    help="Noise weight for StyleGANs")
parser.add_argument("--gan_mixing_prob",
                    type=float,
                    default=0.9,
                    help="Mixing probability for StyleGAN")
# Regularization
parser.add_argument("--gan_dropout",
                    type=float,
                    default=0.0,
                    help="Dropout rate for the GAN model")
parser.add_argument("--gan_use_batchnorm",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will use batch normalization in the GAN model")
parser.add_argument("--gan_spectral_norm_on",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will use spectral normalization in the GAN model")
# Training
parser.add_argument("--gan_beta1",
                    type=float,
                    default=0.5,
                    help="Beta1 for GAN training")
parser.add_argument("--gan_beta2",
                    type=float,
                    default=0.999,
                    help="Beta2 for GAN training")
# EMA Sampling
parser.add_argument("--gan_is_ema",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will use EMA for GAN sampling")
parser.add_argument("--gan_ema_decay",
                    type=float,
                    default=0.999,
                    help="EMA decay for GAN sampling")

# -----------------------------------------------------------------------
# Diffusion Model arguments
# -----------------------------------------------------------------------
parser.add_argument("--diffusion_config",
                    type=str,
                    default=None,
                    help="Name of the diffusion configuration file (e.g. diffusion_config for data/configs/diffusion_config.yaml). If not provided, CLI arguments will be used to create the DiffusionConfig.")
parser.add_argument("--diffusion_model_type",
                    type=str,
                    default="res_unet",
                    choices=["cnn", "res_unet"],
                    help="Architecture of the diffusion model")
parser.add_argument("--diffusion_loss",
                    type=str,
                    default="mse",
                    choices=["mse", "l1"],
                    help="Loss function for the diffusion model")
parser.add_argument("--diffusion_num_classes",
                    type=int,
                    default=None,
                    help="Number of classes for class-conditional diffusion. If unset, the model is unconditional.")
parser.add_argument("--diffusion_cond_drop_prob",
                    type=float,
                    default=0.1,
                    help="Conditional dropout probability for diffusion training")
parser.add_argument("--diffusion_guidance_scale",
                    type=float,
                    default=0.9,
                    help="Classifier-free guidance scale for diffusion sampling")
parser.add_argument("--diffusion_image_size",
                    type=int,
                    default=32,
                    help="Input image size for the diffusion model")
parser.add_argument("--diffusion_image_channels",
                    type=int,
                    default=3,
                    help="Number of image channels for the diffusion model")
parser.add_argument("--diffusion_base_channels",
                    type=int,
                    default=64,
                    help="Base number of channels for the diffusion model")
parser.add_argument("--diffusion_channel_mults",
                    type=int,
                    nargs="+",
                    default=[1, 2, 4],
                    help="Channel multipliers for the diffusion model U-Net")
parser.add_argument("--diffusion_time_emb_dim",
                    type=int,
                    default=128,
                    help="Embedding dimension for the diffusion time step")
parser.add_argument("--diffusion_time_width_coef",
                    type=int,
                    default=4,
                    help="Width multiplier for the time embedding MLP")
# Convolution
parser.add_argument("--diffusion_use_attention",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=True,
                    help="If set, will use attention blocks in the diffusion model")
parser.add_argument("--diffusion_attention_resolutions",
                    type=int,
                    nargs="+",
                    default=[8],
                    help="Spatial resolutions where attention is applied in the diffusion model")
parser.add_argument("--diffusion_num_heads",
                    type=int,
                    default=4,
                    help="Number of attention heads for the diffusion model")
parser.add_argument("--diffusion_dropout",
                    type=float,
                    default=0.0,
                    help="Dropout probability for the diffusion model")
parser.add_argument("--diffusion_kernel_size",
                    type=int,
                    default=3,
                    help="Kernel size for diffusion convolution blocks")
parser.add_argument("--diffusion_stride",
                    type=int,
                    default=1,
                    help="Stride for diffusion convolution blocks")
parser.add_argument("--diffusion_padding",
                    type=int,
                    default=1,
                    help="Padding for diffusion convolution blocks")
parser.add_argument("--diffusion_use_batch_norm",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will use batch normalization in classic CNN diffusion models")
parser.add_argument("--diffusion_num_groups",
                    type=int,
                    default=8,
                    help="Number of groups for GroupNorm layers")
parser.add_argument("--diffusion_eps_groupnorm",
                    type=float,
                    default=1e-5,
                    help="Epsilon value for GroupNorm layers")
parser.add_argument("--diffusion_down_kernel_size",
                    type=int,
                    default=4,
                    help="Kernel size for downsampling blocks")
parser.add_argument("--diffusion_down_stride",
                    type=int,
                    default=2,
                    help="Stride for downsampling blocks")
parser.add_argument("--diffusion_down_padding",
                    type=int,
                    default=1,
                    help="Padding for downsampling blocks")
parser.add_argument("--diffusion_down_num_res_blocks",
                    type=int,
                    default=1,
                    help="Number of residual blocks per downsampling stage")
parser.add_argument("--diffusion_up_kernel_size",
                    type=int,
                    default=4,
                    help="Kernel size for upsampling blocks")
parser.add_argument("--diffusion_up_stride",
                    type=int,
                    default=2,
                    help="Stride for upsampling blocks")
parser.add_argument("--diffusion_up_padding",
                    type=int,
                    default=1,
                    help="Padding for upsampling blocks")
parser.add_argument("--diffusion_up_num_res_blocks",
                    type=int,
                    default=1,
                    help="Number of residual blocks per upsampling stage")
# Training
parser.add_argument("--diffusion_learning_rate",
                    type=float,
                    default=2e-4,
                    help="Learning rate for diffusion model training")
parser.add_argument("--diffusion_beta1",
                    type=float,
                    default=0.5,
                    help="Beta1 for diffusion training optimizer")
parser.add_argument("--diffusion_beta2",
                    type=float,
                    default=0.999,
                    help="Beta2 for diffusion training optimizer")
parser.add_argument("--diffusion_use_torch_compile",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will compile the diffusion model with torch.compile")
parser.add_argument("--diffusion_compile_mode",
                    type=str,
                    default="reduce-overhead",
                    choices=["default", "reduce-overhead", "max-autotune"],
                    help="torch.compile mode for diffusion training")
# Sampling
parser.add_argument("--diffusion_use_ddim",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will use DDIM sampling")
parser.add_argument("--diffusion_ddim_steps",
                    type=int,
                    default=50,
                    help="Number of DDIM sampling steps")
parser.add_argument("--diffusion_use_ema",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will use EMA for diffusion sampling")
parser.add_argument("--diffusion_ema_decay",
                    type=float,
                    default=0.9999,
                    help="EMA decay for diffusion sampling")
# Latent Diffusion
parser.add_argument("--diffusion_use_latent_diffusion",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will use latent diffusion instead of pixel-space diffusion")
parser.add_argument("--diffusion_latent_dim",
                    type=int,
                    default=16,
                    help="Latent dimension used by latent diffusion")
parser.add_argument("--diffusion_latent_hidden_dim",
                    type=int,
                    default=64,
                    help="Hidden dimension of the latent diffusion encoder/decoder")
parser.add_argument("--diffusion_latent_kernel_size",
                    type=int,
                    default=4,
                    help="Kernel size for latent diffusion convolution blocks")
parser.add_argument("--diffusion_latent_stride",
                    type=int,
                    default=2,
                    help="Stride for latent diffusion convolution blocks")
parser.add_argument("--diffusion_latent_padding",
                    type=int,
                    default=1,
                    help="Padding for latent diffusion convolution blocks")
parser.add_argument("--diffusion_latent_scale_factor",
                    type=float,
                    default=0.18215,
                    help="Scale factor used when mapping latent variables to the VAE latent space")

# -----------------------------------------------------------------------
# inference.py arguments
# -----------------------------------------------------------------------
parser.add_argument("--inference_config",
                    type=str,
                    default=None,
                    help="Name of the inference configuration file (e.g. inference_config for data/configs/inference_config.yaml). If not provided, default CLI arguments will be used.")
parser.add_argument("--inference_model_type", 
                    type=str, 
                    default="ddpm", 
                    choices=["ddpm", "stable_diffusion"], 
                    help="Type of model to use for inference")
parser.add_argument("--inference_description", 
                    type=str, 
                    default="a futuristic city at night", 
                    help="Description for stable diffusion model")
parser.add_argument("--inference_batch_size",
                    type=int,
                    default=1,
                    help="Number of images to generate in a batch")

# -----------------------------------------------------------------------
# fine_tuning.py arguments
# -----------------------------------------------------------------------
parser.add_argument("--finetuning_config",
                    type=str,
                    default=None,
                    help="Name of the fine-tuning configuration file (e.g. finetuning_config for data/configs/finetuning_config.yaml). If not provided, default CLI arguments will be used.")
parser.add_argument("--finetuning_experiment",
                    type=str,
                    choices=["baseline", "custom_scheduler_and_sampling", "finetune", "lora"],
                    default="baseline",
                    help="Type of experiment to run for fine-tuning")
# Sampling
parser.add_argument("--finetuning_prompts",
                    type=str,
                    nargs="+",
                    default=[""],
                    help="List of prompts for image generation")
parser.add_argument("--finetuning_sampler", 
                    type=str,
                    choices=["ddpm", "ddim"],
                    default=None,
                    help="Sampler to use for inference during fine-tuning (my own implementation of ddpm or ddim)")
parser.add_argument("--finetuning_eta",
                    type=float,
                    default=0.0,
                    help="Eta parameter for DDIM sampler (default: 0.0, deterministic)")
# Training
parser.add_argument("--is_finetuning",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will run the training loop for full or LoRA fine-tuning")
parser.add_argument("--finetuning_gradient_clip",
                    type=float,
                    default=1.0,
                    help="Gradient clipping value")
# LoRA
parser.add_argument("--finetuning_lora_rank",
                    type=int,
                    default=4,
                    help="LoRA rank")
parser.add_argument("--finetuning_lora_alpha",
                    type=int,
                    default=4,
                    help="LoRA alpha")
# Output
parser.add_argument("--finetuning_lora_name",
                    type=str,
                    default="default",
                    help="Name of the LoRA model to save")

# =======================================
# Argument Validation
# =======================================
args = parser.parse_args()
if args.launch_mode is None:
    parser.error("Please specify a launch mode using --launch_mode (vae, gan, diffusion, inference, finetuning)")

if args.launch_mode == "finetuning":
    if args.dataset not in ["cifar10", "imagefolder"]:
        parser.error("For fine-tuning, please specify a valid dataset using --dataset (cifar10, imagefolder)")


# =======================================
# Path Setup
# =======================================
SAVE_FOLDER = PROJECT_ROOT / "data"
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

CONFIGS_FOLDER = SAVE_FOLDER / "configs"
CONFIGS_FOLDER.mkdir(parents=True, exist_ok=True)

SAVE_MODEL_FOLDER = SAVE_FOLDER / "models_parameters"
SAVE_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUTS_FOLDER = SAVE_FOLDER / "outputs"
OUTPUTS_FOLDER.mkdir(parents=True, exist_ok=True)


# =======================================
# Set Seed and Device
# =======================================
# set seed
if args.seed is not None:
    set_seed(args.seed)

# device selection
if args.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args.device


# =======================================
# Launcher Logic
# =======================================

# -----------------------------------------------------------------------
# Dataset Loading
# -----------------------------------------------------------------------
loader = None
if args.dataset != "none":
    if args.dataset == "cifar10":
        print_section("CIFAR-10 Dataset Loading...")
        if args.subset_size is not None:
            loader = load_cifar10(
                batch_size=args.training_batch_size,
                downsample=args.downsample_size,
                grayscale=args.grayscale,
                normalize=args.normalize,
                flatten=args.flatten,
                train=True,
                subset_size=args.subset_size
            )
        else:
            loader = load_cifar10(
                batch_size=args.training_batch_size,
                downsample=args.downsample_size,
                grayscale=args.grayscale,
                normalize=args.normalize,
                flatten=args.flatten,
                train=True
            )
    elif args.dataset == "fashion_mnist":
        print_section("Fashion-MNIST Dataset Loading...")
        loader = load_fashion_mnist(
            batch_size=args.training_batch_size,
            downsample=args.downsample_size,
            normalize=args.normalize,
            flatten=args.flatten,
            train=True, 
            root=None
        )
    elif args.dataset == "mnist":
        print_section("MNIST Dataset Loading...")
        loader = load_mnist(
            batch_size=args.training_batch_size,
            downsample=args.downsample_size,
            normalize=args.normalize,
            flatten=args.flatten,
            train=True, 
            root=None
        )
    elif args.dataset == "imagefolder":
        print_section("ImageFolder Dataset Loading...")
        raise NotImplementedError("ImageFolder dataset loading is not implemented yet. Please use CIFAR-10, FashionMNIST, or MNIST for now.")

    input_dim = None
    if args.dataset in ["fashion_mnist", "mnist"] or (args.dataset == "cifar10" and args.flatten==True):
        x, _ = next(iter(loader))
        input_dim = x.shape[1]

    image_channels = None
    image_size = None
    if args.dataset == "cifar10" and args.flatten==False:
        image_channels = x.shape[1]
        image_size = x.shape[2]

    num_classes = None
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "fashion_mnist":
        num_classes = 10
    elif args.dataset == "mnist":
        num_classes = 10

# -----------------------------------------------------------------------
# VAE
# -----------------------------------------------------------------------
"""
Minimum command to launch VAE training:

python src/launcher.py 
    --launch_mode vae 
    --seed 42 
    --name sample_vae_test
    --dataset mnist 
    --is_training 
    --n_sample 9 
    --save_model 
    --model_name vae_test
    --epochs 1 
    --vae_config vae_config
"""
if args.launch_mode == "vae":
    print_section("Launching in VAE mode...")

    vae_input_dim = input_dim if input_dim is not None else args.vae_input_dim
    vae_image_channels = image_channels if image_channels is not None else args.vae_image_channels
    vae_image_size = image_size if image_size is not None else args.vae_image_size

    config_path = os.path.join(CONFIGS_FOLDER, f"{args.vae_config}.yaml")
    if args.vae_config is not None:
        config_dict = yaml.safe_load(open(config_path, "r"))
        vae_config = VAEConfig(**config_dict)
        vae_config.input_dim = vae_input_dim
        vae_config.image_channels = vae_image_channels
        vae_config.image_size = vae_image_size
    else:
        vae_config = VAEConfig(
            model_type=args.vae_model_type,
            architecture=args.vae_architecture,
            input_dim=vae_input_dim,
            hidden_dims=args.vae_hidden_dims,
            latent_dim=args.vae_latent_dim,
            image_channels=vae_image_channels,
            image_size=vae_image_size,
            kernel_size=args.vae_kernel_size,
            stride=args.vae_stride,
            padding=args.vae_padding,
            num_embeddings=args.vae_num_embeddings,
            embedding_dim=args.vae_embedding_dim,
            beta_vq=args.vae_beta_vq,
            reconstruction_loss=args.vae_reconstruction_loss,
            dropout=args.vae_dropout,
            use_batchnorm=args.vae_use_batchnorm,
            beta_kl=args.vae_beta_kl,
            gamma=args.vae_gamma,
            learning_rate=args.learning_rate,
            step_size=args.step_size,
            weight_decay=args.weight_decay
        )
        yaml.dump(vae_config.__dict__, open(config_path, "w"))
    print_subsection("VAE Configuration:")
    for key, value in vae_config.__dict__.items():
        print(f"{key}: {value}")

    if vae_config.model_type in ["vae", "vqvae"]:
        vae = BaseVAE(cfg=vae_config, device=device)
    elif vae_config.model_type == "fastvae":
        vae = FastCNNVAE(cfg=vae_config, device=device)
    print_subsection("VAE Architecture:")
    print(vae)

    if args.is_training:
        print_subsection("Starting VAE Training...")
        metrics = vae.fit(loader, epochs=args.epochs, verbose=True)
        print()
        print_subsection("Training Metrics:")
        for epoch, metric in enumerate(metrics):
            print(f"Epoch {epoch+1}: Loss={metric.loss:.4f}, Recon Loss={metric.recon:.4f}, KL Div={metric.kld:.4f}, VQ Loss={metric.vq:.4f}")
        print()

        if args.save_model:
            print_subsection("Saving VAE Model...")
            path = os.path.join(SAVE_MODEL_FOLDER, "VAE", f"{args.model_name}.pth")
            vae.save(path)
            print(f"VAE model saved to {path}")

    else: # Load pre-trained model for inference
        print_subsection("Loading Pre-trained VAE Model...")
        path = os.path.join(SAVE_MODEL_FOLDER, "VAE", f"{args.model_name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pre-trained VAE model not found at {path}. Please train the model first or provide a valid path.")
        vae.load(path)
        print(f"VAE model loaded from {path}")

    print_subsection("Running VAE Inference...")
    vae.plot_image_samples(n=args.n_sample, n_rows=int(np.sqrt(args.n_sample)), save_path=os.path.join(OUTPUTS_FOLDER, f"{args.name}.png"))


# -----------------------------------------------------------------------
# GAN
# -----------------------------------------------------------------------
"""
Minimum command to launch GAN training:

python src/launcher.py 
    --launch_mode gan 
    --seed 42 
    --name sample_gan_test
    --dataset mnist 
    --is_training 
    --n_sample 5 
    --save_model 
    --model_name gan_test
    --epochs 2
    --gan_config gan_config
"""
if args.launch_mode == "gan":
    print_section("Launching in GAN mode...")

    gan_input_dim = input_dim if input_dim is not None else args.gan_input_dim
    gan_image_channels = image_channels if image_channels is not None else args.gan_image_channels
    gan_image_size = image_size if image_size is not None else args.gan_image_size
    gan_num_classes = num_classes if num_classes is not None else args.gan_num_classes

    config_path = os.path.join(CONFIGS_FOLDER, f"{args.gan_config}.yaml")
    if args.gan_config is not None:
        config_dict = yaml.safe_load(open(config_path, "r"))
        gan_config = GANConfig(**config_dict)
        gan_config.input_dim = gan_input_dim
        gan_config.image_channels = gan_image_channels
        gan_config.image_size = gan_image_size
        gan_config.num_classes = gan_num_classes
    else:
        gan_config = GANConfig(
            architecture=args.gan_architecture,
            loss=args.gan_loss,
            latent_dim=args.gan_latent_dim,
            input_dim=gan_input_dim,
            hidden_dims=args.gan_hidden_dims,
            image_channels=gan_image_channels,
            image_size=gan_image_size,
            kernel_size=args.gan_kernel_size,
            stride=args.gan_stride,
            padding=args.gan_padding,
            num_classes=gan_num_classes,
            unrolled_steps=args.gan_unrolled_steps,
            weight_clip=args.gan_weight_clip,
            gradient_penalty_lambda=args.gan_gradient_penalty_lambda,
            n_critic=args.gan_n_critic,
            lsgan_lambda=args.gan_lsgan_lambda,
            style_dim=args.gan_style_dim,
            kernel_size_style_gen=args.gan_kernel_size_style_gen,
            stride_style_gen=args.gan_stride_style_gen,
            padding_style_gen=args.gan_padding_style_gen,
            noise_weight=args.gan_noise_weight,
            mixing_prob=args.gan_mixing_prob,
            dropout=args.gan_dropout,
            use_batchnorm=args.gan_use_batchnorm,
            spectral_norm_on=args.gan_spectral_norm_on,
            learning_rate=args.learning_rate,
            step_size=args.step_size,
            weight_decay=args.weight_decay,
            beta1=args.gan_beta1,
            beta2=args.gan_beta2,
            is_ema=args.gan_is_ema,
            ema_decay=args.gan_ema_decay
        )
        yaml.dump(gan_config.__dict__, open(config_path, "w"))
    print_subsection("GAN Configuration:")
    for key, value in gan_config.__dict__.items():
        print(f"{key}: {value}")

    gan = GAN(cfg=gan_config, device=device)
    print_subsection("GAN Architecture:")
    print(gan)

    if args.is_training:
        print_subsection("Starting GAN Training...")
        G_losses, D_losses = [], []
        history = gan.fit(loader, epochs=args.epochs, verbose=True)
        G_losses.extend([hist.G_loss for hist in history])
        D_losses.extend([hist.D_loss for hist in history])

        plt.figure()
        plt.plot(G_losses, label="Generator Loss")
        plt.plot(D_losses, label="Discriminator Loss")
        plt.legend()
        plt.title(f"{args.name} Loss History on {args.dataset} dataset")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(OUTPUTS_FOLDER, f"loss_history_{args.name}.png"))
        plt.close()

        if args.save_model:
            print_subsection("Saving GAN Model...")
            path = os.path.join(SAVE_MODEL_FOLDER, "GAN", f"{args.model_name}.pth")
            gan.save(path)
            print(f"GAN model saved to {path}")

    else: # Load pre-trained model for inference
        print_subsection("Loading Pre-trained GAN Model...")
        path = os.path.join(SAVE_MODEL_FOLDER, "GAN", f"{args.model_name}.pth")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pre-trained GAN model not found at {path}. Please train the model first or provide a valid path.")
        gan.load(path)
        print(f"GAN model loaded from {path}")

    print_subsection("Running GAN Inference...")
    samples = gan.sample(n=args.n_sample).numpy()
    samples = (samples + 1) / 2  # Rescale from [-1, 1] to [0, 1] for visualization
    plot_images(samples, n=args.n_sample, save_path=os.path.join(OUTPUTS_FOLDER, f"{args.name}.png"))


"""
# -----------------------------------------------------------------------
# Diffusion Model
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Fine-tuning
# -----------------------------------------------------------------------
if args.launch_mode == "finetuning":






    # For args.gan_architecture == "CGAN"
    num_classes: = num_classes of the dataset (e.g., 10 for MNIST, 100 for CIFAR-100, etc.)
update config.num_classes

for conditional embeddings for diffision models
"""
