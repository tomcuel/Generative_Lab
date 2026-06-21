# ===========================
# Path setup
# ===========================
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
from torch.utils.data import DataLoader


# ===========================
# Imports
# ===========================
from src.data.load import load_cifar10
from src.data.utils import plot_images
from src.models.diffusion_models import (
    DiffusionConfig, 
    NoiseScheduler, 
    TimeEmbedding, 
    CNN, 
    ResBlock, 
    AttentionBlock,
    DownBlock, 
    UpBlock,
    UNet, 
    LatentAutoEncoder,
    EMA,
    DiffusionModel
)


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MNIST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mnist")
os.makedirs(MNIST_OUTPUT_DIR, exist_ok=True)
CIFAR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cifar10")
os.makedirs(CIFAR_OUTPUT_DIR, exist_ok=True)

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
# Test Noise Scheduler
# ===========================


# ===========================
# Test Time Embedding
# ===========================


# ===========================
# Test CNN
# ===========================


# ===========================
# Test ResBlock / AttentionBlock / DownBlock / UpBlock / UNet
# ===========================


# ===========================
# Test LatentAutoEncoder
# ===========================


# ===========================
# Test EMA
# ===========================


# ===========================
# Test DiffusionModel on MNIST with CNN
# ===========================


# ===========================
# Test DiffusionModel on MNIST with ResUNet
# ===========================


# ===========================
# Test DiffusionModel on MNIST with Latent Diffusion
# ===========================


# ===========================
# Test DiffusionModel on CIFAR-10 with ResUNet (fast to verify if it's working only)
# ===========================


if __name__ == "__main__":
    clear_data_dir()