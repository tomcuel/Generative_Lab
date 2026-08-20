# =======================================
# Library Imports
# =======================================
import argparse
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path
import sys 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =======================================
# Important Path Setup
# =======================================
from src.data.utils import set_seed
from src.data.load import load_cifar10
from src.models.diffusion_models import NoiseScheduler
from src.pretrained.fine_tuning import PretrainedFineTuning


def _images_to_pil_list(images):
    if torch.is_tensor(images):
        images = [images]
    elif not isinstance(images, (list, tuple)):
        images = list(images)

    pil_images = []
    for image in images:
        if torch.is_tensor(image):
            if image.ndim == 4:
                pil_images.extend(transforms.ToPILImage()(sample.detach().cpu().clamp(0, 1)) for sample in image)
            else:
                pil_images.append(transforms.ToPILImage()(image.detach().cpu().clamp(0, 1)))
        else:
            pil_images.append(image)
    return pil_images


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


parser = argparse.ArgumentParser(description="Generative Lab - Pretrained Diffusion Experiments")

# ------------------------------------------------------------
# Experiment Setup
# ------------------------------------------------------------
parser.add_argument("--is_nrt",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="modify the project root saving path to be compatible with NRT")
parser.add_argument("--seed",
                    type=int,
                    default=None,
                    help="Random seed for reproducible results")
parser.add_argument("--device",
                    type=str,
                    default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Device to run fine-tuning on (auto selects cuda if available)")

# ------------------------------------------------------------
# Experiment Choice
# ------------------------------------------------------------
parser.add_argument("--experiment",
                    type=str,
                    choices=["baseline", "custom_scheduler_and_sampling", "finetune", "lora"],
                    default="baseline",
                    help="Type of experiment to run for fine-tuning")

# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------
parser.add_argument("--prompts",
                    type=str,
                    nargs="+",
                    default=[""],
                    help="List of prompts for image generation")
parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help="Number of images to generate in a batch = per prompt for baseline")
parser.add_argument("--sampler", 
                    type=str,
                    choices=["ddpm", "ddim"],
                    default=None,
                    help="Sampler to use for inference during fine-tuning (my own implementation of ddpm or ddim)")
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
parser.add_argument("--eta",
                    type=float,
                    default=0.0,
                    help="Eta parameter for DDIM sampler (default: 0.0, deterministic)")

# ------------------------------------------------------------
# Custom scheduler
# ------------------------------------------------------------
parser.add_argument("--timesteps",
                    type=int,
                    default=1000,
                    help="Number of timesteps for the noise scheduler")
parser.add_argument("--beta_schedule",
                    type=str,
                    choices=["linear", "cosine"],
                    default="linear",
                    help="Beta schedule for the noise scheduler")
parser.add_argument("--beta_start",
                    type=float,
                    default=1e-4,
                    help="Start value for the beta schedule")
parser.add_argument("--beta_end",
                    type=float,
                    default=0.02,
                    help="End value for the beta schedule")
parser.add_argument("--cosine_s",
                    type=float,
                    default=0.008,
                    help="Cosine schedule parameter")

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
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
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-2,
                    help="Weight decay for training")
parser.add_argument("--gradient_clip",
                    type=float,
                    default=1.0,
                    help="Gradient clipping value")

# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------
parser.add_argument("--lora_rank",
                    type=int,
                    default=4,
                    help="LoRA rank")
parser.add_argument("--lora_alpha",
                    type=int,
                    default=4,
                    help="LoRA alpha")

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
parser.add_argument("--dataset",
                    type=str,
                    choices=["cifar10", "imagefolder"],
                    default="none",
                    help="Dataset to use for training (cifar10, celebA, imagefolder (need to be prepared correctly in data/imagefolder previously))")
parser.add_argument("--subset_size",
                    type=int,
                    default=None,
                    help="Subset size for training (only for mnist and cifar10)")

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
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
parser.add_argument("--lora_name",
                    type=str,
                    default="default",
                    help="Name of the LoRA model to save")
parser.add_argument("--save_name",
                    type=str,
                    default="output",
                    help="Name of the output image file (without extension), in case of multiple images, it becomes the folder name in which the images will be saved as image_0.png, image_1.png, ...")


args = parser.parse_args()


# =======================================
# Path Setup
# =======================================
SAVE_FOLDER = PROJECT_ROOT / "data"
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

SAVE_MODEL_FOLDER = SAVE_FOLDER / "models_parameters"
SAVE_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
model_load_path = SAVE_MODEL_FOLDER / "stable_diffusion_model"
if args.save_model:
    if args.experiment == "finetune":
        model_save_path = SAVE_MODEL_FOLDER / f"finetune_stable_diffusion_model"
    elif args.experiment == "lora":
        model_save_path = SAVE_MODEL_FOLDER / f"lora_stable_diffusion_model"
    else:
        model_save_path = None

SAVE_IMAGES_FOLDER = SAVE_FOLDER / "output"
if args.is_nrt: # saving the pictures outputs in the NRT local path
    SAVE_IMAGES_FOLDER = Path("data")
SAVE_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
image_save_path = SAVE_IMAGES_FOLDER / args.save_name


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
# Scheduler
# =======================================
custom_scheduler = NoiseScheduler(
    timesteps=args.timesteps,
    beta_schedule=args.beta_schedule,
    beta_start=args.beta_start,
    beta_end=args.beta_end,
    s=args.cosine_s,
    device=device
)


# ============================================================
# Pipeline
# ============================================================
pipeline = PretrainedFineTuning(
    model_path=model_load_path,
    device=device,
    custom_scheduler=custom_scheduler
)


# ============================================================
# Architecture
# ============================================================
if args.show_architecture:
    print("\n===== VAE =====")
    print(pipeline.vae)

    print("\n===== UNET =====")
    print(pipeline.unet)

    print("\n===== SCHEDULER =====")
    if args.experiment in ["custom_sampling", "custom_scheduler", "finetune", "lora"]:
        print(pipeline.custom_scheduler)
    else:
        print(pipeline.scheduler)


# ============================================================
# Experiment A
# ============================================================
if args.experiment == "baseline":
    print("\nRunning Tiny-SD baseline...")

    images = pipeline.sample_sd(
        prompt=args.prompts[0], # only the first prompt is used for baseline
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )
    images = _images_to_pil_list(images)
    if len(images) == 1:
        image = images[0]
        image.save(f"{image_save_path}.png")
    else:
        image_save_path.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(image_save_path, f"image_{i}.png"))


# ============================================================
# Experiment B
# ============================================================
elif args.experiment == "custom_scheduler_and_sampling":
    print("\nTiny-SD Scheduler for learning + Tiny-SD UNet + OWN DDPM/DDIM sampler and Scheduler for inference")

    images = pipeline.sample_custom(
        prompts=args.prompts,
        method=args.sampler,
        height=args.height,
        width=args.width,
        steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        eta=args.eta
    )
    images = _images_to_pil_list(images)
    if len(images) == 1:
        image = images[0]
        image.save(f"{image_save_path}.png")
    else:
        image_save_path.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(image_save_path, f"image_{i}.png"))


# ============================================================
# Experiments C/D - Fine-tuning (LoRA or not)
# ============================================================
elif args.experiment in ("finetune", "lora"):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {args.experiment.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Freeze VAE + text encoder
    # --------------------------------------------------------
    print("\n[1/6] Freezing VAE...")
    pipeline.freeze_vae()
    print("      ✓ VAE frozen")

    print("\n[2/6] Freezing text encoder...")
    pipeline.freeze_text_encoder()
    print("      ✓ Text encoder frozen")

    # --------------------------------------------------------
    # 2. Configure UNet
    # --------------------------------------------------------
    if args.experiment == "lora":
        print("\n[3/6] Enabling LoRA...")
        pipeline.enable_lora(rank=args.lora_rank, alpha=args.lora_alpha)
        trainable = list(pipeline.trainable_parameters())
        print(f"      ✓ LoRA enabled")
        print(f"      Trainable tensors: {len(trainable)}")
    else:
        print("\n[3/6] Enabling full UNet fine-tuning...")
        pipeline.unet.requires_grad_(True)
        trainable = list(pipeline.unet.parameters())
        print("      ✓ UNet trainable")

    # --------------------------------------------------------
    # 3. Dataset
    # --------------------------------------------------------
    print("\n[4/6] Loading dataset...")
    if args.dataset == "cifar10":
        print("      Dataset: CIFAR-10")
        loader = load_cifar10(batch_size=args.training_batch_size, downsample=(args.height, args.width), grayscale=False, normalize=True, flatten=False, train=True, subset_size=args.subset_size)
    elif args.dataset == "imagefolder":
        dataset_path = PROJECT_ROOT / "data" / "imagefolder"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found:\n{dataset_path}")

        transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = datasets.ImageFolder(root=str(dataset_path), transform=transform)
        loader = DataLoader(dataset, batch_size=args.training_batch_size, shuffle=True)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    pipeline.class_names = getattr(loader, "class_names", None)
    
    print(f"      ✓ Dataset loaded")
    print(f"      Batches: {len(loader)}")
    print("      Dataset captions:")
    for i, caption in enumerate(pipeline.class_names or []):
        print(f"        {i}: {caption}")

    # --------------------------------------------------------
    # 4. Optimizer
    # --------------------------------------------------------
    trainable_parameters = [p for p in pipeline.trainable_parameters() if p.requires_grad]
    if len(trainable_parameters) == 0:
        raise RuntimeError("No trainable parameters found!")
    n_parameters = sum(p.numel() for p in trainable_parameters)

    print("\nTrainable parameters:")
    print(f"      {n_parameters:,}")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # --------------------------------------------------------
    # 5. Training
    # --------------------------------------------------------
    print("\n[5/6] Starting training...")
    print("-" * 70)
    for epoch in range(args.epochs):
        pipeline.unet.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                labels = (batch[1] if len(batch) > 1 else None)
            else:
                images = batch
                labels = None
            images = images.to(device)

            loss = pipeline.train_step(images, optimizer=optimizer, gradient_clip_value=args.gradient_clip, prompts=None, labels=labels)

            total_loss += float(loss)
            num_batches += 1

            print(f"Epoch {epoch + 1}/{args.epochs} | batch {batch_idx + 1}/{len(loader)} | loss={loss:.6f}", flush=True)

        mean_loss = total_loss / max(num_batches, 1)
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs} | mean loss = {mean_loss:.6f}\n", flush=True)

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------
    print("[6/6] Saving model...")
    if args.save_model:
        if args.experiment == "lora":
            pipeline.save_lora_adapter(model_save_path, adapter_name=args.lora_name)
            pipeline.save_finetuned_model(model_save_path)
            print(f"✓ LoRA saved to {model_save_path / args.lora_name}")
        else:
            pipeline.save_finetuned_model(model_save_path)
            print(f"✓ Fine-tuned UNet saved to {model_save_path}")

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAMPLING AFTER TRAINING")
    print("=" * 70)

    print(f"Sampler          : {args.sampler}")
    print(f"Steps            : {args.num_inference_steps}")
    print(f"Guidance scale   : {args.guidance_scale}")
    print(f"Resolution       : {args.width}x{args.height}")

    print("\nGenerating images...", flush=True)
    images = pipeline.sample_custom(
        prompts=args.prompts,
        method=args.sampler,
        height=args.height,
        width=args.width,
        steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        eta=args.eta
    )
    print("✓ Sampling finished")

    images = _images_to_pil_list(images)
    print(f"Generated {len(images)} image(s)")

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------
    images = _images_to_pil_list(images)
    if len(images) == 1:
        image = images[0]
        image.save(f"{image_save_path}.png")
        print(f"✓ Saved: {image_save_path}.png")
    else:
        image_save_path.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(os.path.join(image_save_path, f"image_{i}.png"))
        print(f"✓ Saved {len(images)} images to: {image_save_path}")

    print("\nDone.")


# === FILE: NRT/NRT_fine_tuning/test.py ===