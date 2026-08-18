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
from src.models.diffusion_models import NoiseScheduler
from src.pretrained.fine_tuning import PretrainedFineTuning


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
                    choices=["baseline", "custom_sampling", "custom_scheduler", "finetune", "lora"],
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
parser.add_argument("--save_name",
                    type=str,
                    default="output",
                    help="Name of the output image file (without extension), in case of multiple images, it becomes the folder name in which the images will be saved as image_0.png, image_1.png, ...")


args = parser.parse_args()


# =======================================
# Argument Validation
# =======================================


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
    device=device,
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
    images = images if isinstance(images, (list, tuple)) else list(images)
    if len(images) == 1:
        image = images[0]
        if torch.is_tensor(image):
            image = transforms.ToPILImage()(image.detach().cpu().clamp(0, 1))
        image.save(f"{image_save_path}.png")
    else:
        image_save_path.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            if torch.is_tensor(img):
                img = transforms.ToPILImage()(img.detach().cpu().clamp(0, 1))
            img.save(os.path.join(image_save_path, f"image_{i}.png"))


# ============================================================
# Experiment B
# ============================================================
elif args.experiment == "custom_sampling":
    print("\nTiny-SD Scheduler + Tiny-SD UNet + OWN DDPM/DDIM sampler")

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
    images = images if isinstance(images, (list, tuple)) else list(images)
    if len(images) == 1:
        image = images[0]
        if torch.is_tensor(image):
            image = transforms.ToPILImage()(image.detach().cpu().clamp(0, 1))
        image.save(f"{image_save_path}.png")
    else:
        image_save_path.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            if torch.is_tensor(img):
                img = transforms.ToPILImage()(img.detach().cpu().clamp(0, 1))
            img.save(os.path.join(image_save_path, f"image_{i}.png"))


"""
# ============================================================
# Experiment C
# ============================================================

elif args.experiment == "custom_scheduler":

    print(
        "\nTiny-SD UNet + YOUR NoiseScheduler "
        "+ YOUR DDPM sampler"
    )

    images = pipeline.sample_custom(
        prompt=args.prompt,
        steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    output_dir = Path(args.save_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    from PIL import Image
    import numpy as np

    for i in range(images.shape[0]):

        image = (
            images[i]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image = Image.fromarray(
            (image * 255).astype(np.uint8)
        )

        image.save(
            output_dir
            / f"{args.save_name}_{i}.png"
        )


# ============================================================
# Experiments D/E - Training
# ============================================================

elif args.experiment in {
    "finetune",
    "lora",
}:

    if args.dataset == "none":

        raise ValueError(
            "Training requires --dataset imagefolder"
        )

    # --------------------------------------------------------
    # Freeze VAE
    # --------------------------------------------------------

    pipeline.freeze_vae()

    # --------------------------------------------------------
    # Freeze text encoder
    # --------------------------------------------------------

    pipeline.freeze_text_encoder()

    # --------------------------------------------------------
    # UNet
    # --------------------------------------------------------

    if args.experiment == "lora":

        print("\nEnabling LoRA...")

        pipeline.enable_lora(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )

    else:

        print("\nFull UNet fine-tuning...")

        pipeline.unet.requires_grad_(
            True
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize(
            (args.height, args.width)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ),
    ])

    dataset = datasets.ImageFolder(
        args.data_dir,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        pipeline.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(args.epochs):

        total_loss = 0.0

        for images, labels in loader:

            images = images.to(
                device
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # ImageFolder labels are not prompts.
            #
            # For now use one prompt.
            # Replace this with your dataset captions.
            # ------------------------------------------------

            prompts = [
                args.prompt
                for _ in range(images.shape[0])
            ]

            loss = pipeline.training_step(
                images,
                prompts,
                optimizer,
            )

            total_loss += loss

        mean_loss = (
            total_loss
            / len(loader)
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"| loss = {mean_loss:.6f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if args.save_model:

        output_dir = Path(
            args.save_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if args.experiment == "lora":

            pipeline.save_lora(
                output_dir
            )

            print(
                f"LoRA saved to {output_dir}"
            )

        else:

            torch.save(
                pipeline.unet.state_dict(),
                output_dir
                / "finetuned_unet.pt",
            )

            print(
                "UNet saved."
            )
"""


# === FILE: NRT/NRT_fine_tuning/test.py ===