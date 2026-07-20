# =======================================
# Library Imports
# =======================================
import argparse
from diffusers import DDPMPipeline
from diffusers import StableDiffusionPipeline
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path
import sys 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import torch


# =======================================
# Important Path Setup
# =======================================
from src.data.utils import set_seed


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


parser = argparse.ArgumentParser(description="Run inference on a pretrained model")
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
                    help="Device to run inference on (auto selects cuda if available)")
parser.add_argument("--model_type", 
                    type=str, 
                    default="ddpm", 
                    choices=["ddpm", "stable_diffusion"], 
                    help="Type of model to use for inference")
parser.add_argument("--num_inference_steps",
                    type=int, 
                    default=20, 
                    help="Number of inference steps")
parser.add_argument("--description", 
                    type=str, 
                    default="a futuristic city at night", 
                    help="Description for stable diffusion model")
parser.add_argument("--save_name",
                    type=str, 
                    default="output", 
                    help="Name of the output image file (without extension), in case of multpile images, it become the folder name in which the images will be saved as image_0.png, image_1.png, ...")
parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help="Number of images to generate in a batch")
parser.add_argument("--guidance_scale",
                    type=float,
                    default=7.5,
                    help="Classifier-free guidance scale for text-to-image models (stable diffusion)")
parser.add_argument("--height",
                    type=int,
                    default=512,
                    help="Height of generated image (stable diffusion)")
parser.add_argument("--width",
                    type=int,
                    default=512,
                    help="Width of generated image (stable diffusion)")
parser.add_argument("--save_model",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will save the model parameters for future reuse without re-downloading the whole thing")
parser.add_argument("--show_architecture",
                    type=str2bool,
                    nargs="?",
                    const=True,
                    default=False,
                    help="If set, will print the architecture of the model")
args = parser.parse_args()


# =======================================
# Argument Validation
# =======================================
if args.model_type == "stable_diffusion" and not args.description:
    parser.error("Description is required for stable diffusion model")


# =======================================
# Path Setup
# =======================================
SAVE_FOLDER = PROJECT_ROOT / "data"
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

SAVE_MODEL_FOLDER = SAVE_FOLDER / "models_parameters"
SAVE_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
model_save_path = SAVE_MODEL_FOLDER / f"{args.model_type}_model"

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
# Inference Logic
# =======================================
class PretrainedInference:
    def __init__(self, model_type, device, args, image_save_path, model_save_path=None):
        self.model_type = model_type
        self.device = device
        self.args = args
        self.image_save_path = image_save_path
        self.model_save_path = model_save_path
        self.pipe = self.load_pipeline(model_save_path)
        self.components = self.get_architecture() if self.model_type == "stable_diffusion" else None

    def load_pipeline(self, model_save_path):
        if model_save_path and Path(model_save_path).exists():
            if self.model_type == "ddpm":
                pipe = DDPMPipeline.from_pretrained(model_save_path)
                pipe.to(self.device)
                return pipe
            elif self.model_type == "stable_diffusion":
                pipe = StableDiffusionPipeline.from_pretrained(model_save_path)
                pipe.to(self.device)
                return pipe
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        else:
            return self.load_pretrained_pipeline()
        
    def load_pretrained_pipeline(self):
        if self.model_type == "ddpm":
            pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
            pipe.to(self.device)
            return pipe
        elif self.model_type == "stable_diffusion":
            pipe = StableDiffusionPipeline.from_pretrained(
                "segmind/tiny-sd",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            pipe.to(self.device)
            return pipe
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def save_pipeline(self):
        if self.model_save_path and self.args.save_model:
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            self.pipe.save_pretrained(self.model_save_path)

    def get_architecture(self):
        components = {}
        for name, obj in [
            ("VAE", self.pipe.vae),
            ("UNet", self.pipe.unet),
            ("Scheduler", self.pipe.scheduler),
            ("Tokenizer", getattr(self.pipe, "tokenizer", None)),
            ("Text Encoder", getattr(self.pipe, "text_encoder", None))
        ]:
            if obj is not None:
                components[name] = obj
        return components

    def print_architecture(self):
        for name, obj in self.components.items():
            print("=" * 40)
            print(f"{name}:")
            print(obj)

    def run_inference(self):
        generator = torch.Generator(self.device).manual_seed(self.args.seed) if self.args.seed is not None else None
        kwargs = {"num_inference_steps": self.args.num_inference_steps, "generator": generator}
        if self.model_type == "stable_diffusion":
            kwargs.update({
                "prompt": self.args.description,
                "guidance_scale": self.args.guidance_scale,
                "height": self.args.height,
                "width": self.args.width,
                "num_images_per_prompt": self.args.batch_size,
            })
        elif self.model_type == "ddpm":
            kwargs.update({"batch_size": self.args.batch_size})
        out = self.pipe(**kwargs)
        return out.images

    def save_images(self, images):
        if len(images) == 1:
            images[0].save(f"{self.image_save_path}.png")
        else:
            self.image_save_path.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images):
                img.save(os.path.join(self.image_save_path, f"image_{i}.png"))

    def run(self):
        if self.model_type == "stable_diffusion" and self.args.show_architecture:
            self.print_architecture()
        images = self.run_inference()
        self.save_images(images)
        self.save_pipeline()


inference = PretrainedInference(model_type=args.model_type, device=device, args=args, image_save_path=image_save_path, model_save_path=model_save_path)
inference.run()