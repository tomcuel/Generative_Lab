# ===========================
# Path setup
# ===========================
import os
from pathlib import Path
import subprocess
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def test_inference_ddpm():
    """
    Test the inference function for the DDPM model
    """
    args = {
        "is_nrt": True,
        "seed": 42,
        "device": "auto",
        "model_type": "ddpm",
        "num_inference_steps": 20,
        "save_name": "ddpm_output",
        "batch_size": 3,
        "save_model": True,
        "show_architecture": False
    }
    print(f"Command to run: python {os.path.join(PROJECT_ROOT, 'src', 'pretrained', 'inference.py')} " + " ".join([f"--{k}={v}" for k, v in args.items()]))
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "pretrained", "inference.py")]
        + [f"--{k}={v}" for k, v in args.items()], 
    capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


def test_inference_stable_diffusion():
    """
    Test the inference function for the Stable Diffusion model
    """
    args = {
        "is_nrt": True,
        "seed": 42,
        "device": "auto",
        "model_type": "stable_diffusion",
        "num_inference_steps": 20,
        "description": "a futuristic city at night",
        "save_name": "futuristic_city",
        "batch_size": 3,
        "guidance_scale": 7.5,
        "height": 512,
        "width": 512,
        "save_model": True,
        "show_architecture": True
    }
    print(f"Command to run: python {os.path.join(PROJECT_ROOT, 'src', 'pretrained', 'inference.py')} " + " ".join([f"--{k}={v}" for k, v in args.items()]))
    result = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "pretrained", "inference.py")]
        + [f"--{k}={v}" for k, v in args.items()], 
    capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


if __name__ == "__main__":
    test_inference_ddpm()
    test_inference_stable_diffusion()



"""
command line usage example (DDPM):
python src/pretrained/inference.py --model_type ddpm --num_inference_steps 20 --batch_size 3 --save_name "ddpm_output" --save_model
command line usage example (stable diffusion):
python src/pretrained/inference.py --model_type stable_diffusion --description "a futuristic city at night" --num_inference_steps 20 --batch_size 3 --save_name "futuristic_city" --save_model --show_architecture
"""

"""
loading the class to play with it without having everything handled by the command line argulents 
"""