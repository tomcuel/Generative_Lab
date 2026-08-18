# ===========================
# Path setup
# ===========================
import os
from pathlib import Path
import subprocess
import sys
import shlex
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


# ===========================
# Command Line Experiments
# ===========================
def build_launch_command(args):
    command = [sys.executable, os.path.join(PROJECT_ROOT, "src", "pretrained", "launch.py")]
    for key, value in args.items():
        command.append(f"--{key}")
        if isinstance(value, (list, tuple)):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command


def test_experiment_a():
    """
    Test the inference function for the DDPM model
    """
    args = {
        "is_nrt": True,
        "seed": 42,
        "device": "auto",
        "experiment": "baseline",
        "prompts": ["a futuristic city at night"],
        "batch_size": 3,
        "height": 256,
        "width": 256,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "save_name": "ddpm_output",
        "show_architecture": False,
        "save_model": False,
        "save_name": "experiment_a"
    }
    command = build_launch_command(args)
    print("Command to run: " + " ".join(shlex.quote(part) for part in command))
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


def test_experiment_b1():
    """
    Test my own DDPM sampling implementation
    """
    args = {
        "is_nrt": True,
        "seed": 42,
        "device": "auto",
        "experiment": "custom_sampling",
        "prompts": [
            "a futuristic city at night",
            "a landscape with mountains and a river",
            "a futuristic city at night",
        ],
        "sampler": "ddpm",
        "height": 256,
        "width": 256,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "save_name": "ddpm_output",
        "show_architecture": False,
        "save_model": False,
        "save_name": "experiment_b1"
    }
    command = build_launch_command(args)
    print("Command to run: " + " ".join(shlex.quote(part) for part in command))
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


def test_experiment_b2():
    """
    Test my own DDIM sampling implementation
    """
    args = {
        "is_nrt": True,
        "seed": 42,
        "device": "auto",
        "experiment": "custom_sampling",
        "prompts": [
            "a futuristic city at night",
            "a landscape with mountains and a river",
            "a futuristic city at night",
        ],
        "sampler": "ddim",
        "height": 256,
        "width": 256,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "save_name": "ddim_output",
        "show_architecture": False,
        "save_model": False,
        "save_name": "experiment_b2"
    }
    command = build_launch_command(args)
    print("Command to run: " + " ".join(shlex.quote(part) for part in command))
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


if __name__ == "__main__":
    test_experiment_a()
    test_experiment_b1()
    test_experiment_b2()




"""
command line usage example (expirement A):
python src/pretrained/launch.py --is_nrt=True --seed=42 --device=auto --experiment=baseline --prompts=a futuristic city at night --batch_size=3 --height=256 --width=256 --num_inference_steps=20 --guidance_scale=7.5 --save_name=experiment_a --show_architecture=False --save_model=False

command line usage example (expirement E):
python src/pretrained/launch.py --is_nrt=True --seed=42 --device=auto A COMPLETER
"""

"""
loading the class to play with it without having everything handled by the command line argulents, 
checking each component individually instead of the whole package
"""

