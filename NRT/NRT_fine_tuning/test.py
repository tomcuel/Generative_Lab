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
        "experiment": "custom_scheduler_and_sampling",
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
        "timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "cosine_s": 0.008,
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
        "experiment": "custom_scheduler_and_sampling",
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
        "timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "cosine_s": 0.008,
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


def test_experiment_c():
    """
    Test the LoRA fine-tuning experiment on the cifar10 dataset to see if it's working correctly
    Warning: Full fine-tuning is way too long even for such a small dataset subsample for my laptop config, 
    but LoRa uses all the same tools, functions, and methods, so if LoRa works, the rest should work too.
    """
    # Inference before fine-tuning to see the difference in the generated images after fine-tuning
    args = {
        "is_nrt": True,
        "seed": 42,
        "device": "auto",
        "experiment": "custom_scheduler_and_sampling",
        "is_training": False,
        "prompts": [
            "an airplane",
            "an automobile",    
            "a bird"
        ],
        "sampler": "ddim",
        "height": 256,
        "width": 256,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "cosine_s": 0.008,
        "show_architecture": False,
        "save_model": False,
        "save_name": "experiment_c_before_finetuning"
    }
    command = build_launch_command(args)
    print("Command to run: " + " ".join(shlex.quote(part) for part in command))
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    # Fine Tuning and Inference with LoRA
    args = {
        "is_nrt": True,
        "seed": 42,
        "device": "auto",
        "experiment": "lora",
        "is_training": True,
        "prompts": [
            "an airplane",
            "an automobile",
            "a bird"
        ],
        "sampler": "ddim",
        "height": 256,
        "width": 256,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "cosine_s": 0.008,
        "epochs": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "dataset": "cifar10",
        "subset_size": 100,
        "show_architecture": False,
        "save_model": True,
        "lora_name": "default",
        "save_name": "experiment_c_after_finetuning"
    }
    command = build_launch_command(args)
    print("Command to run: " + " ".join(shlex.quote(part) for part in command))
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)


if __name__ == "__main__":
    #test_experiment_a()
    #test_experiment_b1()
    #test_experiment_b2()
    test_experiment_c()


"""
command line usage example (baseline inference):
python src/pretrained/launch.py --is_nrt=True --seed=42 --device=auto --experiment=baseline --prompts=a futuristic city at night --batch_size=3
--height=256 --width=256 --num_inference_steps=20 --guidance_scale=7.5 
--show_architecture=False --save_model=False --save_name=experiment_a 

command line usage example (Fine-tuning with LoRA):
python src/pretrained/launch.py --is_nrt=True --seed=42 --device=auto --experiment lora --is_training=True --prompts 'an airplane' 'an automobile' 'a bird' --sampler ddim 
--height 256 --width 256 --num_inference_steps 20 --guidance_scale 7.5 
--timesteps 1000 --beta_schedule linear --beta_start 0.0001 --beta_end 0.02 --cosine_s 0.008 
--epochs 1 --learning_rate 0.0001 --weight_decay 0.01 --gradient_clip 1.0 
--dataset cifar10 --subset_size 100 
--show_architecture False --save_model True --lora_name default --save_name experiment_c
"""

"""
loading the class to play with it without having everything handled by the command line argulents, 
checking each component individually instead of the whole package
"""

