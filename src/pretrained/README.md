# Pretrained 
> A hands-on diffusion-model laboratory built around pretrained Stable Diffusion components and a full inference class for image generation.
>
> The part explores how much of image generation comes from the pretrained **UNet**, the **noise scheduler**, and the **sampling algorithm**, while implementing custom DDPM/DDIM sampling and experimenting with both full UNet fine-tuning and parameter-efficient **LoRA** adaptation. 


# Overview
This part is an experimental framework for understanding and modifying pretrained diffusion models rather than treating them as black boxes. It's also designed to deliver already production-level models for image generation, using inference on common already trained models.

The experiments are built around **Tiny-SD** and compare several configurations:
- Standard pretrained inference (also on **ddpm-cifar10-32**) with a dedidated class
- Custom DDPM sampling
- Custom DDIM sampling
- Custom noise schedules
- Full UNet fine-tuning
- LoRA-based UNet fine-tuning
- Reproducible generation through seeded inference

The main question explored by this part is:

> **How much of the final generative behavior comes from the pretrained UNet, and how much comes from the scheduler and sampling algorithm?** 

> **Can I generate good quality images with a pretrained model ?**


# Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Inference Class](#inference-class)
- [Fine-Tuning Pipeline](#fine-tuning-pipeline)
- [Launch Experiments](#launch-experiments)
  - [Experiment A — Baseline](#experiment-a--baseline)
  - [Experiment B — Custom Scheduler and Sampling](#experiment-b--custom-scheduler-and-sampling)
  - [Experiment C — Full or LoRA Fine-Tuning](#experiment-c--full-or-lora-fine-tuning)
- [CLI Usages](#cli-usages)
- [Model Saving and Reproducibility](#model-saving-and-reproducibility)


# Directory Structure
```text
pretrained/
├── fine_tuning.py
├── inference.py
└── launch.py
```

| File             | Purpose                                                                      |
| ---------------- | ---------------------------------------------------------------------------- |
| `inference.py`   | Loading pretrained pipelines, inspecting architectures and generating images |
| `fine_tuning.py` | Fine-tuning, LoRA, custom schedulers and diffusion samplers                  |
| `launch.py`      | Command-line interface and experiment orchestration                          |


# Inference Class

### Pipeline
```text
CLI arguments
      ↓
PretrainedInference
      ↓
StableDiffusionPipeline/DDPMPipeline loading
      ↓
VAE / UNet / Scheduler / Tokenizer / Text Encoder
      ↓
architecture inspection
      ↓
seeded text-to-image inference
      ↓
3 generated images
      ↓
image saving
      ↓
pipeline saving
```

### Models 
The standalone inference implementation supports both generic pretrained pipelines and Stable Diffusion text-to-image generation.

For DDPM inference, the default pretrained model is:
```text
google/ddpm-cifar10-32
```
The inference pipeline can generate a batch of images using a fixed random seed and save both the generated images and pipeline.
For Stable Diffusion inference, the default pretrained model is:
```text
segmind/tiny-sd
```

### Functions and Components
|                              |                                                 |
| ---------------------------- | ----------------------------------------------- |
| `load_pipeline()`            | Loading a previously saved pipeline             |
| `load_pretrained_pipeline()` | Loading the default pre-trained model           |
| `save_pipeline()`            | Pipeline serialization                          |
| `get_architecture()`         | Access to Stable Diffusion components           |
| `print_architecture()`       | Architecture inspection                         |
| `run_inference()`            | Model-specific inference arguments              |
| `save_images()`              | Batch and single-image output saving            |
| `run()`                      | Complete inference workflow                     |
| **DDPMPipeline**            | Pre-trained loading and image generation         |
| **StableDiffusionPipeline** | Pre-trained loading and text-to-image generation |
| **Scheduler**               | Inference through the loaded pipeline            |
| **VAE**                     | Stable Diffusion pipeline component availability |
| **UNet**                    | Stable Diffusion pipeline component availability |
| **Tokenizer**               | Text-conditioning pipeline availability          |
| **Text Encoder**            | Prompt-conditioning pipeline availability        |


# Fine-Tuning Pipeline
```text
                    PRETRAINED TINY-SD
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
           VAE       Text Encoder       UNet
            │              │              │
            │              │       ┌──────┴──────┐
            │              │       │             │
            │              │     Frozen      Fine-tuned
            │              │                    │
            │              │                  LoRA
            │              │                    │
            └──────────────┼────────────────────┘
                           │
                           ▼
                    Noise Scheduler
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
              DDPM Sampler        DDIM Sampler
                 │                   │
                 └─────────┬─────────┘
                           ▼
                          VAE
                           │
                           ▼
                         Image
```
This architecture makes it possible to change individual parts of the generation process while keeping the rest of the pretrained system fixed (VAE and text encoder frozen). The UNet can be either fully fine-tuned or adapted with LoRA, and the noise scheduler can be replaced with a custom implementation. Both DDPM and DDIM samplers are supported.

To create and deploy this, the `PretrainedFineTuning` class is used to load the pretrained components, freeze the VAE and text encoder, and allow for either full UNet fine-tuning or LoRA-based adaptation. The noise scheduler can also be replaced with a custom implementation, and both DDPM and DDIM samplers are supported. 
```py
class PretrainedFineTuning:
    __init__()
    freeze_vae()
    freeze_text_encoder()
    freeze_unet()
    encode_image()
    decode_image()
    encode_prompt()
    encode_labels()
    add_custom_noise()
    predict_noise()
    enable_lora()
    save_lora_adapter()
    load_lora_adapter()
    save_finetuned_model()
    load_finetuned_model()
    trainable_parameters()
    train_step()
    sample_sd()
    ddpm_step()
    ddim_step()
    sample_custom()
```
Then the `launch.py` script can be used to run the experiments with different configurations, including baseline inference, custom scheduler and sampling, full fine-tuning, and LoRA-based adaptation.

### Custom Scheduler
The project implements its own noise scheduler rather than relying exclusively on the scheduler provided by Diffusers. The full implementation is the same as the one used in the `diffusion_model.py` file : `NoiseScheduler`. The scheduler is responsible for controlling the noise levels during the diffusion process, and it can be customized to experiment with different noise schedules, with the following parameters:
- Number of diffusion timesteps
- Beta schedule
- Beta start
- Beta end
- Alpha values
- Cumulative alpha products
- Forward diffusion coefficients
- Reverse diffusion coefficients
- Two beta schedules are supported: Linear or Cosine

This makes it possible to experiment with different diffusion schedules without changing the UNet architecture.

### Custom DDPM and DDIM Sampling
Instead of using the `sample_sd` function to sample from the pretrained pipeline, the project implements its own `ddpm_step` and `ddim_step` functions and the `sample_custom` function to perform the sampling process. This allows for experimentation with different sampling algorithms while keeping the rest of the pipeline fixed.
- **DDPM (Denoising Diffusion Probabilistic Models)** is a method for generating images by iteratively denoising a random noise image
- **DDIM (Denoising Diffusion Implicit Models)** is a variant of DDPM that allows for faster sampling by using a deterministic process instead of a stochastic one (less steps, but still produces high-quality images)

### Diffusion Training Pipeline
The training process can be summarized as:
```text
                 Training Dataset
                        │
                        ▼
                 Image / Caption
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
             VAE            Text Encoder
              │                   │
              ▼                   ▼
           Latents             Embeddings
              │
              ▼
       Add Gaussian Noise
              │
              ▼
         Noisy Latents
              │
              ▼
             UNet
              │
              ▼
        Predicted Noise
              │
              ▼
        Diffusion Loss
              │
              ▼
        Backpropagation
              │
              ▼
        Updated Parameters
```
Depending on the experiment, the trainable parameters are either:
- **Full Fine-Tuning**: UNet parameters
- **LoRA**: LoRA adapter parameters only


### Fine-Tuning and LoRA

**Full Fine-Tuning**

**LoRA**


### Full Fine-Tuning vs LoRA


# Launch Experiments 
This part is organized around four main experiments
| Experiment                | UNet               | Scheduler | Sampler          | Training  |
| ------------------------- | ------------------ | --------- | ---------------- | --------- |
| **A — Baseline**          | Tiny-SD pretrained | Tiny-SD   | Diffusers        | None      |
| **B — Custom Sampling**   | Tiny-SD pretrained | Custom    | Custom DDPM/DDIM | None      |
| **C1 — Fine-Tuning**      | Fine-tuned         | Custom    | Custom           | Full UNet |
| **C2 — LoRA Fine-Tuning** | LoRA adapted       | Custom    | Custom           | LoRA only |

This structure allows the experiments to progressively modify the diffusion system rather than changing everything simultaneously.

It deliberately separates the different sources of generative behavior.
```text
                    GENERATIVE BEHAVIOR
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           UNet        Scheduler       Sampler
             │             │             │
       What to denoise   How noise      How the
       / learned        evolves         reverse
       representation   over time       process runs
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                         Image
```
This makes the experiments more informative than simply comparing different pretrained checkpoints.

### Experiment A — Baseline
The first experiment provides a reference point using the pretrained Tiny-SD pipeline.
```text
                 Tiny-SD
                    │
          ┌─────────┴─────────┐
          │                   │
         VAE               UNet
          │                   │
          └─────────┬─────────┘
                    │
             Tiny-SD Scheduler
                    │
                    ▼
             Diffusers Sampler
                    │
                    ▼
                  Image
```
No model parameters are modified.
This experiment establishes the baseline generation quality against which the custom scheduler, sampling and fine-tuning experiments can be compared.

### Experiment B — Custom Scheduler and Sampling
The second experiment replaces the default scheduler/sampling implementation with custom diffusion components.
The important constraint is that the UNet should remain compatible with the noise schedule it was trained on.
```text
Tiny-SD UNet
     │
     │ trained with
     ▼
Tiny-SD noise schedule
     │
     │ reproduce / recover
     ▼
Custom Noise Scheduler
     │
     ├───────────────┐
     ▼               ▼
 Custom DDPM     Custom DDIM
     │               │
     └───────┬───────┘
             ▼
            VAE
             │
             ▼
           Image
```
The scheduler therefore defines the noise levels used throughout the diffusion process, while the sampler controls how the reverse diffusion process is performed.

A diffusion UNet is trained to denoise samples according to a particular noise distribution and schedule.
**Scheduler compatibility** is therefore an important experimental variable, as using a different schedule during inference can lead to suboptimal results.

Conceptually:
```text
Training                                        Inference
   │                                               │
   ▼                                               ▼
Clean image                                     Schedule A
   │                                               │
   ▼                                               ▼
Add noise according to schedule A                  UNet
   │                                               │
   ▼                                               ▼
Noisy image                                     Predicted noise
   │                                               │
   ▼                                               ▼
UNet trained under schedule A                     Reverse diffusion
   │                                               │
   ▼                                               ▼
Predicted noise                                 Clean image
```
If the UNet was trained with schedule A but inference uses a substantially different schedule B, the model is being asked to operate under a distribution it was not trained for.
This makes scheduler compatibility an important experimental variable.

### Experiment C — Full or LoRA Fine-Tuning
The third experiment investigates what happens when the pretrained UNet is fully/LoRA fine-tuned on a new dataset/domain.

**Full Fine-Tuning**
```text
                 Tiny-SD
                    │
                    ▼
                 VAE
                    │
                    ▼
                UNet
                    │
              FULL TRAINING
                    │
                    ▼
             Domain-adapted
                  UNet
```
The training workflow includes:
1. Loading the pretrained diffusion model
2. Preparing the training dataset
3. Encoding images into latent space
4. Encoding the corresponding prompts
5. Adding noise according to the diffusion schedule
6. Predicting the added noise with the UNet
7. Computing the training loss
8. Updating the trainable parameters
9. Periodically generating samples
10. Saving the resulting model

The implementation exposes explicit methods for:
- VAE freezing
- Text encoder freezing
- UNet freezing
- Image encoding
- Image decoding
- Prompt encoding
- Label encoding
- Custom noise generation
- Noise prediction
- Model saving/loading

The available fine-tuning methods are defined in the project's PretrainedFineTuning class.

**Low-Rank Adaptation (LoRA)**
Instead of fully fine-tuning the UNet, LoRA allows for a parameter-efficient adaptation of the pretrained model by learning low-rank updates to the model's weights.
```text
Pretrained UNet
      │
      ├── Frozen weights W
      │
      └── Trainable LoRA adapters
                 │
                 ▼
             ΔW = BA
```
The effective weights become: $W' = W + \Delta W$ where: $\Delta W = BA$ and A and B are low-rank matrices.

**Full Fine-Tuning vs LoRA**
The main difference is the number of trainable parameters.
```text
From scratch    
████████████████████████
Full UNet
████████████████████████
LoRA
██
```
With LoRA:
- The original pretrained weights remain frozen
- Only the low-rank adaptation matrices are trained
- The number of trainable parameters is substantially reduced
- The original model can be reused for multiple adaptations

LoRA adapters can be saved independently. The implementation provides:
`enable_lora()`, `save_lora_adapter()`, `load_lora_adapter()` 
alongside the full model save/load functionality.

# CLI Usages
Examples can be found in the `NRT/NRT_inference/test.py` and `NRT/NRT_fine_tuning/test.py` files, which contains a set of functions to launch the different experiments. The experiments can be launched by running the script and uncommenting the desired function call in the `if __name__ == "__main__":` block.

### Inference
```bash
usage: inference.py [-h] [--is_nrt [IS_NRT]] [--seed SEED] [--device {auto,cpu,cuda}] [--model_type {ddpm,stable_diffusion}] [--num_inference_steps NUM_INFERENCE_STEPS] [--description DESCRIPTION] [--save_name SAVE_NAME] [--batch_size BATCH_SIZE] [--guidance_scale GUIDANCE_SCALE] [--height HEIGHT] [--width WIDTH] [--save_model [SAVE_MODEL]] [--show_architecture [SHOW_ARCHITECTURE]]

Run inference on a pretrained model

options:
  -h, --help            
            show this help message and exit
  --is_nrt [IS_NRT]     
            modify the project root saving path to be compatible with NRT
  --seed SEED           
            Random seed for reproducible results
  --device {auto,cpu,cuda} 
            Device to run inference on (auto selects cuda if available)
  --model_type {ddpm,stable_diffusion} 
            Type of model to use for inference
  --num_inference_steps 
            NUM_INFERENCE_STEPS Number of inference steps
  --description 
            DESCRIPTION Description for stable diffusion model
  --save_name SAVE_NAME 
            Name of the output image file (without extension), in case of multpile images, it become the folder name in which the images will be saved as image_0.png, image_1.png, ...
  --batch_size BATCH_SIZE 
            Number of images to generate in a batch
  --guidance_scale GUIDANCE_SCALE 
            Classifier-free guidance scale for text-to-image models (stable diffusion)
  --height HEIGHT       
            Height of generated image (stable diffusion)
  --width WIDTH         
            Width of generated image (stable diffusion)
  --save_model [SAVE_MODEL]
            If set, will save the model parameters for future reuse without re-downloading the whole thing
  --show_architecture [SHOW_ARCHITECTURE]
            If set, will print the architecture of the model
```

### Fine-Tuning
```bash
usage: launch.py [-h] [--is_nrt [IS_NRT]] [--seed SEED] [--device {auto,cpu,cuda}] [--experiment {baseline,custom_scheduler_and_sampling,finetune,lora}] [--is_training [IS_TRAINING]] [--prompts PROMPTS [PROMPTS ...]] [--batch_size BATCH_SIZE] [--sampler {ddpm,ddim}] [--height HEIGHT] [--width WIDTH] [--num_inference_steps NUM_INFERENCE_STEPS] [--guidance_scale GUIDANCE_SCALE] [--eta ETA] [--timesteps TIMESTEPS] [--beta_schedule {linear,cosine}] [--beta_start BETA_START] [--beta_end BETA_END] [--cosine_s COSINE_S] [--training_batch_size TRAINING_BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--gradient_clip GRADIENT_CLIP] [--lora_rank LORA_RANK] [--lora_alpha LORA_ALPHA] [--dataset {cifar10,imagefolder}] [--subset_size SUBSET_SIZE] [--show_architecture [SHOW_ARCHITECTURE]] [--save_model [SAVE_MODEL]] [--lora_name LORA_NAME] [--save_name SAVE_NAME]

Generative Lab - Pretrained Diffusion Experiments

options:
  -h, --help            
            show this help message and exit
  --is_nrt [IS_NRT]     
            modify the project root saving path to be compatible with NRT
  --seed SEED           
            Random seed for reproducible results
  --device {auto,cpu,cuda}
            Device to run fine-tuning on (auto selects cuda if available)
  --experiment {baseline,custom_scheduler_and_sampling,finetune,lora}
            Type of experiment to run for fine-tuning
--is_training [IS_TRAINING]
            If set, will run the training loop for fine-tuning
  --prompts PROMPTS [PROMPTS ...]
            List of prompts for image generation
  --batch_size BATCH_SIZE
            Number of images to generate in a batch = per prompt for baseline
  --sampler {ddpm,ddim}
            Sampler to use for inference during fine-tuning (my own implementation of ddpm or ddim)
  --height HEIGHT       
            Height of the generated images
  --width WIDTH         
            Width of the generated images
  --num_inference_steps NUM_INFERENCE_STEPS
            Number of inference steps for image generation
  --guidance_scale GUIDANCE_SCALE
            Guidance scale for image generation (default: 7.5, no guidance)
  --eta ETA             
            Eta parameter for DDIM sampler (default: 0.0, deterministic)
  --timesteps TIMESTEPS
            Number of timesteps for the noise scheduler
  --beta_schedule {linear,cosine}
            Beta schedule for the noise scheduler
  --beta_start BETA_START
            Start value for the beta schedule
  --beta_end BETA_END   
            End value for the beta schedule
  --cosine_s COSINE_S   
            Cosine schedule parameter
  --training_batch_size TRAINING_BATCH_SIZE
            Batch size for training
  --epochs EPOCHS       
            Number of epochs for training
  --learning_rate LEARNING_RATE
            Learning rate for training
  --weight_decay WEIGHT_DECAY
            Weight decay for training
  --gradient_clip GRADIENT_CLIP
            Gradient clipping value
  --lora_rank LORA_RANK
            LoRA rank
  --lora_alpha LORA_ALPHA
            LoRA alpha
  --dataset {cifar10,imagefolder}
            Dataset to use for training (cifar10, celebA, imagefolder (need to be prepared correctly in data/imagefolder previously))
  --subset_size SUBSET_SIZE
            Subset size for training (only for mnist and cifar10)
  --show_architecture [SHOW_ARCHITECTURE]
            If set, will print the architecture of the model
  --save_model [SAVE_MODEL]
            If set, will save the model parameters for future reuse without re-downloading the whole thing
  --lora_name LORA_NAME
            Name of the LoRA model to save
  --save_name SAVE_NAME
            Name of the output image file (without extension), in case of multiple images, it becomes the folder name in which the images will be saved as image_0.png, image_1.png, ...
```


# Model Saving and Reproducibility
Both complete pipelines and LoRA adapters can be serialized.
The inference implementation supports:
```py
save_pipeline()
load_pipeline()
save_images()
```
while the fine-tuning implementation additionally supports:
```py
save_lora_adapter()
load_lora_adapter()
save_finetuned_model()
load_finetuned_model()
```
This allows trained models to be reused without downloading and reconstructing the entire pretrained pipeline every time.

Generation can be made reproducible using a fixed random seed.
```bash
--seed 42
```
The seed is propagated to the generation process through a PyTorch generator.
For deterministic experiments, the same `model + scheduler + sampler + seed + inference parameters`
should be kept constant.

Note that complete bit-level reproducibility can still depend on the hardware and backend configuration.