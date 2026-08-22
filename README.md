# Generative Lab
> An experimental PyTorch framework for studying, implementing, and fine-tuning deep generative models   from classical latent-variable and adversarial approaches to modern diffusion models and pretrained image generators.

The project is organized around two complementary goals:
1. **Build generative models from scratch** to understand their architectures, objectives, training procedures, and limitations (**VAEs**, **GANs**, **Diffusion Models**)
2. **Work with pretrained diffusion models** to investigate inference, sampling strategies, and fine-tuning techniques such as LoRA

The repository therefore combines educational implementations, controlled experiments, reusable model components, and pretrained-model workflows.

The performance of the models from scratch is not the primary focus of this project since I don't have the computational resources. Instead, the emphasis is on **understanding the underlying principles** and **providing a modular framework** for experimentation. On the other hand, the pretrained diffusion models are already well-trained and can generate high-quality images, allowing for meaningful experiments on inference and fine-tuning.

This repository is part of a broader collection of machine-learning projects focused on implementing models from first principles, experimenting with modern architectures, and understanding the trade-offs between research-oriented implementations and practical pretrained-model workflows.


# Table of Contents
- [Table of Contents](#table-of-contents)
- [Directory Structure](#directory-structure)
- [Link to folders](#link-to-folders)
- [Architecture](#architecture)
- [Models from Scratch](#models-from-scratch)
    - [Variational Autoencoders (VAEs)](#variational-autoencoders-vaes)
    - [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
    - [Diffusion Models](#diffusion-models)
    - [A word on Hybrid Models](#a-word-on-hybrid-models)
- [Pretrained Diffusion Models](#pretrained-diffusion-models)
    - [Inference](#inference)
    - [Fine-tuning](#fine-tuning)
- [Installation](#installation)
- [CLI Usage](#cli-usage)
- [Outputs](#outputs)
- [What This Project Explores](#what-this-project-explores)


# Directory Structure
```text
Generative_Lab/
├── data/
│   ├── cifar10/
│   ├── configs/
│   ├── fashion_mnist/
│   ├── imagefolder/
│   ├── mnist/
│   ├── models_parameters/
│   ├── output/
│   └── readme_pictures/
│ 
├── notebooks/
│   ├── diffusion_models/
│   ├── GANs/
│   ├── hybrids/
│   └── VAEs/
│ 
├── NRT/
│   ├── NRT_diffusion_models/
│   ├── NRT_fine_tuning/
│   ├── NRT_GANs/
│   ├── NRT_inference/
│   ├── NRT_load/
│   ├── NRT_utils/
│   └── NRT_VAEs/
│ 
└── src/
    ├── data/
    │   ├── load.py
    │   └── utils.py
    │
    ├── models/
    │   ├── diffusion_models.py
    │   ├── GANs.py
    │   └── VAEs.py
    │
    ├── pretrained/
    │   ├── fine_tuning.py
    │   ├── inference.py
    │   └── launch.py
    │   
    └── launcher.py
```

| Directory         | Purpose                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| `data/`           | Datasets, configurations, model parameters, generated outputs, and README assets |
| `notebooks/`      | Interactive experiments and research-oriented implementations                    |
| `NRT/`            | Non-real-time implementations used to test and validate models and procedures    |
| `src/data/`       | Reusable data loading and preprocessing utilities                                |
| `src/models/`     | Reusable implementations of VAEs, GANs, and diffusion architectures              |
| `src/pretrained/` | Pretrained inference, sampling, and fine-tuning workflows                        |
| `src/launcher.py` | Main launcher for running experiments                                            |


# Repository Navigation
### Data
* [**data**](./data/)
    * [cifar10](./data/cifar10/)
    * [configs](./data/configs/)
    * [fashion_mnist](./data/fashion_mnist/)
    * [imagefolder](./data/imagefolder/)
    * [mnist](./data/mnist/)
    * [models_parameters](./data/models_parameters/)
    * [output](./data/output/)
    * [readme_pictures](./data/readme_pictures/)

### Notebooks
* [**notebooks**](./notebooks/) 
    * [diffusion_models](./notebooks/diffusion_models/)
    * [GANs](./notebooks/GANs/)
    * [hybrids](./notebooks/hybrids/)
    * [VAEs](./notebooks/VAEs/)

### Non-Real-Time Implementations
* [**NRT**](./NRT/)
    * [NRT_diffusion_models](./NRT/NRT_diffusion_models/)
    * [NRT_fine_tuning](./NRT/NRT_fine_tuning/)
    * [NRT_GANs](./NRT/NRT_GANs/)
    * [NRT_inference](./NRT/NRT_inference/)
    * [NRT_load](./NRT/NRT_load/)
    * [NRT_utils](./NRT/NRT_utils/)
    * [NRT_VAEs](./NRT/NRT_VAEs/)

### Source
* [**src**](./src/)
    * [data](./src/data/)
        * [load.py](./src/data/load.py)
        * [utils.py](./src/data/utils.py)
    * [models](./src/models/)
        * [diffusion_models.py](./src/models/diffusion_models.py)
        * [GANs.py](./src/models/GANs.py)
        * [VAEs.py](./src/models/VAEs.py)
    * [pretrained](./src/pretrained/)
        * [fine_tuning.py](./src/pretrained/fine_tuning.py)
        * [inference.py](./src/pretrained/inference.py)
        * [launch.py](./src/pretrained/launch.py)
    * [launcher.py](./src/launcher.py)


# Architecture
At a high level, the complete project can be represented as:
```text
                              ┌─────────────────────┐
                              │    Generative Lab   │
                              └──────────┬──────────┘
                                         │
              ┌──────────────────────────┼─────────────────────────┐
              │                          │                         │
              ▼                          ▼                         ▼
       From-Scratch Models          Pretrained Models          Experiments
              │                          │                         │
       ┌──────┼──────┐             ┌─────┴─────┐            ┌──────┴──────┐
       │      │      │             │           │            │             │
      VAE    GAN  Diffusion     Inference   Fine-tuning   Notebooks      NRT
       │      │      │             │           │
       │      │      │             │      ┌────┴────┐
       │      │      │             │      │         │
       │      │      │             │    Full       LoRA
       │      │      │             │    U-Net.      │
       │      │      │             │      │         │
       └──────┴──────┴─────────────┴──────────┬─────┘
                                              │
                                              ▼
                                      Sampling / Generation
                                              │
                                              ▼
                                       Generated Images
```


# Models from Scratch
The `src/models/` module contains implementations covering several generations of deep generative modeling.

| Family | Implementations | Learning principle | Main strengths | Main limitations |
|--------|-----------------|--------------------|----------------|------------------|
| **VAEs** | MLP-VAE, CNN-VAE, VQ-VAE | Variational inference | Stable training, structured latent representations | Can produce blurry samples |
| **GANs** | MLP-GAN, DCGAN, Unrolled GAN, CGAN, WGAN, StyleGAN | Adversarial optimization | Sharp and realistic samples | Training instability, mode collapse |
| **Diffusion Models** | CNN, Residual U-Net, Latent Diffusion, DDPM, DDIM | Iterative denoising | High-quality and diverse generation | Sampling can be computationally expensive |
| **Hybrid Models** | AAE, Diffusion-GAN, VAE-GAN | Combination of paradigms | Combines complementary properties | Increased architectural and optimization complexity |

### Variational Autoencoders (VAEs)
Variational Autoencoders learn a probabilistic latent representation of the input data.
The repository explores several variants, including:
- MLP-based VAEs
- CNN-based VAEs
- Vector-Quantized VAEs (VQ-VAEs)

The experiments focus on the relationship between:
```text
Input
  │
  ▼
Encoder
  │
  ▼
Latent representation
  │
  ▼
Decoder
  │
  ▼
Reconstructed / generated image
```
VAEs provide a useful foundation for understanding latent-variable generative modeling and representation learning.

### Generative Adversarial Networks (GANs)
Generative Adversarial Networks approach generation as a game between two neural networks:
```text
                 Random Noise
                      │
                      ▼
                 ┌─────────┐
                 │Generator│
                 └────┬────┘
                      │
                Generated Image
                      │
                      ▼
                 ┌─────────┐
Real Image ─────►│Discrim. │
                 └────┬────┘
                      │
                      ▼
                 Real / Fake
```
The repository implements several GAN variants to explore how changes to the adversarial objective and architecture affect training and generation:
* MLP-GAN
* DCGAN
* Conditional GAN (CGAN)
* Wasserstein GAN (WGAN)
* Unrolled GAN
* StyleGAN

These experiments are particularly useful for studying:
* adversarial optimization
* discriminator/generator dynamics
* conditional generation
* mode collapse
* training stability
* image quality

### Diffusion Models
Diffusion models use a fundamentally different generation mechanism.
Instead of directly mapping noise to an image, the model learns to progressively reverse a noise-adding process.
```text
Forward process

Clean image
    │
    ▼
 add noise
    │
    ▼
 more noise
    │
    ▼
 Gaussian noise


Reverse process

Gaussian noise
    │
    ▼
 denoise
    │
    ▼
 denoise
    │
    ▼
 generated image
```
The repository explores both the architecture and the sampling procedure, including:
* DDPM
* DDIM
* CNN-based diffusion models
* Residual U-Net architectures
* Latent diffusion
* Custom noise schedules

This makes it possible to study diffusion models at several levels rather than relying exclusively on high-level pretrained pipelines.

### A Word on Hybrid Models
The notebook experiments also investigate combinations of generative paradigms.
These include:
* **Adversarial Autoencoders (AAE)**
* **VAE-GAN**
* **Diffusion-GAN**

The objective is to explore whether combining different learning principles can leverage their complementary strengths.
For example:
```text
                  ┌─────────────┐
                  │   Encoder   │
                  └──────┬──────┘
                         │
                    Latent Space
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
          Decoder              Discriminator
              │                     │
              ▼                     │
        Generated Image ◄───────────┘
```
These models are primarily explored through notebooks rather than exposed as the main reusable model API, because I was too lazy to implement several others instance of models without really pushing the training to explore their full potential. However, the experiments are still useful for understanding how different generative paradigms can be combined.


# Pretrained Diffusion Models
The second major part of the project focuses on pretrained diffusion models.
Rather than treating pretrained models as black boxes, the goal is to expose and modify the components responsible for generation.
The experiments use pretrained diffusion models such as:
* **Tiny-SD**
* **DDPM CIFAR-10 (32×32)**

The pretrained-model framework investigates several questions:
> **How much of the final generated image comes from the pretrained U-Net, and how much comes from the scheduler and sampling algorithm?**
and:
> **How much can the behavior of a pretrained diffusion model be modified without training the entire model from scratch?**

### Inference
The repository supports standard pretrained inference as well as custom inference procedures.
The workflow can be summarized as:
```text
Prompt / Conditioning
        │
        ▼
Pretrained Model
        │
        ├───────────────┐
        │               │
        ▼               ▼
   DDPM Sampling    DDIM Sampling
        │               │
        └───────┬───────┘
                ▼
         Generated Image
```
Experiments compare:
* standard pretrained inference
* custom DDPM sampling
* custom DDIM sampling
* custom noise schedules

This provides a controlled environment for understanding how sampling affects the final result.

### Fine-tuning
The project also explores adapting pretrained diffusion models to new generation tasks.
Two main approaches are investigated:

**Full U-Net fine-tuning:** 
the pretrained U-Net is updated directly during training, providing maximum flexibility but requires substantially more trainable parameters and computational resources.
```text
Pretrained U-Net
      │
      ▼
 Fine-tuning
      │
      ▼
Task-adapted U-Net
```

**LoRA fine-tuning:**
Low-Rank Adaptation (LoRA) introduces trainable low-rank updates while keeping most of the pretrained model frozen, to investigate the trade-off between trainable parameters, computational cost, adaptation speed, and generated-image quality.
```text
              Pretrained weights
                     │
                     │ frozen
                     ▼
Input ─────────► Pretrained U-Net ─────────► Output
                     │
                     ▲
                     │
                 LoRA layers
                  trainable
```


# Installation
1. Clone the repository:
```bash
git clone git@github.com:tomcuel/Generative_Lab.git
cd Generative_Lab
```
2. Create a python virtual environment: 
```bash
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
```
3. Install the requirements:
```bash
pip -m pip install -r requirements.txt
```
4. Make sure to have Jupyter Notebook installed to run the `.ipynb` experimental files


# CLI Usage
Experiments can be launched through:
```bash
python src/launcher.py --CLI_arguments
```

The exact arguments depend on the experiment configuration exposed by the launcher
A typical workflow is:

```text
Select experiment
      │
      ▼
Load configuration
      │
      ▼
Load dataset
      │
      ▼
Build model
      │
      ▼
Train / load pretrained weights
      │
      ▼
Generate samples
      │
      ▼
Save outputs
```

Here is the fully detailed CLI arguments obtained by running:
```bash
python src/launcher.py --help
```
```text
usage: launcher.py [-h] [--launch_mode {vae,gan,diffusion,inference,finetuning}] [--seed SEED] [--device {auto,cpu,cuda}] [--name NAME] [--dataset {cifar10,fashion_mnist,imagefolder,mnist}] [--subset_size SUBSET_SIZE] [--is_training [IS_TRAINING]] [--timesteps TIMESTEPS] [--beta_schedule {linear,cosine}] [--beta_start BETA_START] [--beta_end BETA_END] [--cosine_s COSINE_S] [--training_batch_size TRAINING_BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] [--step_size STEP_SIZE] [--weight_decay WEIGHT_DECAY] [--batch_size BATCH_SIZE] [--height HEIGHT] [--width WIDTH] [--num_inference_steps NUM_INFERENCE_STEPS] [--guidance_scale GUIDANCE_SCALE] [--show_architecture [SHOW_ARCHITECTURE]] [--save_model [SAVE_MODEL]] [--vae_config VAE_CONFIG] [--vae_model_type {vae,vqvae,fastvae}] [--vae_architecture {mlp,cnn}] [--vae_reconstruction_loss {mse,bce}] [--vae_input_dim VAE_INPUT_DIM] [--vae_hidden_dims VAE_HIDDEN_DIMS [VAE_HIDDEN_DIMS ...]] [--vae_latent_dim VAE_LATENT_DIM] [--vae_image_channels VAE_IMAGE_CHANNELS] [--vae_image_size VAE_IMAGE_SIZE] [--vae_kernel_size VAE_KERNEL_SIZE] [--vae_stride VAE_STRIDE] [--vae_padding VAE_PADDING] [--vae_num_embeddings VAE_NUM_EMBEDDINGS] [--vae_embedding_dim VAE_EMBEDDING_DIM] [--vae_beta_vq VAE_BETA_VQ] [--vae_dropout VAE_DROPOUT] [--vae_use_batchnorm [VAE_USE_BATCHNORM]] [--vae_beta_kl VAE_BETA_KL] [--vae_gamma VAE_GAMMA] [--gan_config GAN_CONFIG] [--gan_architecture {GAN,CGAN,DCGAN,MLP_UnrolledGAN,DC_UnrolledGAN,StyleGAN}] [--gan_loss {Default,Wasserstein,LeastSquare}] [--gan_latent_dim GAN_LATENT_DIM] [--gan_hidden_dims GAN_HIDDEN_DIMS [GAN_HIDDEN_DIMS ...]] [--gan_image_size GAN_IMAGE_SIZE] [--gan_image_channels GAN_IMAGE_CHANNELS] [--gan_kernel_size GAN_KERNEL_SIZE] [--gan_stride GAN_STRIDE] [--gan_padding GAN_PADDING] [--gan_noise_coef GAN_NOISE_COEF] [--gan_num_classes GAN_NUM_CLASSES] [--gan_unrolled_steps GAN_UNROLLED_STEPS] [--gan_weight_clip GAN_WEIGHT_CLIP] [--gan_gradient_penalty_lambda GAN_GRADIENT_PENALTY_LAMBDA] [--gan_n_critic GAN_N_CRITIC] [--gan_lsgan_lambda GAN_LSGAN_LAMBDA] [--gan_style_dim GAN_STYLE_DIM] [--gan_kernel_size_style_gen GAN_KERNEL_SIZE_STYLE_GEN] [--gan_stride_style_gen GAN_STRIDE_STYLE_GEN] [--gan_padding_style_gen GAN_PADDING_STYLE_GEN] [--gan_noise_weight GAN_NOISE_WEIGHT] [--gan_mixing_prob GAN_MIXING_PROB] [--gan_dropout GAN_DROPOUT] [--gan_use_batchnorm [GAN_USE_BATCHNORM]] [--gan_spectral_norm_on [GAN_SPECTRAL_NORM_ON]] [--gan_beta1 GAN_BETA1] [--gan_beta2 GAN_BETA2] [--gan_is_ema [GAN_IS_EMA]] [--gan_ema_decay GAN_EMA_DECAY] [--diffusion_config DIFFUSION_CONFIG] [--diffusion_model_type {cnn,res_unet}] [--diffusion_loss {mse,l1}] [--diffusion_num_classes DIFFUSION_NUM_CLASSES] [--diffusion_cond_drop_prob DIFFUSION_COND_DROP_PROB] [--diffusion_guidance_scale DIFFUSION_GUIDANCE_SCALE] [--diffusion_image_size DIFFUSION_IMAGE_SIZE] [--diffusion_image_channels DIFFUSION_IMAGE_CHANNELS] [--diffusion_base_channels DIFFUSION_BASE_CHANNELS] [--diffusion_channel_mults DIFFUSION_CHANNEL_MULTS [DIFFUSION_CHANNEL_MULTS ...]] [--diffusion_time_emb_dim DIFFUSION_TIME_EMB_DIM] [--diffusion_time_width_coef DIFFUSION_TIME_WIDTH_COEF] [--diffusion_use_attention [DIFFUSION_USE_ATTENTION]] [--diffusion_attention_resolutions DIFFUSION_ATTENTION_RESOLUTIONS [DIFFUSION_ATTENTION_RESOLUTIONS ...]] [--diffusion_num_heads DIFFUSION_NUM_HEADS] [--diffusion_dropout DIFFUSION_DROPOUT] [--diffusion_kernel_size DIFFUSION_KERNEL_SIZE] [--diffusion_stride DIFFUSION_STRIDE] [--diffusion_padding DIFFUSION_PADDING] [--diffusion_use_batch_norm [DIFFUSION_USE_BATCH_NORM]] [--diffusion_num_groups DIFFUSION_NUM_GROUPS] [--diffusion_eps_groupnorm DIFFUSION_EPS_GROUPNORM] [--diffusion_down_kernel_size DIFFUSION_DOWN_KERNEL_SIZE] [--diffusion_down_stride DIFFUSION_DOWN_STRIDE] [--diffusion_down_padding DIFFUSION_DOWN_PADDING] [--diffusion_down_num_res_blocks DIFFUSION_DOWN_NUM_RES_BLOCKS] [--diffusion_up_kernel_size DIFFUSION_UP_KERNEL_SIZE] [--diffusion_up_stride DIFFUSION_UP_STRIDE] [--diffusion_up_padding DIFFUSION_UP_PADDING] [--diffusion_up_num_res_blocks DIFFUSION_UP_NUM_RES_BLOCKS] [--diffusion_learning_rate DIFFUSION_LEARNING_RATE] [--diffusion_beta1 DIFFUSION_BETA1] [--diffusion_beta2 DIFFUSION_BETA2] [--diffusion_use_torch_compile [DIFFUSION_USE_TORCH_COMPILE]] [--diffusion_compile_mode {default,reduce-overhead,max-autotune}] [--diffusion_use_ddim [DIFFUSION_USE_DDIM]] [--diffusion_ddim_steps DIFFUSION_DDIM_STEPS] [--diffusion_use_ema [DIFFUSION_USE_EMA]] [--diffusion_ema_decay DIFFUSION_EMA_DECAY] [--diffusion_use_latent_diffusion [DIFFUSION_USE_LATENT_DIFFUSION]] [--diffusion_latent_dim DIFFUSION_LATENT_DIM] [--diffusion_latent_hidden_dim DIFFUSION_LATENT_HIDDEN_DIM] [--diffusion_latent_kernel_size DIFFUSION_LATENT_KERNEL_SIZE] [--diffusion_latent_stride DIFFUSION_LATENT_STRIDE] [--diffusion_latent_padding DIFFUSION_LATENT_PADDING] [--diffusion_latent_scale_factor DIFFUSION_LATENT_SCALE_FACTOR] [--inference_model_type {ddpm,stable_diffusion}] [--inference_description INFERENCE_DESCRIPTION] [--inference_batch_size INFERENCE_BATCH_SIZE] [--finetuning_experiment {baseline,custom_scheduler_and_sampling,finetune,lora}] [--finetuning_prompts FINETUNING_PROMPTS [FINETUNING_PROMPTS ...]] [--finetuning_sampler {ddpm,ddim}] [--finetuning_eta FINETUNING_ETA] [--is_finetuning [IS_FINETUNING]] [--finetuning_gradient_clip FINETUNING_GRADIENT_CLIP] [--finetuning_lora_rank FINETUNING_LORA_RANK] [--finetuning_lora_alpha FINETUNING_LORA_ALPHA] [--finetuning_lora_name FINETUNING_LORA_NAME]

Generative Lab - Project Launcher

options:
  -h, --help            
            show this help message and exit
  --launch_mode {vae,gan,diffusion,inference,finetuning}
            Mode to launch the project in (vae, gan, diffusion, inference, finetuning)
  --seed SEED           
            Random seed for reproducible results
  --device {auto,cpu,cuda}
            Device to run inference on (auto selects cuda if available)
  --name NAME           
            Name of the experiment (used for saving outputs, configurations, models, etc.)

  --dataset {cifar10,fashion_mnist,imagefolder,mnist}
            Dataset to use for training (cifar10, celebA, imagefolder (need to be prepared correctly in data/imagefolder previously))
  --subset_size SUBSET_SIZE
            Subset size for training (only for mnist and cifar10)

  --is_training [IS_TRAINING]
            If set, will run the training loop

  --timesteps TIMESTEPS
            Number of diffusion timesteps
  --beta_schedule {linear,cosine}
            Noise schedule used by the diffusion process
  --beta_start BETA_START
            Start value of the beta schedule
  --beta_end BETA_END   
            End value of the beta schedule
  --cosine_s COSINE_S   
            Small offset used in the cosine beta schedule

  --training_batch_size TRAINING_BATCH_SIZE
            Batch size for training
  --epochs EPOCHS       
            Number of epochs for training
  --learning_rate LEARNING_RATE
            Learning rate for training
  --step_size STEP_SIZE
            Step size for learning rate scheduler in training
  --weight_decay WEIGHT_DECAY
            Weight decay for training

  --batch_size BATCH_SIZE
            Number of images to generate in a batch = per prompt for baseline
  --height HEIGHT       
            Height of the generated images
  --width WIDTH         
            Width of the generated images
  --num_inference_steps NUM_INFERENCE_STEPS
            Number of inference steps for image generation
  --guidance_scale GUIDANCE_SCALE
            Guidance scale for image generation (default: 7.5, no guidance)

  --show_architecture [SHOW_ARCHITECTURE]
            If set, will print the architecture of the model
  --save_model [SAVE_MODEL]
            If set, will save the model parameters for future reuse without re-downloading the whole thing

  --vae_config VAE_CONFIG
            Path to the VAE configuration file (e.g. data/configs/vae_config.yaml). If not provided, CLI arguments will be used to create the VAEConfig.
  --vae_model_type {vae,vqvae,fastvae}
            Type of VAE model to use
  --vae_architecture {mlp,cnn}
            Architecture of the VAE model
  --vae_reconstruction_loss {mse,bce}
            Reconstruction loss for the VAE model
  --vae_input_dim VAE_INPUT_DIM
            Input dimension for MLP VAE (flattened image size)
  --vae_hidden_dims VAE_HIDDEN_DIMS [VAE_HIDDEN_DIMS ...]
            Hidden dimensions for MLP VAE (list of integers)
  --vae_latent_dim VAE_LATENT_DIM
            Latent dimension for the VAE model
  --vae_image_channels VAE_IMAGE_CHANNELS
            Number of image channels for CNN VAE
  --vae_image_size VAE_IMAGE_SIZE
            Image size for CNN VAE (assumes square images)
  --vae_kernel_size VAE_KERNEL_SIZE
            Kernel size for CNN VAE
  --vae_stride VAE_STRIDE
            Stride for CNN VAE
  --vae_padding VAE_PADDING
            Padding for CNN VAE
  --vae_num_embeddings VAE_NUM_EMBEDDINGS
            Number of embeddings for VQ-VAE
  --vae_embedding_dim VAE_EMBEDDING_DIM
            Embedding dimension for VQ-VAE
  --vae_beta_vq VAE_BETA_VQ
            Beta parameter for VQ-VAE
  --vae_dropout VAE_DROPOUT
            Dropout rate for the VAE model
  --vae_use_batchnorm [VAE_USE_BATCHNORM]
            If set, will use batch normalization in the VAE model
  --vae_beta_kl VAE_BETA_KL
            Beta parameter for KL divergence in VAE training
  --vae_gamma VAE_GAMMA
            Gamma parameter for VAE training

  --gan_config GAN_CONFIG
            Path to the GAN configuration file (e.g. data/configs/gan_config.yaml). If not provided, CLI arguments will be used to create the GANConfig.
  --gan_architecture {GAN,CGAN,DCGAN,MLP_UnrolledGAN,DC_UnrolledGAN,StyleGAN}
            Architecture of the GAN model
  --gan_loss {Default,Wasserstein,LeastSquare}
            Loss function for the GAN model
  --gan_latent_dim GAN_LATENT_DIM
            Latent dimension for the GAN model
  --gan_hidden_dims GAN_HIDDEN_DIMS [GAN_HIDDEN_DIMS ...]
            Hidden dimensions for MLP GAN (list of integers)
  --gan_image_size GAN_IMAGE_SIZE
            Image size for DCGAN (assumes square images)
  --gan_image_channels GAN_IMAGE_CHANNELS
            Number of image channels for DCGAN and StyleGAN
  --gan_kernel_size GAN_KERNEL_SIZE
            Kernel size for DCGAN and StyleGAN discriminator
  --gan_stride GAN_STRIDE
            Stride for DCGAN and StyleGAN discriminator
  --gan_padding GAN_PADDING
            Padding for DCGAN and StyleGAN discriminator
  --gan_noise_coef GAN_NOISE_COEF
            Noise coefficient for DCGAN and StyleGAN
  --gan_num_classes GAN_NUM_CLASSES
            Number of classes for CGANs (e.g., 10 for MNIST, 100 for CIFAR-100, etc.)
  --gan_unrolled_steps GAN_UNROLLED_STEPS
            Number of unrolled steps for Unrolled GANs
  --gan_weight_clip GAN_WEIGHT_CLIP
            Weight clipping value for WGANs
  --gan_gradient_penalty_lambda GAN_GRADIENT_PENALTY_LAMBDA
            Gradient penalty lambda for WGAN-GP
  --gan_n_critic GAN_N_CRITIC
            Number of critic updates per generator update for WGANs
  --gan_lsgan_lambda GAN_LSGAN_LAMBDA
            Lambda parameter for LSGANs
  --gan_style_dim GAN_STYLE_DIM
            Style dimension for StyleGANs
  --gan_kernel_size_style_gen GAN_KERNEL_SIZE_STYLE_GEN
            Kernel size for StyleGAN generator
  --gan_stride_style_gen GAN_STRIDE_STYLE_GEN
            Stride for StyleGAN generator
  --gan_padding_style_gen GAN_PADDING_STYLE_GEN
            Padding for StyleGAN generator
  --gan_noise_weight GAN_NOISE_WEIGHT
            Noise weight for StyleGANs
  --gan_mixing_prob GAN_MIXING_PROB
            Mixing probability for StyleGAN
  --gan_dropout GAN_DROPOUT
            Dropout rate for the GAN model
  --gan_use_batchnorm [GAN_USE_BATCHNORM]
            If set, will use batch normalization in the GAN model
  --gan_spectral_norm_on [GAN_SPECTRAL_NORM_ON]
            If set, will use spectral normalization in the GAN model
  --gan_beta1 GAN_BETA1
            Beta1 for GAN training
  --gan_beta2 GAN_BETA2
            Beta2 for GAN training
  --gan_is_ema [GAN_IS_EMA]
            If set, will use EMA for GAN sampling
  --gan_ema_decay GAN_EMA_DECAY
            EMA decay for GAN sampling

  --diffusion_config DIFFUSION_CONFIG
            Path to the diffusion configuration file (e.g. data/configs/diffusion_config.yaml). If not provided, CLI arguments will be used to create the DiffusionConfig.
  --diffusion_model_type {cnn,res_unet}
            Architecture of the diffusion model
  --diffusion_loss {mse,l1}
            Loss function for the diffusion model
  --diffusion_num_classes DIFFUSION_NUM_CLASSES
            Number of classes for class-conditional diffusion. If unset, the model is unconditional.
  --diffusion_cond_drop_prob DIFFUSION_COND_DROP_PROB
            Conditional dropout probability for diffusion training
  --diffusion_guidance_scale DIFFUSION_GUIDANCE_SCALE
            Classifier-free guidance scale for diffusion sampling
  --diffusion_image_size DIFFUSION_IMAGE_SIZE
            Input image size for the diffusion model
  --diffusion_image_channels DIFFUSION_IMAGE_CHANNELS
            Number of image channels for the diffusion model
  --diffusion_base_channels DIFFUSION_BASE_CHANNELS
            Base number of channels for the diffusion model
  --diffusion_channel_mults DIFFUSION_CHANNEL_MULTS [DIFFUSION_CHANNEL_MULTS ...]
            Channel multipliers for the diffusion model U-Net
  --diffusion_time_emb_dim DIFFUSION_TIME_EMB_DIM
            Embedding dimension for the diffusion time step
  --diffusion_time_width_coef DIFFUSION_TIME_WIDTH_COEF
            Width multiplier for the time embedding MLP
  --diffusion_use_attention [DIFFUSION_USE_ATTENTION]
            If set, will use attention blocks in the diffusion model
  --diffusion_attention_resolutions DIFFUSION_ATTENTION_RESOLUTIONS [DIFFUSION_ATTENTION_RESOLUTIONS ...]
            Spatial resolutions where attention is applied in the diffusion model
  --diffusion_num_heads DIFFUSION_NUM_HEADS
            Number of attention heads for the diffusion model
  --diffusion_dropout DIFFUSION_DROPOUT
            Dropout probability for the diffusion model
  --diffusion_kernel_size DIFFUSION_KERNEL_SIZE
            Kernel size for diffusion convolution blocks
  --diffusion_stride DIFFUSION_STRIDE
            Stride for diffusion convolution blocks
  --diffusion_padding DIFFUSION_PADDING
            Padding for diffusion convolution blocks
  --diffusion_use_batch_norm [DIFFUSION_USE_BATCH_NORM]
            If set, will use batch normalization in classic CNN diffusion models
  --diffusion_num_groups DIFFUSION_NUM_GROUPS
            Number of groups for GroupNorm layers
  --diffusion_eps_groupnorm DIFFUSION_EPS_GROUPNORM
            Epsilon value for GroupNorm layers
  --diffusion_down_kernel_size DIFFUSION_DOWN_KERNEL_SIZE
            Kernel size for downsampling blocks
  --diffusion_down_stride DIFFUSION_DOWN_STRIDE
            Stride for downsampling blocks
  --diffusion_down_padding DIFFUSION_DOWN_PADDING
            Padding for downsampling blocks
  --diffusion_down_num_res_blocks DIFFUSION_DOWN_NUM_RES_BLOCKS
            Number of residual blocks per downsampling stage
  --diffusion_up_kernel_size DIFFUSION_UP_KERNEL_SIZE
            Kernel size for upsampling blocks
  --diffusion_up_stride DIFFUSION_UP_STRIDE
            Stride for upsampling blocks
  --diffusion_up_padding DIFFUSION_UP_PADDING
            Padding for upsampling blocks
  --diffusion_up_num_res_blocks DIFFUSION_UP_NUM_RES_BLOCKS
            Number of residual blocks per upsampling stage
  --diffusion_learning_rate DIFFUSION_LEARNING_RATE
            Learning rate for diffusion model training
  --diffusion_beta1 DIFFUSION_BETA1
            Beta1 for diffusion training optimizer
  --diffusion_beta2 DIFFUSION_BETA2
            Beta2 for diffusion training optimizer
  --diffusion_use_torch_compile [DIFFUSION_USE_TORCH_COMPILE]
            If set, will compile the diffusion model with torch.compile
  --diffusion_compile_mode {default,reduce-overhead,max-autotune}
            torch.compile mode for diffusion training
  --diffusion_use_ddim [DIFFUSION_USE_DDIM]
            If set, will use DDIM sampling
  --diffusion_ddim_steps DIFFUSION_DDIM_STEPS
            Number of DDIM sampling steps
  --diffusion_use_ema [DIFFUSION_USE_EMA]
            If set, will use EMA for diffusion sampling
  --diffusion_ema_decay DIFFUSION_EMA_DECAY
            EMA decay for diffusion sampling
  --diffusion_use_latent_diffusion [DIFFUSION_USE_LATENT_DIFFUSION]
            If set, will use latent diffusion instead of pixel-space diffusion
  --diffusion_latent_dim DIFFUSION_LATENT_DIM
            Latent dimension used by latent diffusion
  --diffusion_latent_hidden_dim DIFFUSION_LATENT_HIDDEN_DIM
            Hidden dimension of the latent diffusion encoder/decoder
  --diffusion_latent_kernel_size DIFFUSION_LATENT_KERNEL_SIZE
            Kernel size for latent diffusion convolution blocks
  --diffusion_latent_stride DIFFUSION_LATENT_STRIDE
            Stride for latent diffusion convolution blocks
  --diffusion_latent_padding DIFFUSION_LATENT_PADDING
            Padding for latent diffusion convolution blocks
  --diffusion_latent_scale_factor DIFFUSION_LATENT_SCALE_FACTOR
            Scale factor used when mapping latent variables to the VAE latent space

  --inference_model_type {ddpm,stable_diffusion}
            Type of model to use for inference
  --inference_description INFERENCE_DESCRIPTION
            Description for stable diffusion model
  --inference_batch_size INFERENCE_BATCH_SIZE
            Number of images to generate in a batch

  --finetuning_experiment {baseline,custom_scheduler_and_sampling,finetune,lora}
            Type of experiment to run for fine-tuning
  --finetuning_prompts FINETUNING_PROMPTS [FINETUNING_PROMPTS ...]
            List of prompts for image generation
  --finetuning_sampler {ddpm,ddim}
            Sampler to use for inference during fine-tuning (my own implementation of ddpm or ddim)
  --finetuning_eta FINETUNING_ETA
            Eta parameter for DDIM sampler (default: 0.0, deterministic)
  --is_finetuning [IS_FINETUNING]
            If set, will run the training loop for full or LoRA fine-tuning
  --finetuning_gradient_clip FINETUNING_GRADIENT_CLIP
            Gradient clipping value
  --finetuning_lora_rank FINETUNING_LORA_RANK
            LoRA rank
  --finetuning_lora_alpha FINETUNING_LORA_ALPHA
            LoRA alpha
  --finetuning_lora_name FINETUNING_LORA_NAME
            Name of the LoRA model to save
```
To run a specific experiment, you need to figure which parameters are relevant for the experiment you want to run, and then pass them as CLI arguments. For example, to run a diffusion model training experiment, you would use the ones beginning by `diffusion_`, and to run a GAN experiment, you would use the ones beginning by `gan_`. The launcher will automatically filter out irrelevant parameters for the selected experiment. Those only accounts for the model specific ones, then you need to pass the general parameters going from `launch_mode` (in `{vae,gan,diffusion,inference,finetuning}`) to `save_model` in the previous list.

To note that a config file provide the model specific parameters, and can be passed to the launcher through the `--vae_config`, `--gan_config`, or `--diffusion_config` arguments. If a config file is provided, the CLI arguments will be ignored for the model specific parameters. Still, the general parameters will need to be given through the CLI arguments, so that the launcher properly knows what to do with the experiment (otherwise it can be launched with a default config or crash if some minimal requirements are not met).


# Outputs
Configurations can be manually set through the CLI and then saved in `data/configs/` or loaded from a configuration file

Datasets used for training and evaluation are stored by their name in the `data/` folder, and can be loaded through the `load.py` module.

Generated samples and experiment artifacts are stored under:
```text
data/output/
```

Model parameters and saved weights are organized under:
```text
data/models_parameters/
```


# What This Project Explores
The project is ultimately structured around several questions:

### From-Scratch vs Pretrained
A central distinction in the project is between understanding a generative model and using one.

| Approach             | Main objective                                  | Training cost | Control   |
| -------------------- | ----------------------------------------------- | ------------: | --------- |
| From scratch         | Understand and experiment with the architecture |          High | High      |
| Pretrained inference | Generate images using existing knowledge        |           Low | Medium    |
| Full fine-tuning     | Adapt the entire model                          |     Very high | Very high |
| LoRA fine-tuning     | Efficiently adapt a pretrained model            |         Lower | High      |

This makes the project both a learning framework and an experimentation platform.

### Generative modeling
> How do VAEs, GANs, and diffusion models learn to generate data?

### Latent representations
> How does the structure of a latent space affect generation and interpolation?

### Adversarial learning
> Why are GANs capable of producing sharp images, and why can their optimization be unstable?

### Diffusion
> How does progressively removing noise allow a model to generate complex images?

### Sampling
> How much does the sampling algorithm influence the final generated image?

### Pretrained models
> How much generative capability is already contained in a pretrained diffusion model?

### Fine-tuning
> Can a pretrained model be adapted efficiently without updating all of its parameters?

### LoRA
> How much adaptation can be achieved using only a small number of trainable parameters?