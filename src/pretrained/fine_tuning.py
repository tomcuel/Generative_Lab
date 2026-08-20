# =======================================
# Library Imports
# =======================================
from diffusers import StableDiffusionPipeline
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path
from peft import LoraConfig
import sys 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
import torch
import torch.nn.functional as F
from typing import Optional


# =======================================
# Important Path Setup
# =======================================
from src.models.diffusion_models import NoiseScheduler


# =======================================
# Tiny-SD Model Wrapper for Fine-tuning and Inference
# =======================================
class PretrainedFineTuning:
    def __init__(self, 
        model_path: str, 
        device: str = "cpu",
        custom_scheduler: Optional[NoiseScheduler] = None,
        class_names: Optional[list[str]] = None
    ):
        self.device = device

        # Load Tiny-SD model
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32, safety_checker=None)
        self.pipe = self.pipe.to(device)

        # Components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.latent_scaling_factor = self.vae.config.scaling_factor

        # Scheduler
        self.custom_scheduler = custom_scheduler

        # Optional label -> prompt mapping for class-conditional fine-tuning
        self.class_names = class_names

    # =============================
    # Freeze components
    # =============================
    def freeze_vae(self):
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def freeze_text_encoder(self):
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def freeze_unet(self):
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

    # =============================
    # Encode image
    # =============================
    @torch.no_grad()
    def encode_image(self, images):
        posterior = self.vae.encode(images).latent_dist
        latents = posterior.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    # =============================
    # Decode image
    # =============================
    @torch.no_grad()
    def decode_image(self, latents):
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        # return images.clamp(-1, 1)
        # images = images / 2 + 0.5
        # return images.clamp(0, 1)
        images = (images.clamp(-1, 1) + 1) / 2
        return images

    # =============================
    # Text embedding
    # =============================
    @torch.no_grad()
    def encode_prompt(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        tokens = self.tokenizer(prompts, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        text_embeddings = self.text_encoder(input_ids)[0]
        return text_embeddings

    @torch.no_grad()
    def encode_labels(self, labels, template: str = "a photo of a {}"): # for cifar10
        if self.class_names is None:
            raise ValueError("class_names must be provided to convert labels into prompts.")

        prompts = [template.format(self.class_names[int(label)]) for label in labels]
        return self.encode_prompt(prompts)

    # =============================
    # Training timestep
    # =============================
    def add_noise_custom(self, latents, noise, t):
        return self.custom_scheduler.q_sample(latents, t, noise)

    # =============================
    # U-Net prediction
    # =============================
    def predict_noise(self, noisy_latents, timestep, text_embeddings):
        output = self.unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings)
        return output.sample

    # =============================
    # LoRA Fine-tuning
    # =============================
    def enable_lora(self, rank: int = 4, alpha: int = 4):
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
            ],
            lora_dropout=0.0,
            bias="none",
        )
        self.unet.add_adapter(config)

        # Only LoRA parameters train
        for name, param in self.unet.named_parameters():
            param.requires_grad = ("lora_" in name)

    # =============================
    # Save and load LoRA adapter, and Fine-tuned model
    # =============================
    def save_lora_adapter(self, save_path: str, adapter_name: str = "default"):
        self.unet.save_lora_adapter(save_path, adapter_name=adapter_name)

    def load_lora_adapter(self, load_path: str, adapter_name: str = "default"):
        self.unet.load_lora_adapter(load_path, adapter_name=adapter_name)

    def save_finetuned_model(self, save_path: str):
        save_path.mkdir(parents=True, exist_ok=True)
        self.pipe.save_pretrained(save_path)

    def load_finetuned_model(self, load_path: str):
        self.pipe = StableDiffusionPipeline.from_pretrained(load_path, torch_dtype=torch.float32, safety_checker=None)
        self.pipe = self.pipe.to(self.device)
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

    # =============================
    # Trainable parameters
    # =============================
    def trainable_parameters(self):
        return [p for p in self.unet.parameters() if p.requires_grad]
    
    # =============================
    # Training
    # =============================
    def train_step(self, images, optimizer, gradient_clip_value: float = 1.0, prompts=None, labels=None):
        
        # VAE : Encode image -> latent
        with torch.no_grad():
            latents = self.encode_image(images)
            if prompts is not None:
                text_embeddings = self.encode_prompt(prompts)
            elif labels is not None:
                text_embeddings = self.encode_labels(labels)
            else:
                text_embeddings = self.encode_prompt([""] * latents.shape[0])

        if optimizer is None:
            raise ValueError("optimizer must be provided.")

        # Timestep
        T = self.custom_scheduler.timesteps
        batch_size = latents.shape[0]
        t = torch.randint(0, T, (batch_size,), device=self.device,).long()

        # Noise
        noise = torch.randn_like(latents)

        # Forward diffusion
        noisy_latents = self.add_noise_custom(latents, noise, t)

        # UNet
        noise_pred = self.predict_noise(noisy_latents, t, text_embeddings)

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        # Backprop / Optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), gradient_clip_value)
        optimizer.step()
        return loss.item()

    # =============================
    # Sampling using Tiny-SD scheduler (as done in inference.py)
    # =============================
    @torch.no_grad()
    def sample_sd(self, prompt: str, batch_size: int, height: int = 256, width: int = 256, steps: int = 30, guidance_scale: float = 7.5, seed: Optional[int] = None):
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(prompt=prompt, num_images_per_prompt=batch_size, height=height, width=width, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator)
        return result.images

    # =============================
    # Sampling using my own scheduler
    # =============================
    @torch.no_grad()
    def ddpm_step(self, x_t, t, eps):
        beta_t = self.custom_scheduler.betas[t]
        alpha_t = self.custom_scheduler.alphas[t]
        alpha_bar_t = self.custom_scheduler.alpha_bar[t]
        sqrt_alpha_bar_t = self.custom_scheduler.sqrt_alpha_bar[t]
        sqrt_one_minus_alpha_bar = self.custom_scheduler.sqrt_one_minus_alpha_bar[t]

        # predicted x0
        x0_pred = (x_t - sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1) * eps) / sqrt_alpha_bar_t.view(-1, 1, 1, 1)

        # DDPM posterior mean
        alpha_bar_prev = self.custom_scheduler.alpha_bar_prev[t]
        coef1 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mean = coef1.view(-1, 1, 1, 1) * x0_pred + coef2.view(-1, 1, 1, 1) * x_t
        if torch.all(t == 0):
            return mean

        variance = self.custom_scheduler.posterior_variance[t]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance).view(-1, 1, 1, 1) * noise

    @torch.no_grad()
    def ddim_step(self, x_t, t, t_prev, eps, eta: float = 0.0): # lots of clamping to avoid NaNs and errors
        # alpha_bar(t)
        alpha_bar_t = self.custom_scheduler.alpha_bar[t].to(self.device)
        alpha_bar_prev = torch.where(t_prev >= 0, self.custom_scheduler.alpha_bar[torch.clamp(t_prev, min=0)].to(self.device), torch.ones_like(alpha_bar_t))

        # Make them [B, 1, 1, 1]
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(-1, 1, 1, 1)

        sqrt_alpha_bar_t = torch.sqrt(torch.clamp(alpha_bar_t, min=1e-8))
        sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0))

        # Predict x_0 = (xt - sqrt(1-a_bar_t)*eps) / sqrt(a_bar_t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

        # DDIM sigma_t = eta * sqrt((1-a_prev)/(1-a_t)) * sqrt(1-a_t/a_prev)
        sigma = eta * torch.sqrt(torch.clamp((1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar_t, min=1e-8), min=0.0)) * torch.sqrt(torch.clamp(1.0 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-8), min=0.0))

        # Direction pointing to x_t
        direction = torch.sqrt( torch.clamp(1.0 - alpha_bar_prev - sigma.square(), min=0.0)) * eps

        # x_{t-1}
        x_prev = torch.sqrt(torch.clamp(alpha_bar_prev, min=0.0)) * x0_pred + direction

        # Optional stochasticity
        if eta > 0.0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

    @torch.no_grad()
    def sample_custom(self, prompts, method="ddim", height=256, width=256, steps=50, guidance_scale=7.5, seed=None, eta=0.0):
        if self.custom_scheduler is None:
            raise RuntimeError("Custom scheduler is not initialized.")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        batch_size = len(prompts)

        # Text conditioning
        cond_embeddings = self.encode_prompt(prompts)
    
        # Empty prompt for classifier-free guidance
        if guidance_scale > 1:
            uncond_prompts = ["" for _ in prompts]
            uncond_embeddings = self.encode_prompt(uncond_prompts)

        # Latent dimensions
        latent_channels = self.unet.config.in_channels
        latent_h = height // 8
        latent_w = width // 8
        latents = torch.randn(batch_size, latent_channels, latent_h, latent_w, device=self.device, generator=generator)

        # Timesteps (reverse order for sampling)
        timesteps = torch.linspace(self.custom_scheduler.timesteps - 1, 0, steps, device=self.device).long()

        # Reverse diffusion
        for i, t_value in enumerate(timesteps):
            t = torch.full((batch_size,), t_value, device=self.device, dtype=torch.long)

            # CFG
            if guidance_scale > 1:
                eps_cond = self.predict_noise(latents, t, cond_embeddings)
                eps_uncond = self.predict_noise(latents, t, uncond_embeddings)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.predict_noise(latents, t, cond_embeddings)

            # DDIM / DDPM
            if method == "ddim":
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                t = torch.full((batch_size,), int(t_value), device=self.device, dtype=torch.long)
                t_prev = torch.full( (batch_size,), int(t_prev), device=self.device, dtype=torch.long)
                latents = self.ddim_step(latents, t, t_prev, eps, eta=eta)
            elif method == "ddpm":
                latents = self.ddpm_step(latents, t, eps)
            else:
                raise ValueError(f"Unknown sampling method: {method}")

        # VAE decode
        images = self.decode_image(latents)
        return images


# === FILE: NRT/NRT_fine_tuning/test.py ===