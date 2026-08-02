# NRT – Inference
> This test suite validates the complete inference workflow for the pre-trained diffusion models available in the project. Its objective is not to benchmark image quality or inference speed, but to ensure that the inference pipeline can be correctly initialized, executed, saved, and reloaded across the supported models and command-line configurations.
>
> The tests deliberately use a small number of inference steps and images to keep execution time reasonable while still exercising the complete inference path.

# Directory Structure

```text
NRT_inference/
├── outputs/
│   ├── ddpm_output/
│   │   └── *.png
│   └── futuristic_city/
│       └── *.png
│
└── test.py
```
Model parameters are downloaded to `./data/models_parameters/model_name`for the root of the project (or stored then used from there to avoid duplicates). The inference tests save generated images and serialized pipelines to `NRT/NRT_inference/outputs/`.

# What is validated?
Each test executes the inference entry point through the same command-line interface used by the project.

The test suite validates:
* Pre-trained pipeline loading
* Command-line argument handling
* Reproducible inference with a fixed seed
* Number of inference steps
* Batch image generation
* Model-specific inference arguments
* Image saving
* Pipeline saving with `save_pretrained`
* Pipeline loading from a previously saved directory
* Stable Diffusion architecture inspection
* End-to-end execution of `PretrainedInference`

Two inference workflows are currently covered:
* **DDPM**
* **Stable Diffusion**

The tests are intentionally lightweight and are primarily intended to detect regressions in the inference workflow rather than assess the quality of the generated images.


# DDPM Inference
```py
test_inference_ddpm()
```

### Configuration
The DDPM inference test runs the project inference script through a subprocess using the following configuration:
```text
model_type           = ddpm
seed                 = 42
device               = auto
num_inference_steps  = 20
batch_size           = 3
save_name            = ddpm_output
save_model           = True
show_architecture    = False
```

The corresponding command is:
```bash
python src/pretrained/inference.py \
    --is_nrt=True \
    --seed=42 \
    --device=auto \
    --model_type=ddpm \
    --num_inference_steps=20 \
    --save_name=ddpm_output \
    --batch_size=3 \
    --save_model=True \
    --show_architecture=False
```

### Pipeline
If no previously saved model is provided, the inference class loads the pre-trained DDPM pipeline:
```text
google/ddpm-cifar10-32
```

The pipeline is moved to the selected device before inference.
The inference call uses:
```py
{
    "num_inference_steps": 20,
    "generator": seeded_generator,
    "batch_size": 3
}
```
A fixed seed (`42`) is used so that the NRT execution remains reproducible.

### What is checked?
The test exercises the complete path:
```text
CLI arguments
      ↓
PretrainedInference
      ↓
DDPMPipeline loading
      ↓
device placement
      ↓
seeded inference
      ↓
3 generated images
      ↓
image saving
      ↓
pipeline saving
```
The test therefore verifies that the DDPM pipeline can be loaded and used through the project's actual inference entry point rather than only through isolated model calls.

### Saving
With `save_model=True`, the pipeline is serialized using:
```py
self.pipe.save_pretrained(self.model_save_path)
```
This is important for the NRT because it also exercises the project's model persistence path.

### Output
The test saves the generated images to:
```text
NRT/NRT_inference/data/ddpm_output/
```
<table>
  <tr>
    <td style="text-align:center;">
      <img src="./data/ddpm_output/image_0.png" width="200"/>
      <br><b>Generated image 0</b>
    </td>
    <td style="text-align:center;">
      <img src="./data/ddpm_output/image_1.png" width="200"/>
      <br><b>Generated image 1</b>
    </td>
    <td style="text-align:center;">
      <img src="./data/ddpm_output/image_2.png" width="200"/>
      <br><b>Generated image 2</b>
    </td>
  </tr>
<table>
The pictures remain still blurry and it's difficult to distinguish anything. We can see that the model is not trained enough to generate good images, relatively small. The purpose of this test is to validate the inference workflow, not to produce high-quality images.


# Stable Diffusion Inference
```py
test_inference_stable_diffusion()
```

### Configuration
The Stable Diffusion test uses:
```text
model_type           = stable_diffusion
seed                 = 42
device               = auto
num_inference_steps  = 20
description          = "a futuristic city at night"
save_name            = futuristic_city
batch_size           = 3
guidance_scale       = 7.5
height               = 512
width                = 512
save_model           = True
show_architecture    = True
```
The corresponding command is:
```bash
python src/pretrained/inference.py \
    --is_nrt=True \
    --seed=42 \
    --device=auto \
    --model_type=stable_diffusion \
    --num_inference_steps=20 \
    --description="a futuristic city at night" \
    --save_name=futuristic_city \
    --batch_size=3 \
    --guidance_scale=7.5 \
    --height=512 \
    --width=512 \
    --save_model=True \
    --show_architecture=True
```

### Pipeline
When no previously saved pipeline is supplied, the test loads:
```text
segmind/tiny-sd
```
The selected tensor precision depends on the device:
```py
torch.float16 if self.device == "cuda" else torch.float32
```
This allows the same inference workflow to be exercised on both GPU and CPU configurations.

### Inference configuration
The Stable Diffusion pipeline receives:
```py
{
    "prompt": "a futuristic city at night",
    "num_inference_steps": 20,
    "generator": seeded_generator,
    "guidance_scale": 7.5,
    "height": 512,
    "width": 512,
    "num_images_per_prompt": 3
}
```
The test therefore validates not only the generic inference path but also the arguments specific to text-to-image generation.

### Architecture inspection
Unlike the DDPM test, `show_architecture=True`.
The inference class retrieves the main Stable Diffusion components:
```text
VAE
UNet
Scheduler
Tokenizer
Text Encoder
```
and prints their architectures before generation.

This provides an additional regression check that the expected pipeline components are available and accessible through the loaded pipeline. It's particularly useful to compare to my own model implementation.

The architecture retrieval is implemented through:
```py
get_architecture()
```
which collects the components directly from the loaded Stable Diffusion pipeline.

### What is checked?
The complete workflow is:
```text
CLI arguments
      ↓
PretrainedInference
      ↓
StableDiffusionPipeline loading
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
This makes the Stable Diffusion test particularly useful for detecting changes in the pipeline API or in the expected model components.

### Output
The test saves the generated images to:
```text
NRT/NRT_inference/data/futuristic_city/
```
<table>
  <tr>
    <td style="text-align:center;">
      <img src="./data/futuristic_city/image_0.png" width="200"/>
      <br><b>Generated image 0</b>
    </td>
    <td style="text-align:center;">
      <img src="./data/futuristic_city/image_1.png" width="200"/>
      <br><b>Generated image 1</b>
    </td>
    <td style="text-align:center;">
      <img src="./data/futuristic_city/image_2.png" width="200"/>
      <br><b>Generated image 2</b>
    </td>
  </tr>
<table>

The generated samples show that the `tiny-sd` pipeline successfully follows the provided prompt, producing futuristic city scenes with substantial visual detail at 512×512 resolution. The three samples also show some diversity while consistently preserving the main semantic elements of the prompt.

Despite using only 20 inference steps, the generated images remain detailed and coherent. 
The higher resolution and Stable Diffusion pipeline make this test noticeably more computationally expensive than the lightweight DDPM inference test for a result infinitely better than the time wasted, but the resulting outputs provide a useful qualitative confirmation that the complete text-to-image workflow is functioning correctly.

The results therefore confirm several aspects of the inference pipeline at once: the prompt is correctly passed to the text-conditioning components, the requested image dimensions are respected, multiple images can be generated in a single batch, and the resulting outputs can be successfully saved to disk.


# Inference Class
The tests exercise the `PretrainedInference` class rather than duplicating inference logic inside the NRT.
| Method                       | Main validation                       |
| ---------------------------- | ------------------------------------- |
| `load_pipeline()`            | Loading a previously saved pipeline   |
| `load_pretrained_pipeline()` | Loading the default pre-trained model |
| `save_pipeline()`            | Pipeline serialization                |
| `get_architecture()`         | Access to Stable Diffusion components |
| `print_architecture()`       | Architecture inspection               |
| `run_inference()`            | Model-specific inference arguments    |
| `save_images()`              | Batch and single-image output saving  |
| `run()`                      | Complete inference workflow           |

This is important because the NRT therefore validates the **same code path used by users**, including argument parsing and filesystem operations.

# Reproducibility
Both tests use:
```py
seed = 42
```
and create a device-specific PyTorch generator:
```py
torch.Generator(self.device).manual_seed(self.args.seed)
```
The fixed seed is intended to make the generated outputs reproducible and easier to compare when investigating regressions.

The tests do not attempt to guarantee identical results across every hardware/backend configuration. Their primary purpose is to ensure that the seeded inference workflow remains functional and produces valid outputs.

# Command-Line Validation
The NRT tests intentionally launch:
```text
src/pretrained/inference.py
```
through `subprocess.run()` rather than directly calling the inference class.

This validates an additional layer of the application:
```text
NRT
 ↓
command-line arguments
 ↓
argument parsing
 ↓
inference.py
 ↓
PretrainedInference
 ↓
pre-trained pipeline
```
As a result, regressions in either the inference implementation **or the command-line interface** can be detected.

# Summary
| Test | Model | Main validation |
| ---- | ------| --------------- |
| `test_inference_ddpm()` | DDPM  | Pipeline loading, seeded generation, batch inference, image saving and pipeline serialization |
| `test_inference_stable_diffusion()` | Stable Diffusion | Text-to-image generation, guidance, architecture inspection, batch inference, image saving and pipeline serialization |

### Components covered
| Component                   | Validation                                       |
| --------------------------- | ------------------------------------------------ |
| **CLI**                     | Argument parsing and inference entry point       |
| **DDPMPipeline**            | Pre-trained loading and image generation         |
| **StableDiffusionPipeline** | Pre-trained loading and text-to-image generation |
| **Seed**                    | Reproducible inference configuration             |
| **Scheduler**               | Inference through the loaded pipeline            |
| **VAE**                     | Stable Diffusion pipeline component availability |
| **UNet**                    | Stable Diffusion pipeline component availability |
| **Tokenizer**               | Text-conditioning pipeline availability          |
| **Text Encoder**            | Prompt-conditioning pipeline availability        |
| **Image saving**            | Single and batch output handling                 |
| **Pipeline saving**         | `save_pretrained()` workflow                     |
| **Pipeline loading**        | Reloading saved pipelines                        |

> **Note:** These experiments are designed as **non-regression tests, not performance or image-quality benchmarks**. The number of inference steps and generated images is deliberately kept low to limit execution time. Generated images are useful for verifying that inference still produces valid outputs, but their visual quality should not be interpreted as a benchmark of the underlying pre-trained models.
>
> The Stable Diffusion test additionally validates the project's interaction with the complete diffusion pipeline, including the VAE, UNet, scheduler, tokenizer, and text encoder. This makes it useful for detecting regressions caused by changes to model loading, pipeline configuration, or dependency/API updates.
>
> The use of the actual command-line entry point also ensures that the NRT covers the complete user-facing inference workflow rather than only testing individual Python methods.
