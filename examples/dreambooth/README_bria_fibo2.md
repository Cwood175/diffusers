# DreamBooth LoRA training for fibo-2 (Bria)

[fibo-2](https://huggingface.co/briaai) is Bria's flow-matching text-to-image and image-editing model
(`BriaFibo2Pipeline`): a `BriaFibo2Transformer2DModel` denoiser conditioned on a Qwen3-VL text encoder, with an
`AutoencoderKLFlux2` VAE. This directory has two LoRA trainers:

- `train_dreambooth_lora_bria_fibo2.py` — text-to-image.
- `train_dreambooth_lora_bria_fibo2_img2img.py` — image editing (image-to-image).

Both train a LoRA adapter on the transformer only; the text encoder and VAE are frozen.

> [!NOTE]
> fibo-2 is trained on **structured JSON captions** and renders them best, but a plain-text short description is
> accepted too (and used as-is). Keep your training captions in the form you intend to prompt with. The trainers
> clean captions the way the pipeline does, so a JSON caption and its cleaned, compact form train the same.

## Running locally

### Installing the dependencies

Install `diffusers` from source and then the trainer requirements:

```bash
pip install -e .
cd examples/dreambooth
pip install -r requirements_bria_fibo2.txt
```

And initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment:

```bash
accelerate config default
```

fibo-2's Qwen3-VL text encoder needs `transformers>=5.20.1`.

## Text-to-image

Train a LoRA on a folder of instance images (one trigger prompt for all of them):

```bash
export MODEL_NAME="briaai/fibo-2"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="trained-fibo2-lora"

accelerate launch train_dreambooth_lora_bria_fibo2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of sks dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --seed=0 \
  --push_to_hub
```

To train on a dataset with per-image captions, pass `--dataset_name` and `--caption_column` (the captions can be
structured JSON captions or plain short descriptions) instead of `--instance_data_dir`:

```bash
accelerate launch train_dreambooth_lora_bria_fibo2.py \
  --pretrained_model_name_or_path="briaai/fibo-2" \
  --dataset_name="your-dataset" \
  --caption_column="text" \
  --instance_prompt="in the style of TOK" \
  --output_dir="trained-fibo2-lora" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=1000
```

### Inference

```python
from diffusers import BriaFibo2Pipeline
import torch

pipeline = BriaFibo2Pipeline.from_pretrained("briaai/fibo-2", torch_dtype=torch.bfloat16).to("cuda")
pipeline.load_lora_weights("trained-fibo2-lora", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("A photo of sks dog in a bucket").images[0]
image.save("fibo2.png")
```

## Image editing (image-to-image)

The editing trainer learns from **(conditioning image, target image, edit instruction)** triplets, so it needs a
dataset (`--dataset_name`) with a source-image column, a target-image column and a caption column holding the
instruction. The conditioning and target images are given the same geometry so each pair stays aligned; the
conditioning image is both shown to the Qwen3-VL text encoder (as the image to edit) and VAE-encoded into the
transformer's clean context tokens, exactly as `BriaFibo2Pipeline` does at inference.

```bash
accelerate launch train_dreambooth_lora_bria_fibo2_img2img.py \
  --pretrained_model_name_or_path="briaai/fibo-2" \
  --dataset_name="your-edit-dataset" \
  --image_column="target_image" \
  --cond_image_column="source_image" \
  --caption_column="instruction" \
  --instance_prompt="make it a watercolor painting" \
  --output_dir="trained-fibo2-edit-lora" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=2000 \
  --validation_prompt="make it a watercolor painting" \
  --validation_image="path/to/source.png" \
  --validation_epochs=25
```

`--instance_prompt` is the fallback instruction used when a pair has no caption. `--caption_column` holds the edit
instruction as plain text.

> [!NOTE]
> For editing, the conditioning image is shown to Qwen3-VL, and its vision tokens count towards
> `--max_sequence_length` (a 768px image is up to ~576 tokens). The editing trainer defaults it to 1024 for that
> reason; raise it further only if you pair large images with long structured captions.

### Inference

```python
from diffusers import BriaFibo2Pipeline
from diffusers.utils import load_image
import torch

pipeline = BriaFibo2Pipeline.from_pretrained("briaai/fibo-2", torch_dtype=torch.bfloat16).to("cuda")
pipeline.load_lora_weights("trained-fibo2-edit-lora", weight_name="pytorch_lora_weights.safetensors")
image = load_image("path/to/source.png")
edited = pipeline("make it a watercolor painting", image=image).images[0]
edited.save("fibo2_edit.png")
```

## Notes

- **LoRA layers.** By default the adapter targets attention (`to_q`, `to_k`, `to_v`, `to_out.0`) and the SwiGLU
  feed-forward projections (`w1`, `w2`, `w3`) across every block. Pass `--lora_layers` with a comma-separated list
  to target a different set (e.g. `--lora_layers="to_q,to_k,to_v,to_out.0"` for attention only).
- **Aspect-ratio bucketing.** Pass `--use_aspect_ratio_buckets` to bucket images by aspect ratio (optionally with an
  explicit `--aspect_ratio_buckets` list), instead of resizing everything to a square `--resolution`.
- **Caption dropout.** `--caption_dropout 0.1` trains a fraction of samples unconditionally, which improves
  classifier-free guidance. For editing, the conditioning image is always kept; only the instruction is dropped.
- **Memory.** Use `--cache_latents` to pre-encode images, `--gradient_checkpointing`, `--use_8bit_adam`, and
  `--offload` to move the VAE and text encoder to CPU when idle. `--bnb_quantization_config_path` enables
  bitsandbytes-quantized training of the transformer, and `--do_fp8_training` enables FP8 (TorchAO) training.
- **Prodigy.** Pass `--optimizer="prodigy"` (with `--learning_rate=1.0`) to use the Prodigy optimizer.

> [!NOTE]
> These scripts are provided as a starting point; tune hyperparameters (rank, learning rate, steps) for your data.
