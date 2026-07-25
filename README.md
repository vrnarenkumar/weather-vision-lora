# weather-vision-lora

Fine-tunes a pretrained Vision Transformer (`google/vit-base-patch16-224`) to classify
weather/road conditions (rain, snow, fog, frost, glaze, hail, etc.) from images, using
LoRA (PEFT) instead of full fine-tuning.

- Only 303,371 of 85,807,115 parameters (0.35%) are trainable.
- 90% accuracy on held-out data after 6 epochs on an Apple M1 GPU (MPS).
- Full experiment tracking with MLflow (hyperparameters, per-epoch metrics, confusion
  matrix and sample-prediction artifacts).

`weather_lora_peft.ipynb` walks through the whole pipeline end-to-end: data exploration,
LoRA setup with a trainable-parameter comparison against full fine-tuning, training,
evaluation, and MLflow logging.

## Why weather classification

Framed as a proof-of-concept perception module: a lightweight, cheaply-trained image
classifier like this could plausibly feed a downstream system such as a TPMS (tire
pressure monitoring system) that adjusts tire pressure automatically for detected road
conditions. This repo demonstrates the model + training pipeline, not an actual
automotive integration.

## Also included

- `classify.py` — a smaller standalone demo: pulls a pretrained ViT from Hugging Face
  and evaluates it on a sample image.
- `fetch_subset.py` — downloads a small per-class subset of the weather dataset.
- `finetune_lora.py` — script form of the LoRA fine-tuning pipeline (superseded by the
  notebook, kept for reference).
