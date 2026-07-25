"""
Fine-tune google/vit-base-patch16-224 with LoRA on the 11-class weather
image dataset (asdwddd/autotrain-data-weather-classification).

This demonstrates the core PEFT workflow:
  1. Load a pretrained backbone + swap its classification head for our task.
  2. Freeze the backbone, attach small trainable LoRA adapters to the
     attention query/value projections.
  3. Train only the adapters + new head.
  4. Compare trainable-parameter count against full fine-tuning.
  5. Evaluate on a held-out split.
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "google/vit-base-patch16-224"
DATA_DIR = "./weather_raw/raw/image_folders/auto/dataset"
OUTPUT_DIR = "./vit-weather-lora"

# Keep the demo fast on CPU: cap examples per class.
MAX_PER_CLASS_TRAIN = 40
MAX_PER_CLASS_EVAL = 10


def main():
    print(f"Loading dataset from: {DATA_DIR}")
    full_ds = load_dataset("imagefolder", data_dir=DATA_DIR)["train"]
    split_ds = full_ds.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    train_ds, eval_ds = split_ds["train"], split_ds["test"]

    label_col = "label"
    labels = train_ds.features[label_col].names
    print(f"Classes ({len(labels)}): {labels}")

    def subsample(dataset, per_class):
        idx_by_label = {}
        for i, lab in enumerate(dataset[label_col]):
            idx_by_label.setdefault(lab, []).append(i)
        keep = []
        for lab, idxs in idx_by_label.items():
            keep.extend(idxs[:per_class])
        return dataset.select(keep)

    train_ds = subsample(train_ds, MAX_PER_CLASS_TRAIN)
    eval_ds = subsample(eval_ds, MAX_PER_CLASS_EVAL)
    print(f"Using {len(train_ds)} train / {len(eval_ds)} eval images")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    def transform(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = batch[label_col]
        return inputs

    train_ds.set_transform(transform)
    eval_ds.set_transform(transform)

    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    base_model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # replaces the 1000-class head with an 11-class one
    )

    full_params = sum(p.numel() for p in base_model.parameters())

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],  # attention projections in ViT
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],  # new head must stay fully trainable
    )
    model = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {full_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/full_params:.2f}%)\n")

    def collate_fn(batch):
        pixel_values = torch.stack([torch.tensor(b["pixel_values"][0]) for b in batch])
        labels_t = torch.tensor([b["labels"] for b in batch])
        return {"pixel_values": pixel_values, "labels": labels_t}

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels_arr).mean()
        return {"accuracy": acc}

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-4,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    print("Starting training...\n")
    trainer.train()

    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(metrics)

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
