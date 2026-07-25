"""
Pull a pretrained image-classification model from Hugging Face and evaluate
it on a sample image with a known ground-truth label.
"""

import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

MODEL_NAME = "google/vit-base-patch16-224"

# Known test image + its ground-truth ImageNet label (for evaluation)
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"  # two cats on a couch
GROUND_TRUTH_KEYWORDS = ["cat", "tabby", "tiger cat", "Egyptian cat"]

def load_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    from io import BytesIO
    return Image.open(BytesIO(resp.content)).convert("RGB")

def main():
    print(f"Loading model + processor: {MODEL_NAME}")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Downloading image: {IMAGE_URL}")
    image = load_image(IMAGE_URL)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top5 = torch.topk(probs, 5)

    print("\nTop-5 predictions:")
    predicted_labels = []
    for score, idx in zip(top5.values, top5.indices):
        label = model.config.id2label[idx.item()]
        predicted_labels.append(label)
        print(f"  {label:30s} {score.item()*100:5.2f}%")

    # --- Evaluation ---
    top1_label = predicted_labels[0]
    correct = any(kw.lower() in top1_label.lower() for kw in GROUND_TRUTH_KEYWORDS)
    print(f"\nGround truth keywords: {GROUND_TRUTH_KEYWORDS}")
    print(f"Top-1 prediction: '{top1_label}'")
    print(f"Evaluation: {'CORRECT' if correct else 'INCORRECT'}")

if __name__ == "__main__":
    main()
