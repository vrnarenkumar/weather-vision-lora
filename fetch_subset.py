"""Download only a small per-class subset of the weather dataset (fast)."""

import os
from huggingface_hub import HfFileSystem, hf_hub_download

REPO_ID = "asdwddd/autotrain-data-weather-classification"
BASE = "raw/image_folders/auto/dataset"
PER_CLASS = 60  # 40 train + 10 eval + a few spare
LOCAL_DIR = "./weather_raw"

fs = HfFileSystem()

classes = sorted(
    p.split("/")[-1]
    for p in fs.ls(f"datasets/{REPO_ID}/{BASE}", detail=False)
)
print(f"Classes: {classes}")

for cls in classes:
    files = fs.ls(f"datasets/{REPO_ID}/{BASE}/{cls}", detail=False)
    files = sorted(files)[:PER_CLASS]
    print(f"{cls}: downloading {len(files)} images")
    for f in files:
        rel_path = f.split(f"datasets/{REPO_ID}/")[-1]
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=rel_path,
            local_dir=LOCAL_DIR,
        )

print("Done.")
