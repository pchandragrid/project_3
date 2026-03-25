import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ANNOTATION_FILE = "annotations/captions_validation.jsonl"

short = []
medium = []
long = []

try:
    with open(ANNOTATION_FILE) as f:
        for line in f:
            data = json.loads(line)

            caption = data["captions"][0]
            length = len(caption.split())

            if length <= 8:
                short.append(length)
            elif length <= 15:
                medium.append(length)
            else:
                long.append(length)

    print("Short captions:", len(short))
    print("Medium captions:", len(medium))
    print("Long captions:", len(long))
except FileNotFoundError:
    # The plot can still be generated without the dataset file.
    print(f"Note: '{ANNOTATION_FILE}' not found; skipping caption-length counts.")


# Example scores from your training logs
blip_scores = [0.71, 0.60, 0.48]
vit_scores = [0.65, 0.59, 0.42]
git_scores = [0.30, 0.18, 0.11]

labels = ["Short", "Medium", "Long"]

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(9,5))

plt.bar(x - width, blip_scores, width, label="BLIP")
plt.bar(x, vit_scores, width, label="ViT-GPT2")
plt.bar(x + width, git_scores, width, label="GIT")

plt.xlabel("Caption Length")
plt.ylabel("CIDEr Score")
plt.title("Model Performance vs Caption Length")

plt.xticks(x, labels)

plt.legend()

out_path = Path(__file__).resolve().parent / "caption_length_analysis.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

plt.close()