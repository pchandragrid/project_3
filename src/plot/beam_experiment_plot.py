import matplotlib.pyplot as plt
from pathlib import Path

# Beam sizes tested
beam_sizes = [1, 3, 5, 10]

# Example CIDEr scores from experiments
blip_scores = [0.52, 0.59, 0.61, 0.60]
vit_scores = [0.50, 0.56, 0.60, 0.58]
git_scores = [0.12, 0.16, 0.17, 0.16]

plt.figure(figsize=(8,5))

plt.plot(beam_sizes, blip_scores, marker='o', label="BLIP")
plt.plot(beam_sizes, vit_scores, marker='o', label="ViT-GPT2")
plt.plot(beam_sizes, git_scores, marker='o', label="GIT")

plt.xlabel("Beam Size")
plt.ylabel("CIDEr Score")
plt.title("Effect of Beam Size on Caption Quality")

plt.legend()

plt.grid(True)

out_path = Path(__file__).resolve().parent / "beam_search_experiment.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

plt.close()