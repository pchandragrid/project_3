import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from dataset_advanced import COCODataset
from torch.utils.data import random_split
from tqdm import tqdm
from PIL import Image
from pycocoevalcap.cider.cider import Cider


def generate_caption(model, processor, image, device,
                     num_beams=5,
                     max_length=20,
                     length_penalty=1.0):

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=max_length,
            length_penalty=length_penalty
        )

    caption = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
    return caption


def evaluate_config(model, processor, val_dataset, device,
                    num_beams, max_length, length_penalty,
                    max_samples=200):

    model.eval()
    cider_scorer = Cider()

    ground_truth = {}
    predictions = {}

    print(f"\nTesting: beams={num_beams}, "
          f"max_len={max_length}, "
          f"len_penalty={length_penalty}")

    for idx in tqdm(range(min(max_samples, len(val_dataset)))):
        real_idx = val_dataset.indices[idx]
        ann = val_dataset.dataset.annotations[real_idx]

        image_path = os.path.join("train2017", ann["image"])
        image = Image.open(image_path).convert("RGB")

        pred_caption = generate_caption(
            model,
            processor,
            image,
            device,
            num_beams=num_beams,
            max_length=max_length,
            length_penalty=length_penalty
        )

        ground_truth[idx] = ann["captions"]
        predictions[idx] = [pred_caption]

    score, _ = cider_scorer.compute_score(ground_truth, predictions)

    print(f"CIDEr: {score:.4f}")

    model.train()
    return score


def main():

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available.")

    device = torch.device("mps")
    print("Using device:", device)

    # Load best Phase 2 model
    model_dir = "saved_model_phase2"

    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir)

    model.to(device)

    # Load validation split
    full_dataset = COCODataset(
        "annotations/subset_10k.jsonl",
        "train2017",
        processor
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    # =========================
    # Experiment Grid
    # =========================

    beam_sizes = [5]
    max_lengths = [20]
    length_penalties = [1.0]

    results = []

    for beams in beam_sizes:
        for max_len in max_lengths:
            for lp in length_penalties:

                score = evaluate_config(
                    model,
                    processor,
                    val_dataset,
                    device,
                    num_beams=beams,
                    max_length=max_len,
                    length_penalty=lp
                )

                results.append((beams, max_len, lp, score))

    print("\n===== FINAL RESULTS =====")
    for r in results:
        print(f"Beams={r[0]}, MaxLen={r[1]}, "
              f"LenPenalty={r[2]} -> CIDEr={r[3]:.4f}")


if __name__ == "__main__":
    main()