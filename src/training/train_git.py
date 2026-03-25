import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import GitProcessor, GitForCausalLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.data.coco_git_dataset import COCODatasetGIT
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
from PIL import Image


def generate_caption(model, processor, image, device):

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=5,
            max_length=20
        )

    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def evaluate_cider(model, processor, val_dataset, device, max_samples=200):

    model.eval()
    cider_scorer = Cider()

    ground_truth = {}
    predictions = {}

    for idx in tqdm(range(min(max_samples, len(val_dataset))), desc="CIDEr Eval"):

        real_idx = val_dataset.indices[idx]
        ann = val_dataset.dataset.annotations[real_idx]

        image_path = os.path.join("train2017", ann["image"])
        image = Image.open(image_path).convert("RGB")

        pred_caption = generate_caption(model, processor, image, device)

        ground_truth[idx] = ann["captions"]
        predictions[idx] = [pred_caption]

    score, _ = cider_scorer.compute_score(ground_truth, predictions)

    print(f"CIDEr Score: {score:.4f}")
    model.train()
    return score


def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    EPOCHS = 20
    BATCH_SIZE = 4
    LR = 5e-5
    SAVE_DIR = "saved_git_model"

    os.makedirs(SAVE_DIR, exist_ok=True)

    processor = GitProcessor.from_pretrained("microsoft/git-base")
    model = GitForCausalLM.from_pretrained("microsoft/git-base")

    model.to(device)

    dataset = COCODatasetGIT(
        "annotations/subset_20k.jsonl",
        "train2017",
        processor,
        mode="mixed"
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_cider = 0

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Train Loss: {total_loss / len(train_loader):.4f}")

        cider_score = evaluate_cider(model, processor, val_dataset, device)

        if cider_score > best_cider:
            best_cider = cider_score
            model.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)
            print("Best GIT model saved.")

        scheduler.step()

    print("GIT Training complete.")


if __name__ == "__main__":
    main()
