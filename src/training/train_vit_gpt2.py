import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    ViTModel
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.data.coco_vit_gpt2_dataset import COCODatasetViTGPT2
from tqdm import tqdm
from pycocoevalcap.cider.cider import Cider
from PIL import Image


# ==========================================
# GENERATE CAPTION
# ==========================================
def generate_caption(model, processor, tokenizer, image, device):

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            num_beams=5,
            max_length=20,
            length_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ==========================================
# CIDEr EVALUATION
# ==========================================
def evaluate_cider(model, processor, tokenizer, val_dataset, device, max_samples=200):

    model.eval()
    cider_scorer = Cider()

    ground_truth = {}
    predictions = {}

    for idx in tqdm(range(min(max_samples, len(val_dataset))), desc="CIDEr Eval"):

        real_idx = val_dataset.indices[idx]
        ann = val_dataset.dataset.annotations[real_idx]

        image_path = os.path.join("train2017", ann["image"])
        image = Image.open(image_path).convert("RGB")

        pred_caption = generate_caption(model, processor, tokenizer, image, device)

        ground_truth[idx] = ann["captions"]
        predictions[idx] = [pred_caption]

    score, _ = cider_scorer.compute_score(ground_truth, predictions)

    print(f"CIDEr Score: {score:.4f}")

    model.train()
    return score


# ==========================================
# MAIN
# ==========================================
def main():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    EPOCHS = 5
    BATCH_SIZE = 6
    LR = 3e-5
    SAVE_DIR = "saved_vit_gpt2"

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------
    # Build Encoder + Decoder
    # ------------------------------------------

    encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    decoder_config = GPT2Config.from_pretrained("gpt2")
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=decoder_config)

    model = VisionEncoderDecoderModel(
        encoder=encoder,
        decoder=decoder
    )

    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.to(device)

    # ------------------------------------------
    # DATASET
    # ------------------------------------------

    dataset = COCODatasetViTGPT2(
        "annotations/subset_10k.jsonl",
        "train2017",
        processor,
        tokenizer,
        mode="short"
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_cider = 0

    # ==========================================
    # TRAIN LOOP
    # ==========================================
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

        # ------------------------------------------
        # CIDEr Evaluation
        # ------------------------------------------
        cider_score = evaluate_cider(
            model,
            processor,
            tokenizer,
            val_dataset,
            device
        )

        # Save best model
        if cider_score > best_cider:
            best_cider = cider_score
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            processor.save_pretrained(SAVE_DIR)
            print("Best model saved.")

        scheduler.step()

    print("ViT-GPT2 Training complete.")


if __name__ == "__main__":
    main()
