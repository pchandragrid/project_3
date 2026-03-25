import argparse
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

try:
    from tasks.task1_blip_optimization.data import CocoCaptionDataset
except ModuleNotFoundError:
    from data import CocoCaptionDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_set_processor_resolution(processor: BlipProcessor, image_size: int) -> None:
    image_processor = getattr(processor, "image_processor", None)
    if not image_processor:
        return
    if isinstance(image_processor.size, dict):
        image_processor.size["height"] = image_size
        image_processor.size["width"] = image_size
    else:
        image_processor.size = {"height": image_size, "width": image_size}


def run_validation(
    model: BlipForConditionalGeneration,
    val_loader: DataLoader,
    device: torch.device,
    use_mixed_precision: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    autocast_enabled = use_mixed_precision and device.type in {"mps", "cuda"}

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                outputs = model(**batch)
                loss = outputs.loss.float()
            total_loss += loss.item()

    return total_loss / max(1, len(val_loader))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1: BLIP fine-tuning with gradient checkpointing and mixed precision."
    )
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--annotation_path", type=str, default="src/data/processed/subset_10k.jsonl")
    parser.add_argument("--image_dir", type=str, default="src/data/raw/train2017")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--sample_limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224",
    )
    parser.add_argument("--disable_mixed_precision", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = get_device()
    use_mixed_precision = not args.disable_mixed_precision
    print(f"Using device: {device}")
    print(f"Mixed precision enabled: {use_mixed_precision}")

    processor = BlipProcessor.from_pretrained(args.model_name)
    maybe_set_processor_resolution(processor, args.image_size)

    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    dataset = CocoCaptionDataset(
        annotation_path=args.annotation_path,
        image_folder=args.image_dir,
        processor=processor,
        max_length=args.max_length,
        sample_limit=args.sample_limit,
    )
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and use_mixed_precision)
    autocast_enabled = use_mixed_precision and device.type in {"mps", "cuda"}

    best_val_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                outputs = model(**batch)
                loss = outputs.loss

            loss_fp32 = loss.float()

            if scaler.is_enabled():
                scaler.scale(loss_fp32).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_fp32.backward()
                optimizer.step()

            running_loss += loss_fp32.item()
            progress.set_postfix(train_loss=f"{loss_fp32.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        val_loss = run_validation(model, val_loader, device, use_mixed_precision)
        epoch_time = time.time() - epoch_start

        metrics: Dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_seconds": epoch_time,
        }
        history.append(metrics)
        print(json.dumps(metrics))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"Saved best checkpoint to {args.output_dir}")

    metrics_path = os.path.join(args.output_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Training complete. Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()

