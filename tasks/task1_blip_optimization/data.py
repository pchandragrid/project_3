import json
import os
import random
from typing import Dict, List

from PIL import Image
from torch.utils.data import Dataset


class CocoCaptionDataset(Dataset):
    """COCO-style caption dataset backed by jsonl annotations."""

    def __init__(
        self,
        annotation_path: str,
        image_folder: str,
        processor,
        max_length: int = 40,
        sample_limit: int = 0,
    ) -> None:
        self.annotation_path = annotation_path
        self.image_folder = image_folder
        self.processor = processor
        self.max_length = max_length

        with open(annotation_path, "r", encoding="utf-8") as handle:
            self.annotations: List[Dict] = [json.loads(line) for line in handle if line.strip()]

        if sample_limit and sample_limit > 0:
            self.annotations = self.annotations[:sample_limit]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        row = self.annotations[idx]
        caption = random.choice(row["captions"]) if row.get("captions") else ""
        image_path = os.path.join(self.image_folder, row["image"])
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": input_ids.clone(),
        }

