import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class COCODatasetGIT(Dataset):

    def __init__(self, annotation_file, image_folder, processor, mode="mixed"):

        self.annotations = []
        self.image_folder = image_folder
        self.processor = processor
        self.mode = mode

        # Proper JSONL loading
        with open(annotation_file, "r") as f:
            for line in f:
                self.annotations.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.annotations)

    def select_caption(self, captions):

        if self.mode == "short":
            captions = [c for c in captions if len(c.split()) <= 10]

        elif self.mode == "long":
            captions = [c for c in captions if len(c.split()) > 10]

        if len(captions) == 0:
            captions = self.annotations[
                random.randint(0, len(self.annotations) - 1)
            ]["captions"]

        return random.choice(captions)

    def __getitem__(self, idx):

        ann = self.annotations[idx]

        image_path = os.path.join(self.image_folder, ann["image"])
        image = Image.open(image_path).convert("RGB")

        caption = self.select_caption(ann["captions"])

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values = encoding["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids   # GIT uses input_ids as labels
        }