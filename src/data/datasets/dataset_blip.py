import json
import os
import random
import re
from torch.utils.data import Dataset
from PIL import Image


class COCODatasetAdvanced(Dataset):
    def __init__(self,
                 annotation_path,
                 image_folder,
                 processor,
                 mode="mixed",
                 max_length=40):

        self.image_folder = image_folder
        self.processor = processor
        self.max_length = max_length
        self.mode = mode

        with open(annotation_path, "r") as f:
            raw_data = [json.loads(line) for line in f]

        self.annotations = []

        for ann in raw_data:

            filtered_captions = []

            for cap in ann["captions"]:

                cap = cap.strip().lower()

                # ---------- QUALITY FILTERS ----------

                # Remove very short captions
                if len(cap.split()) < 3:
                    continue

                # Remove repeated words
                words = cap.split()
                if len(set(words)) < len(words) * 0.6:
                    continue

                # Remove non-alphabetic captions
                if not re.search(r"[a-z]", cap):
                    continue

                word_count = len(words)

                # ---------- LENGTH FILTERS ----------

                if self.mode == "short" and word_count <= 8:
                    filtered_captions.append(cap)

                elif self.mode == "long" and word_count > 15:
                    filtered_captions.append(cap)

                elif self.mode == "mixed":
                    filtered_captions.append(cap)

            if len(filtered_captions) > 0:
                self.annotations.append({
                    "image": ann["image"],
                    "captions": filtered_captions
                })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        ann = self.annotations[idx]
        file_name = ann["image"]
        caption = random.choice(ann["captions"])

        image_path = os.path.join(self.image_folder, file_name)
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": input_ids.clone()
        }