import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image


class COCODatasetViTGPT2(Dataset):
    def __init__(self,
                 annotation_path,
                 image_folder,
                 image_processor,
                 tokenizer,
                 mode="short",
                 max_length=20):

        self.image_folder = image_folder
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        with open(annotation_path, "r") as f:
            raw_data = [json.loads(line) for line in f]

        self.annotations = []

        for ann in raw_data:
            filtered = []

            for cap in ann["captions"]:
                words = cap.split()
                wc = len(words)

                if mode == "short" and wc <= 8:
                    filtered.append(cap)
                elif mode == "long" and wc > 15:
                    filtered.append(cap)
                elif mode == "mixed":
                    filtered.append(cap)

            if len(filtered) > 0:
                self.annotations.append({
                    "image": ann["image"],
                    "captions": filtered
                })

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        ann = self.annotations[idx]
        caption = random.choice(ann["captions"])

        image_path = os.path.join(self.image_folder, ann["image"])
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = tokenized.input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": input_ids
        }