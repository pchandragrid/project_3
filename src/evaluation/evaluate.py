import argparse
import os
import torch
import torch.nn.functional as F
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from PIL import Image

# ---------------------------------------
# Load Models
# ---------------------------------------
def load_models():

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Using device:", device)

    caption_model = BlipForConditionalGeneration.from_pretrained("saved_model_phase2")
    caption_processor = BlipProcessor.from_pretrained("saved_model_phase2")

    caption_model.to(device)
    caption_model.eval()

    # Toxicity model
    tox_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
    tox_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

    tox_model.to(device)
    tox_model.eval()

    return caption_model, caption_processor, tox_model, tox_tokenizer, device


# ---------------------------------------
# Generate Caption + Confidence
# ---------------------------------------
def generate_caption(model, processor, image, device):

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=5,
            max_length=20,
            length_penalty=1.0,
            output_scores=True,
            return_dict_in_generate=True
        )

    generated_ids = outputs.sequences
    caption = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    # True confidence
    seq_score = outputs.sequences_scores[0]
    confidence = torch.exp(seq_score).item()

    return caption, confidence


# ---------------------------------------
# Toxicity Score
# ---------------------------------------
def check_toxicity(tox_model, tox_tokenizer, caption, device):

    inputs = tox_tokenizer(
        caption,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = tox_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    toxic_score = probs[0][1].item()
    return toxic_score


# ---------------------------------------
# Evaluate Single Image
# ---------------------------------------
def evaluate_image(image_path, models):

    caption_model, caption_processor, tox_model, tox_tokenizer, device = models

    image = Image.open(image_path).convert("RGB")

    caption, confidence = generate_caption(
        caption_model,
        caption_processor,
        image,
        device
    )

    toxic_score = check_toxicity(
        tox_model,
        tox_tokenizer,
        caption,
        device
    )

    print("\n===================================")
    print("Image:", image_path)
    print("Caption:", caption)
    print(f"Confidence: {confidence:.3f}")
    print(f"Toxicity Score: {toxic_score:.3f}")

    if toxic_score > 0.6:
        print("⚠️ WARNING: Caption flagged as toxic")
    print("===================================\n")


# ---------------------------------------
# Main
# ---------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")

    args = parser.parse_args()

    if not args.image and not args.folder:
        print("Please provide --image or --folder")
        return

    models = load_models()

    if args.image:
        evaluate_image(args.image, models)

    if args.folder:
        for file in os.listdir(args.folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(args.folder, file)
                evaluate_image(path, models)


if __name__ == "__main__":
    main()