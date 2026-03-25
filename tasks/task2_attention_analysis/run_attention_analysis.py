import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from tasks.task2_attention_analysis.attention_utils import (
    attention_rollout,
    binarize_heatmap,
    boxes_for_word,
    boxes_to_mask,
    build_coco_box_index,
    iou,
    parse_image_id_from_filename,
    plot_2x5_attention_grid,
    vector_to_grid,
)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_jsonl(path: str, limit: int) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def decode_words(caption: str) -> List[str]:
    return [w for w in caption.strip().split() if w]


def get_start_token_id(processor: BlipProcessor, model: BlipForConditionalGeneration) -> int:
    text_cfg = getattr(model.config, "text_config", None)
    for candidate in (
        getattr(model.config, "decoder_start_token_id", None),
        getattr(text_cfg, "decoder_start_token_id", None),
        getattr(text_cfg, "bos_token_id", None),
        processor.tokenizer.bos_token_id,
        processor.tokenizer.cls_token_id,
        processor.tokenizer.eos_token_id,
    ):
        if candidate is not None:
            return int(candidate)
    return 0


def generate_with_cross_attention(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
    device: torch.device,
    max_new_tokens: int,
) -> Tuple[str, List[str], List[List[np.ndarray]]]:
    model.eval()
    with torch.no_grad():
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
        encoder_hidden_states = vision_outputs.last_hidden_state
        encoder_attention_mask = torch.ones(
            encoder_hidden_states.shape[:2], dtype=torch.long, device=device
        )

        start_id = get_start_token_id(processor, model)
        eos_id = processor.tokenizer.eos_token_id
        input_ids = torch.tensor([[start_id]], dtype=torch.long, device=device)

        token_texts: List[str] = []
        step_layer_vectors: List[List[np.ndarray]] = []

        for _ in range(max_new_tokens):
            decoder_outputs = model.text_decoder(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=True,
                return_dict=True,
            )

            layer_vectors: List[np.ndarray] = []
            if decoder_outputs.cross_attentions:
                for layer_attn in decoder_outputs.cross_attentions:
                    arr = layer_attn.detach().float().cpu().numpy()  # [B, H, T, S]
                    vec = arr[0].mean(axis=0)[-1]
                    layer_vectors.append(vec)
            if layer_vectors:
                step_layer_vectors.append(layer_vectors)

            next_token = int(torch.argmax(decoder_outputs.logits[:, -1, :], dim=-1).item())
            token_piece = processor.tokenizer.decode([next_token], skip_special_tokens=True).strip()
            token_texts.append(token_piece if token_piece else f"<{next_token}>")

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            if eos_id is not None and next_token == int(eos_id):
                break

        caption = processor.tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
        return caption, token_texts, step_layer_vectors


def run(args) -> Dict:
    device = get_device(args.device)
    os.makedirs(args.figure_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.attention_maps_dir, exist_ok=True)
    os.makedirs(args.rollout_maps_dir, exist_ok=True)

    model = BlipForConditionalGeneration.from_pretrained(args.checkpoint_dir).to(device)
    processor = BlipProcessor.from_pretrained(args.checkpoint_dir)
    model.eval()

    samples = load_jsonl(args.annotation_path, args.num_samples)
    coco_index = build_coco_box_index(args.instances_json)

    rows = []
    iou_by_caption_len: Dict[int, List[float]] = defaultdict(list)

    for idx, row in enumerate(tqdm(samples, desc="Task2 attention analysis")):
        image_name = row["image"]
        image_path = os.path.join(args.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image)
        height, width = image_np.shape[:2]

        caption, step_words, step_layer_vectors = generate_with_cross_attention(
            model=model,
            processor=processor,
            image=image,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        words = decode_words(caption)
        if not step_words:
            step_words = words
        step_layer_vectors = step_layer_vectors[: args.grid_steps]
        if not step_layer_vectors:
            continue

        token_maps = []
        per_word_iou = []
        image_id = parse_image_id_from_filename(image_name)

        for step_i, layer_vectors in enumerate(step_layer_vectors):
            per_layer_map = [vector_to_grid(v, (height, width), patch_size=args.patch_size) for v in layer_vectors]
            rollout_vec = attention_rollout(layer_vectors)
            rollout_map = vector_to_grid(rollout_vec, (height, width), patch_size=args.patch_size)
            token_maps.append(rollout_map)

            # Persist per-step maps into artifact folders.
            attn_map_path = os.path.join(
                args.attention_maps_dir,
                f"img_{idx:04d}_step_{step_i:02d}_layers.npz",
            )
            rollout_map_path = os.path.join(
                args.rollout_maps_dir,
                f"img_{idx:04d}_step_{step_i:02d}_rollout.npy",
            )
            np.savez_compressed(attn_map_path, *per_layer_map)
            np.save(rollout_map_path, rollout_map)

            if step_i < len(step_words):
                word = step_words[step_i]
                boxes = boxes_for_word(coco_index, image_id, word)
                if boxes:
                    attn_mask = binarize_heatmap(rollout_map, quantile=args.heatmap_quantile)
                    box_mask = boxes_to_mask(boxes, (height, width))
                    score = iou(attn_mask, box_mask)
                    per_word_iou.append(score)
                    iou_by_caption_len[len(words)].append(score)

        fig_path = os.path.join(args.figure_dir, f"attention_grid_{idx:04d}.png")
        plot_2x5_attention_grid(
            image=image,
            words=step_words,
            token_maps=token_maps,
            output_path=fig_path,
            title=f"Image: {image_name} | Caption: {caption}",
        )

        rows.append(
            {
                "image": image_name,
                "caption": caption,
                "caption_length": len(words),
                "mean_alignment_iou": float(np.mean(per_word_iou)) if per_word_iou else 0.0,
                "num_aligned_words": len(per_word_iou),
                "figure_path": fig_path,
            }
        )
        if not per_word_iou:
            iou_by_caption_len[len(words)].append(0.0)

    detail_csv = os.path.join(args.report_dir, "alignment_details.csv")
    with open(detail_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "caption",
                "caption_length",
                "mean_alignment_iou",
                "num_aligned_words",
                "figure_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for cap_len in sorted(iou_by_caption_len):
        vals = iou_by_caption_len[cap_len]
        summary_rows.append(
            {
                "caption_length": cap_len,
                "mean_alignment_iou": float(np.mean(vals)) if vals else 0.0,
                "num_word_matches": len(vals),
            }
        )

    summary_csv = os.path.join(args.report_dir, "caption_length_alignment_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["caption_length", "mean_alignment_iou", "num_word_matches"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md = os.path.join(args.report_dir, "attention_summary.md")
    with open(summary_md, "w", encoding="utf-8") as handle:
        handle.write("# Task 2 Attention Analysis Summary\n\n")
        handle.write(f"- Images analyzed: {len(rows)}\n")
        handle.write(f"- Checkpoint: `{args.checkpoint_dir}`\n")
        handle.write(f"- Caption generation max_new_tokens: {args.max_new_tokens}\n")
        handle.write("\n## Caption Length -> Mean Alignment IoU\n\n")
        handle.write("| Caption Length | Mean Alignment IoU | Word Matches |\n")
        handle.write("|---:|---:|---:|\n")
        for item in summary_rows:
            handle.write(
                f"| {item['caption_length']} | {item['mean_alignment_iou']:.4f} | {item['num_word_matches']} |\n"
            )
        handle.write("\n## Interpretation Notes\n\n")
        handle.write("- Higher IoU indicates better grounding of words to object regions.\n")
        handle.write("- Very long captions may show lower mean IoU due to attention diffusion.\n")
        handle.write("- Check low-IoU words as potential hallucination candidates.\n")

    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "attention_maps_dir": args.attention_maps_dir,
        "rollout_maps_dir": args.rollout_maps_dir,
        "num_rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: BLIP cross-attention visualization and rollout analysis.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224",
    )
    parser.add_argument("--annotation_path", type=str, default="src/data/raw/captions_validation.jsonl")
    parser.add_argument("--image_dir", type=str, default="src/data/raw/val2017")
    parser.add_argument(
        "--instances_json",
        type=str,
        default="src/data/raw/instances_val2017.json",
        help="COCO instances json with bounding boxes. If missing, IoU metrics become zero/no matches.",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Reserved; analysis currently uses greedy decoding for stable per-step cross-attention.",
    )
    parser.add_argument("--grid_steps", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--heatmap_quantile", type=float, default=0.85)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--figure_dir", type=str, default="tasks/task2_attention_analysis/results/figures")
    parser.add_argument("--report_dir", type=str, default="tasks/task2_attention_analysis/results/reports")
    parser.add_argument(
        "--attention_maps_dir",
        type=str,
        default="tasks/task2_attention_analysis/artifacts/attention_maps",
    )
    parser.add_argument(
        "--rollout_maps_dir",
        type=str,
        default="tasks/task2_attention_analysis/artifacts/rollout_maps",
    )
    args = parser.parse_args()

    outputs = run(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

