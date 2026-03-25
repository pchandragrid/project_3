import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from tasks.task4_style_steering.utils import (
    classify_style,
    cosine_pairwise_diversity,
    load_jsonl,
    save_json,
    tokenize,
    top_keywords,
    unique_ngram_ratio,
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


def nucleus_sample_from_logits(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= top_p
    if not torch.any(keep):
        keep[0] = True
    keep_ids = sorted_ids[keep]
    keep_probs = sorted_probs[keep]
    keep_probs = keep_probs / keep_probs.sum()
    sampled_idx = torch.multinomial(keep_probs, num_samples=1).item()
    return int(keep_ids[sampled_idx].item())


def compute_prebeam_hidden_diversity(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
    device: torch.device,
    top_k: int = 8,
) -> float:
    with torch.no_grad():
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
        enc = vision_outputs.last_hidden_state
        enc_mask = torch.ones(enc.shape[:2], dtype=torch.long, device=device)

        start_id = get_start_token_id(processor, model)
        inp = torch.tensor([[start_id]], dtype=torch.long, device=device)
        out = model.text_decoder(
            input_ids=inp,
            attention_mask=torch.ones_like(inp),
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits[0, -1, :]
        token_emb = model.text_decoder.bert.embeddings.word_embeddings.weight  # [V, D]
        top_ids = torch.topk(logits, k=min(top_k, logits.numel())).indices
        vecs = token_emb[top_ids].detach().float().cpu().numpy()
    return cosine_pairwise_diversity(vecs)


def generate_nucleus_captions(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
    device: torch.device,
    num_captions: int,
    top_p: float,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    captions = []
    eos_id = processor.tokenizer.eos_token_id
    start_id = get_start_token_id(processor, model)

    with torch.no_grad():
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
        enc = vision_outputs.last_hidden_state
        enc_mask = torch.ones(enc.shape[:2], dtype=torch.long, device=device)

        for _ in range(num_captions):
            ids = torch.tensor([[start_id]], dtype=torch.long, device=device)
            for _step in range(max_new_tokens):
                out = model.text_decoder(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    encoder_hidden_states=enc,
                    encoder_attention_mask=enc_mask,
                    return_dict=True,
                )
                next_id = nucleus_sample_from_logits(out.logits[0, -1, :], top_p=top_p, temperature=temperature)
                ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
                if eos_id is not None and next_id == int(eos_id):
                    break
            cap = processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
            captions.append(cap)
    return captions


def compute_style_direction(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    samples: List[Dict],
    image_dir: str,
    style_from: str,
    style_to: str,
    device: torch.device,
    max_samples: int,
) -> Optional[np.ndarray]:
    style_vecs = defaultdict(list)
    used = 0

    for row in tqdm(samples, desc=f"style vectors {style_from}->{style_to}"):
        image_path = os.path.join(image_dir, row["image"])
        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            enc = model.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state
            enc_mask = torch.ones(enc.shape[:2], dtype=torch.long, device=device)

            for cap in row.get("captions", []):
                style = classify_style(cap)
                if style not in {style_from, style_to}:
                    continue
                toks = processor.tokenizer(cap, return_tensors="pt", truncation=True, max_length=40).to(device)
                out = model.text_decoder(
                    input_ids=toks["input_ids"],
                    attention_mask=toks["attention_mask"],
                    encoder_hidden_states=enc,
                    encoder_attention_mask=enc_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_h = out.hidden_states[-1][0]  # [T, D]
                vec = last_h.mean(dim=0).detach().float().cpu().numpy()
                style_vecs[style].append(vec)
                used += 1
                if used >= max_samples:
                    break
        if used >= max_samples:
            break

    if not style_vecs[style_from] or not style_vecs[style_to]:
        return None
    mean_from = np.mean(np.stack(style_vecs[style_from], axis=0), axis=0)
    mean_to = np.mean(np.stack(style_vecs[style_to], axis=0), axis=0)
    direction = mean_to - mean_from
    norm = np.linalg.norm(direction) + 1e-8
    return (direction / norm).astype(np.float32)


def generate_with_steering(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
    direction: np.ndarray,
    lam: float,
    device: torch.device,
    top_p: float,
    max_new_tokens: int,
    temperature: float,
) -> str:
    eos_id = processor.tokenizer.eos_token_id
    start_id = get_start_token_id(processor, model)
    dir_t = torch.from_numpy(direction).to(device).view(1, 1, -1)

    with torch.no_grad():
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        enc = model.vision_model(pixel_values=pixel_values, return_dict=True).last_hidden_state
        enc_mask = torch.ones(enc.shape[:2], dtype=torch.long, device=device)
        ids = torch.tensor([[start_id]], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            out = model.text_decoder(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                encoder_hidden_states=enc,
                encoder_attention_mask=enc_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1][:, -1:, :]  # [1,1,D]
            steered = last_hidden + lam * dir_t

            if hasattr(model.text_decoder, "cls"):
                steered_logits = model.text_decoder.cls(steered)[:, -1, :]
            else:
                steered_logits = out.logits[:, -1, :]

            next_id = nucleus_sample_from_logits(steered_logits[0], top_p=top_p, temperature=temperature)
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if eos_id is not None and next_id == int(eos_id):
                break
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def calibrate_direction_sign(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    rows: List[Dict],
    image_dir: str,
    direction: np.ndarray,
    lam: float,
    device: torch.device,
    top_p: float,
    max_new_tokens: int,
    temperature: float,
    calibration_images: int,
) -> np.ndarray:
    subset = rows[: min(calibration_images, len(rows))]
    if not subset:
        return direction

    diffs = []
    for row in subset:
        image_path = os.path.join(image_dir, row["image"])
        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path).convert("RGB")
        baseline = generate_nucleus_captions(
            model=model,
            processor=processor,
            image=image,
            device=device,
            num_captions=1,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0]
        steered = generate_with_steering(
            model=model,
            processor=processor,
            image=image,
            direction=direction,
            lam=lam,
            device=device,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        diffs.append(len(tokenize(steered)) - len(tokenize(baseline)))

    if diffs and float(np.mean(diffs)) < 0.0:
        return -direction
    return direction


def draw_style_shift_chart(stats: Dict[str, float], output_path: str) -> None:
    labels = list(stats.keys())
    values = [stats[k] for k in labels]

    w, h = 860, 460
    canvas = Image.new("RGB", (w, h), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "Task 4 Style Steering: Mean Caption Length", fill=(20, 20, 20))

    margin_x = 90
    baseline_y = 390
    bar_w = 120
    gap = 55
    vmax = max(values) if values else 1.0
    scale = 270.0 / max(1e-8, vmax)

    for i, (lab, val) in enumerate(zip(labels, values)):
        x0 = margin_x + i * (bar_w + gap)
        x1 = x0 + bar_w
        y1 = baseline_y
        y0 = int(y1 - val * scale)
        color = (80 + 20 * i, 120 + 15 * i, 210 - 18 * i)
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255), width=2)
        draw.text((x0 + 18, y1 + 10), lab, fill=(30, 30, 30))
        draw.text((x0 + 25, y0 - 22), f"{val:.1f}", fill=(30, 30, 30))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def run(args) -> Dict:
    device = get_device(args.device)
    os.makedirs(args.caption_artifact_dir, exist_ok=True)
    os.makedirs(args.steering_artifact_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    model = BlipForConditionalGeneration.from_pretrained(args.checkpoint_dir).to(device)
    processor = BlipProcessor.from_pretrained(args.checkpoint_dir)
    model.eval()
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    eval_rows = load_jsonl(args.annotation_path, limit=args.num_images)
    style_rows = load_jsonl(args.style_annotation_path, limit=args.style_samples)

    diversity_records = []
    all_generated = []

    for row in tqdm(eval_rows, desc="Task4 diversity sampling"):
        image_path = os.path.join(args.image_dir, row["image"])
        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path).convert("RGB")

        t0 = time.perf_counter()
        captions = generate_nucleus_captions(
            model=model,
            processor=processor,
            image=image,
            device=device,
            num_captions=args.num_captions_per_image,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        diversity = unique_ngram_ratio(captions, n_values=(1, 2, 3))
        hidden_div = compute_prebeam_hidden_diversity(
            model=model,
            processor=processor,
            image=image,
            device=device,
            top_k=args.hidden_div_topk,
        )

        diversity_records.append(
            {
                "image": row["image"],
                "caption_diversity": diversity,
                "prebeam_hidden_diversity": hidden_div,
                "latency_ms_for_5_caps": latency_ms,
                "captions": captions,
            }
        )
        all_generated.extend(captions)

    diversity_records = sorted(diversity_records, key=lambda x: x["caption_diversity"])
    low_k = max(1, min(args.top_k_images, len(diversity_records)))
    low_div = diversity_records[:low_k]
    high_div = diversity_records[-low_k:]

    low_keywords = top_keywords((c for r in low_div for c in r["captions"]), top_k=12)
    high_keywords = top_keywords((c for r in high_div for c in r["captions"]), top_k=12)

    diversity_jsonl = os.path.join(args.caption_artifact_dir, "diversity_captions.jsonl")
    with open(diversity_jsonl, "w", encoding="utf-8") as handle:
        for rec in diversity_records:
            handle.write(json.dumps(rec) + "\n")

    # Steering vectors
    long_dir = compute_style_direction(
        model=model,
        processor=processor,
        samples=style_rows,
        image_dir=args.style_image_dir,
        style_from="short",
        style_to="long",
        device=device,
        max_samples=args.max_style_hidden_samples,
    )
    detail_dir = compute_style_direction(
        model=model,
        processor=processor,
        samples=style_rows,
        image_dir=args.style_image_dir,
        style_from="short",
        style_to="detailed",
        device=device,
        max_samples=args.max_style_hidden_samples,
    )

    if long_dir is None or detail_dir is None:
        raise RuntimeError(
            "Could not build style directions. Increase style samples or verify captions include short/long/detailed styles."
        )

    np.save(os.path.join(args.steering_artifact_dir, "dir_short_to_long.npy"), long_dir)
    np.save(os.path.join(args.steering_artifact_dir, "dir_short_to_detailed.npy"), detail_dir)

    steer_rows = eval_rows[: min(args.num_steer_images, len(eval_rows))]
    long_dir = calibrate_direction_sign(
        model=model,
        processor=processor,
        rows=steer_rows,
        image_dir=args.image_dir,
        direction=long_dir,
        lam=args.lambda_long,
        device=device,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        calibration_images=args.calibration_images,
    )

    steering_records = []
    for row in tqdm(steer_rows, desc="Task4 steering generation"):
        image_path = os.path.join(args.image_dir, row["image"])
        if not os.path.exists(image_path):
            continue
        image = Image.open(image_path).convert("RGB")

        baseline = generate_nucleus_captions(
            model=model,
            processor=processor,
            image=image,
            device=device,
            num_captions=1,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )[0]
        steered_long = generate_with_steering(
            model=model,
            processor=processor,
            image=image,
            direction=long_dir,
            lam=args.lambda_long,
            device=device,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        steered_short = generate_with_steering(
            model=model,
            processor=processor,
            image=image,
            direction=long_dir,
            lam=-abs(args.lambda_short),
            device=device,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        steered_detail = generate_with_steering(
            model=model,
            processor=processor,
            image=image,
            direction=detail_dir,
            lam=args.lambda_detail,
            device=device,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        steering_records.append(
            {
                "image": row["image"],
                "baseline": baseline,
                "steered_short": steered_short,
                "steered_long": steered_long,
                "steered_detailed": steered_detail,
                "len_baseline": len(tokenize(baseline)),
                "len_short": len(tokenize(steered_short)),
                "len_long": len(tokenize(steered_long)),
                "len_detailed": len(tokenize(steered_detail)),
            }
        )

    steering_csv = os.path.join(args.report_dir, "steering_outputs.csv")
    with open(steering_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "baseline",
                "steered_short",
                "steered_long",
                "steered_detailed",
                "len_baseline",
                "len_short",
                "len_long",
                "len_detailed",
            ],
        )
        writer.writeheader()
        writer.writerows(steering_records)

    mean_stats = {
        "baseline": float(np.mean([r["len_baseline"] for r in steering_records])) if steering_records else 0.0,
        "short": float(np.mean([r["len_short"] for r in steering_records])) if steering_records else 0.0,
        "long": float(np.mean([r["len_long"] for r in steering_records])) if steering_records else 0.0,
        "detailed": float(np.mean([r["len_detailed"] for r in steering_records])) if steering_records else 0.0,
    }
    chart_path = os.path.join(args.figure_dir, "style_length_shift.png")
    draw_style_shift_chart(mean_stats, chart_path)

    summary_md = os.path.join(args.report_dir, "style_steering_summary.md")
    with open(summary_md, "w", encoding="utf-8") as handle:
        handle.write("# Task 4 Diversity and Style Steering Summary\n\n")
        handle.write(f"- Images evaluated for diversity: {len(diversity_records)}\n")
        handle.write(f"- Captions per image: {args.num_captions_per_image} (nucleus sampling, p={args.top_p})\n")
        handle.write(
            f"- Mean caption diversity (unique ngrams / total ngrams): "
            f"{np.mean([r['caption_diversity'] for r in diversity_records]) if diversity_records else 0.0:.4f}\n"
        )
        handle.write(
            f"- Mean pre-beam hidden-state diversity: "
            f"{np.mean([r['prebeam_hidden_diversity'] for r in diversity_records]) if diversity_records else 0.0:.4f}\n"
        )
        handle.write("\n## Diverse vs Repetitive Image Types (keyword proxy)\n\n")
        handle.write(f"- Low-diversity keywords: {low_keywords}\n")
        handle.write(f"- High-diversity keywords: {high_keywords}\n")
        handle.write("\n## Steering Effect (Mean Caption Length)\n\n")
        handle.write(f"- Baseline: {mean_stats['baseline']:.2f}\n")
        handle.write(f"- Steered Short: {mean_stats['short']:.2f}\n")
        handle.write(f"- Steered Long: {mean_stats['long']:.2f}\n")
        handle.write(f"- Steered Detailed: {mean_stats['detailed']:.2f}\n")
        handle.write("\n## Interpretation\n\n")
        handle.write("- If steered-long > baseline > steered-short, steering direction is working.\n")
        handle.write("- Detailed steering should increase descriptive tokens and relational words.\n")

    summary_json = {
        "diversity_jsonl": diversity_jsonl,
        "steering_csv": steering_csv,
        "summary_md": summary_md,
        "style_shift_chart": chart_path,
        "mean_style_lengths": mean_stats,
        "low_diversity_keywords": low_keywords,
        "high_diversity_keywords": high_keywords,
    }
    save_json(os.path.join(args.report_dir, "style_steering_summary.json"), summary_json)
    return summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: Caption diversity + concept activation vector style steering.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224",
    )
    parser.add_argument("--annotation_path", type=str, default="src/data/raw/captions_validation.jsonl")
    parser.add_argument("--image_dir", type=str, default="src/data/raw/val2017")
    parser.add_argument(
        "--style_annotation_path",
        type=str,
        default="src/data/processed/subset_10k.jsonl",
    )
    parser.add_argument("--style_image_dir", type=str, default="src/data/raw/train2017")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--num_steer_images", type=int, default=80)
    parser.add_argument("--num_captions_per_image", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--max_style_hidden_samples", type=int, default=300)
    parser.add_argument("--style_samples", type=int, default=3000)
    parser.add_argument("--hidden_div_topk", type=int, default=8)
    parser.add_argument("--lambda_short", type=float, default=0.8)
    parser.add_argument("--lambda_long", type=float, default=0.8)
    parser.add_argument("--lambda_detail", type=float, default=0.8)
    parser.add_argument("--top_k_images", type=int, default=20)
    parser.add_argument("--calibration_images", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument(
        "--caption_artifact_dir",
        type=str,
        default="tasks/task4_style_steering/artifacts/captions",
    )
    parser.add_argument(
        "--steering_artifact_dir",
        type=str,
        default="tasks/task4_style_steering/artifacts/steering",
    )
    parser.add_argument("--report_dir", type=str, default="tasks/task4_style_steering/results/reports")
    parser.add_argument("--figure_dir", type=str, default="tasks/task4_style_steering/results/figures")
    args = parser.parse_args()

    out = run(args)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

