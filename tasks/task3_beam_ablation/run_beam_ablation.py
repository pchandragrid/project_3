import argparse
import csv
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from tasks.task3_beam_ablation.metrics import bleu4_mean, cider_mean, meteor_mean, rouge_l_mean


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


def draw_cider_heatmap(
    configs: Sequence[Tuple[int, float]],
    rows: Sequence[Dict],
    output_path: str,
) -> None:
    beam_values = sorted({int(x[0]) for x in configs})
    penalty_values = sorted({float(x[1]) for x in configs})
    value_map = {(int(r["beam_size"]), float(r["length_penalty"])): float(r["cider"]) for r in rows}

    cell_w, cell_h = 180, 100
    left_w, top_h = 160, 90
    width = left_w + len(penalty_values) * cell_w + 20
    height = top_h + len(beam_values) * cell_h + 20
    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)

    vals = [value_map.get((b, p), 0.0) for b in beam_values for p in penalty_values]
    vmin, vmax = float(min(vals)), float(max(vals))
    denom = (vmax - vmin) + 1e-8

    draw.text((15, 12), "CIDEr Heatmap (Beam Size x Length Penalty)", fill=(20, 20, 20))
    draw.text((left_w + 10, 45), "Length Penalty", fill=(20, 20, 20))
    draw.text((18, top_h + 10), "Beam", fill=(20, 20, 20))

    for j, p in enumerate(penalty_values):
        x = left_w + j * cell_w
        draw.text((x + 60, 65), f"{p:.1f}", fill=(20, 20, 20))

    for i, b in enumerate(beam_values):
        y = top_h + i * cell_h
        draw.text((55, y + 40), str(b), fill=(20, 20, 20))
        for j, p in enumerate(penalty_values):
            score = value_map.get((b, p), 0.0)
            t = (score - vmin) / denom
            # Blue -> Green -> Yellow style gradient.
            color = (
                int(30 + 200 * t),
                int(80 + 140 * t),
                int(220 - 180 * t),
            )
            x0 = left_w + j * cell_w + 8
            y0 = y + 8
            x1 = x0 + cell_w - 16
            y1 = y0 + cell_h - 16
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255), width=2)
            draw.text((x0 + 45, y0 + 35), f"{score:.3f}", fill=(10, 10, 10))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run(args) -> Dict:
    device = get_device(args.device)
    os.makedirs(args.artifact_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    model = BlipForConditionalGeneration.from_pretrained(args.checkpoint_dir).to(device)
    processor = BlipProcessor.from_pretrained(args.checkpoint_dir)
    model.eval()
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    samples = load_jsonl(args.annotation_path, args.num_samples)
    beam_sizes = parse_int_list(args.beam_sizes)
    penalties = parse_float_list(args.length_penalties)
    configs = [(b, p) for b in beam_sizes for p in penalties]

    all_rows = []

    for beam_size, length_penalty in configs:
        preds = []
        refs = []
        latencies_ms = []
        cap_lens = []

        caption_artifact_path = os.path.join(
            args.artifact_dir,
            f"captions_beam{beam_size}_lp{length_penalty:.1f}.jsonl",
        )
        if args.reuse_artifacts and os.path.exists(caption_artifact_path):
            with open(caption_artifact_path, "r", encoding="utf-8") as cap_file:
                for line in cap_file:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    pred = str(payload.get("prediction", "")).strip()
                    refs_row = payload.get("references", [])
                    dt_ms = float(payload.get("latency_ms", 0.0))
                    clen = int(payload.get("caption_length", len(pred.split())))

                    preds.append(pred)
                    refs.append(refs_row)
                    latencies_ms.append(dt_ms)
                    cap_lens.append(clen)
        else:
            with open(caption_artifact_path, "w", encoding="utf-8") as cap_file:
                for row in tqdm(samples, desc=f"beam={beam_size}, lp={length_penalty:.1f}"):
                    image_path = os.path.join(args.image_dir, row["image"])
                    image = Image.open(image_path).convert("RGB")
                    model_inputs = processor(images=image, return_tensors="pt").to(device)

                    t0 = time.perf_counter()
                    gen_kwargs = {
                        "num_beams": beam_size,
                        "max_new_tokens": args.max_new_tokens,
                    }
                    if beam_size > 1:
                        gen_kwargs["length_penalty"] = length_penalty
                    with torch.no_grad():
                        output_ids = model.generate(**model_inputs, **gen_kwargs)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    pred = processor.decode(output_ids[0], skip_special_tokens=True).strip()

                    refs_row = row.get("captions", [])
                    preds.append(pred)
                    refs.append(refs_row)
                    latencies_ms.append(dt_ms)
                    cap_lens.append(len(pred.split()))

                    cap_file.write(
                        json.dumps(
                            {
                                "image": row["image"],
                                "prediction": pred,
                                "references": refs_row,
                                "latency_ms": dt_ms,
                                "caption_length": len(pred.split()),
                            }
                        )
                        + "\n"
                    )

        metrics_row = {
            "beam_size": beam_size,
            "length_penalty": length_penalty,
            "bleu4": bleu4_mean(preds, refs),
            "meteor": meteor_mean(preds, refs),
            "cider": cider_mean(preds, refs),
            "rouge_l": rouge_l_mean(preds, refs),
            "mean_caption_length": float(np.mean(cap_lens)) if cap_lens else 0.0,
            "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
            "samples": len(preds),
            "artifact": caption_artifact_path,
        }
        all_rows.append(metrics_row)

    all_rows = sorted(all_rows, key=lambda x: (x["beam_size"], x["length_penalty"]))
    metrics_csv = os.path.join(args.report_dir, "beam_length_ablation_metrics.csv")
    with open(metrics_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "beam_size",
                "length_penalty",
                "bleu4",
                "meteor",
                "cider",
                "rouge_l",
                "mean_caption_length",
                "mean_latency_ms",
                "p95_latency_ms",
                "samples",
                "artifact",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    heatmap_path = os.path.join(args.figure_dir, "cider_heatmap.png")
    draw_cider_heatmap(configs=configs, rows=all_rows, output_path=heatmap_path)

    best_quality = max(all_rows, key=lambda x: x["cider"])
    # Score balances quality vs speed; higher is better.
    best_tradeoff = max(all_rows, key=lambda x: x["cider"] / max(1.0, x["mean_latency_ms"]))

    summary_md = os.path.join(args.report_dir, "ablation_summary.md")
    with open(summary_md, "w", encoding="utf-8") as handle:
        handle.write("# Task 3 Beam Search and Length Penalty Ablation\n\n")
        handle.write(f"- Samples evaluated: {len(samples)}\n")
        handle.write(f"- Checkpoint: `{args.checkpoint_dir}`\n")
        handle.write(f"- Config grid size: {len(configs)}\n")
        handle.write("\n## Best by CIDEr\n\n")
        handle.write(
            f"- beam_size={best_quality['beam_size']}, length_penalty={best_quality['length_penalty']:.1f}, "
            f"CIDEr={best_quality['cider']:.4f}, latency={best_quality['mean_latency_ms']:.2f} ms\n"
        )
        handle.write("\n## Best Quality/Speed Trade-off\n\n")
        handle.write(
            f"- beam_size={best_tradeoff['beam_size']}, length_penalty={best_tradeoff['length_penalty']:.1f}, "
            f"CIDEr={best_tradeoff['cider']:.4f}, latency={best_tradeoff['mean_latency_ms']:.2f} ms\n"
        )
        handle.write("\n## Notes\n\n")
        handle.write("- Higher beam size generally improves quality but increases latency.\n")
        handle.write("- Length penalty affects caption verbosity and may shift CIDEr/ROUGE.\n")

    payload = {
        "metrics_csv": metrics_csv,
        "heatmap_png": heatmap_path,
        "summary_md": summary_md,
        "best_quality": best_quality,
        "best_tradeoff": best_tradeoff,
    }
    result_json = os.path.join(args.report_dir, "ablation_summary.json")
    with open(result_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["summary_json"] = result_json
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: Beam search and length penalty ablation for BLIP captions.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224",
    )
    parser.add_argument("--annotation_path", type=str, default="src/data/raw/captions_validation.jsonl")
    parser.add_argument("--image_dir", type=str, default="src/data/raw/val2017")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--beam_sizes", type=str, default="1,3,5")
    parser.add_argument("--length_penalties", type=str, default="0.8,1.0,1.2")
    parser.add_argument(
        "--reuse_artifacts",
        action="store_true",
        help="Reuse existing caption artifact files to recompute metrics/plots without regeneration.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--artifact_dir", type=str, default="tasks/task3_beam_ablation/artifacts/captions")
    parser.add_argument("--report_dir", type=str, default="tasks/task3_beam_ablation/results/reports")
    parser.add_argument("--figure_dir", type=str, default="tasks/task3_beam_ablation/results/figures")
    args = parser.parse_args()

    out = run(args)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

