import argparse
import json
import os
import statistics
import time
from typing import Dict, List

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import BlipForConditionalGeneration, BlipProcessor


def load_jsonl(path: str, limit: int) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_start_token_id(processor: BlipProcessor, model: BlipForConditionalGeneration) -> int:
    tokenizer = processor.tokenizer
    text_config = getattr(model.config, "text_config", None)
    for candidate in (
        getattr(model.config, "decoder_start_token_id", None),
        getattr(text_config, "decoder_start_token_id", None),
        getattr(text_config, "bos_token_id", None),
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
    ):
        if candidate is not None:
            return int(candidate)
    return 0


def pytorch_generate(
    model: BlipForConditionalGeneration,
    processor: BlipProcessor,
    image: Image.Image,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, num_beams=1, max_new_tokens=max_new_tokens)
    return processor.decode(output_ids[0], skip_special_tokens=True)


def _select_output_value(outputs: Dict, preferred_key: str):
    if preferred_key in outputs:
        return outputs[preferred_key]
    return next(iter(outputs.values()))


def coreml_generate(
    encoder_model,
    decoder_model,
    processor: BlipProcessor,
    image: Image.Image,
    start_token_id: int,
    max_new_tokens: int,
) -> str:
    encoded = processor(images=image, return_tensors="pt")
    pixel_values = encoded["pixel_values"].cpu().numpy().astype(np.float32)
    encoder_outputs = encoder_model.predict({"pixel_values": pixel_values})
    encoder_hidden_states = _select_output_value(encoder_outputs, "encoder_hidden_states")

    eos_token = processor.tokenizer.eos_token_id
    token_ids = [start_token_id]

    for _ in range(max_new_tokens):
        input_ids = np.array([token_ids], dtype=np.int32)
        attention_mask = np.ones_like(input_ids, dtype=np.int32)
        decoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states.astype(np.float32),
        }
        decoder_outputs = decoder_model.predict(decoder_inputs)
        logits = _select_output_value(decoder_outputs, "logits")
        next_token = int(np.argmax(logits[0, -1, :]))
        token_ids.append(next_token)
        if eos_token is not None and next_token == eos_token:
            break

    return processor.tokenizer.decode(token_ids, skip_special_tokens=True)


def compute_bleu4(pred: str, references: List[str]) -> float:
    references_tokens = [ref.lower().split() for ref in references if ref]
    prediction_tokens = pred.lower().split()
    if not references_tokens or not prediction_tokens:
        return 0.0
    smoother = SmoothingFunction().method1
    return float(
        sentence_bleu(
            references_tokens,
            prediction_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoother,
        )
    )


def run_benchmark(args) -> Dict:
    samples = load_jsonl(args.annotation_path, args.num_samples)
    device = get_device()

    model = BlipForConditionalGeneration.from_pretrained(args.checkpoint_dir).to(device)
    model.eval()
    processor = BlipProcessor.from_pretrained(args.checkpoint_dir)

    compute_units = getattr(ct.ComputeUnit, args.compute_units)
    encoder_model = ct.models.MLModel(args.coreml_encoder, compute_units=compute_units)
    decoder_model = ct.models.MLModel(args.coreml_decoder, compute_units=compute_units)
    start_token_id = get_start_token_id(processor, model)

    torch_latencies = []
    coreml_latencies = []
    torch_bleu = []
    coreml_bleu = []

    for row in samples:
        image_path = os.path.join(args.image_dir, row["image"])
        image = Image.open(image_path).convert("RGB")
        references = row.get("captions", [])

        t0 = time.perf_counter()
        torch_caption = pytorch_generate(model, processor, image, device, args.max_new_tokens)
        torch_latencies.append((time.perf_counter() - t0) * 1000.0)
        torch_bleu.append(compute_bleu4(torch_caption, references))

        t1 = time.perf_counter()
        coreml_caption = coreml_generate(
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            processor=processor,
            image=image,
            start_token_id=start_token_id,
            max_new_tokens=args.max_new_tokens,
        )
        coreml_latencies.append((time.perf_counter() - t1) * 1000.0)
        coreml_bleu.append(compute_bleu4(coreml_caption, references))

    return {
        "num_samples": len(samples),
        "pytorch_full_precision": {
            "latency_ms_mean": statistics.mean(torch_latencies),
            "latency_ms_std": statistics.pstdev(torch_latencies) if len(torch_latencies) > 1 else 0.0,
            "bleu4_mean": statistics.mean(torch_bleu),
        },
        "coreml_quantized_q4": {
            "latency_ms_mean": statistics.mean(coreml_latencies),
            "latency_ms_std": statistics.pstdev(coreml_latencies) if len(coreml_latencies) > 1 else 0.0,
            "bleu4_mean": statistics.mean(coreml_bleu),
        },
    }


def write_markdown(report: Dict, path: str) -> None:
    lines = [
        "# Task 1 CoreML Benchmark",
        "",
        f"Evaluated on {report['num_samples']} samples.",
        "",
        "| Model | Mean Latency (ms) | Std (ms) | BLEU-4 |",
        "|---|---:|---:|---:|",
        (
            "| PyTorch (full precision) | "
            f"{report['pytorch_full_precision']['latency_ms_mean']:.2f} | "
            f"{report['pytorch_full_precision']['latency_ms_std']:.2f} | "
            f"{report['pytorch_full_precision']['bleu4_mean']:.4f} |"
        ),
        (
            "| CoreML (4-bit quantized) | "
            f"{report['coreml_quantized_q4']['latency_ms_mean']:.2f} | "
            f"{report['coreml_quantized_q4']['latency_ms_std']:.2f} | "
            f"{report['coreml_quantized_q4']['bleu4_mean']:.4f} |"
        ),
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs CoreML quantized BLIP captioning.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224",
    )
    parser.add_argument("--annotation_path", type=str, default="src/data/raw/captions_validation.jsonl")
    parser.add_argument("--image_dir", type=str, default="src/data/raw/val2017")
    parser.add_argument(
        "--coreml_encoder",
        type=str,
        default="tasks/task1_blip_optimization/exports/coreml/blip_encoder_q4.mlpackage",
    )
    parser.add_argument(
        "--coreml_decoder",
        type=str,
        default="tasks/task1_blip_optimization/exports/coreml/blip_decoder_q4.mlpackage",
    )
    parser.add_argument("--compute_units", type=str, default="CPU_AND_NE")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument(
        "--output_json",
        type=str,
        default="tasks/task1_blip_optimization/results/coreml_benchmark.json",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="tasks/task1_blip_optimization/results/coreml_benchmark.md",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    report = run_benchmark(args)

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_markdown(report, args.output_md)

    print(json.dumps(report, indent=2))
    print(f"Wrote report JSON to {args.output_json}")
    print(f"Wrote report markdown to {args.output_md}")


if __name__ == "__main__":
    main()

