# Task 1 - BLIP Optimization (Memory-Efficient Training + CoreML Export)

This task is fully implemented outside `src` in `tasks/task1_blip_optimization/`.

## What this task does

1. Fine-tunes BLIP on 10k captions with:
   - gradient checkpointing
   - mixed precision forward (fp16), fp32 loss handling
   - 224px training resolution
2. Exports encoder and decoder to ONNX with dynamic axes.
3. Converts to CoreML (`CPU_AND_NE`) and applies 4-bit quantization.
4. Benchmarks PyTorch vs quantized CoreML on latency + BLEU-4.

## Folder layout

- `train_blip_memory_efficient.py` - memory-efficient fine-tuning
- `export_onnx.py` - ONNX export for encoder + decoder
- `convert_coreml.py` - CoreML conversion + 4-bit quantization
- `benchmark.py` - latency/quality benchmark
- `checkpoints/` - trained BLIP checkpoints
- `exports/onnx/` - ONNX artifacts
- `exports/coreml/` - CoreML artifacts
- `results/` - benchmark outputs

## Step-by-step run

### 1) Train
```bash
python -m tasks.task1_blip_optimization.train_blip_memory_efficient \
  --annotation_path src/data/processed/subset_10k.jsonl \
  --image_dir src/data/raw/train2017 \
  --batch_size 4 \
  --image_size 224
```

### 2) Export ONNX
```bash
python -m tasks.task1_blip_optimization.export_onnx
```

### 3) Convert to CoreML + quantize
Recommended stable path in this repo:
```bash
python -m tasks.task1_blip_optimization.convert_coreml \
  --compute_units CPU_AND_NE \
  --conversion_mode torch
```

### 4) Benchmark
```bash
python -m tasks.task1_blip_optimization.benchmark \
  --compute_units CPU_AND_NE \
  --num_samples 100
```

## Key outputs

- Checkpoint: `tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224/`
- ONNX: `tasks/task1_blip_optimization/exports/onnx/`
- CoreML: `tasks/task1_blip_optimization/exports/coreml/`
- Benchmark:
  - `tasks/task1_blip_optimization/results/coreml_benchmark.json`
  - `tasks/task1_blip_optimization/results/coreml_benchmark.md`

## Notes

- If ONNX->CoreML conversion is unavailable in your local stack, `--conversion_mode torch` handles conversion directly from model wrappers.
- CoreML conversion works best in Python 3.11/3.12 environment.

