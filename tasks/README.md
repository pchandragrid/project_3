# Project 3 Task Runner Guide

This is the master execution guide for all Project 3 tasks.
All tasks are implemented outside `src` inside the `tasks/` folder.

📊 **[View Full Report](report/index.html)** 

## Task Folder Structure

- `tasks/task1_blip_optimization/`
- `tasks/task2_attention_analysis/`
- `tasks/task3_beam_ablation/`
- `tasks/task4_style_steering/`
- `tasks/task5_fairness_safety/`

## Recommended Execution Order

1. Task 1 (train/export/benchmark)
2. Task 2 (attention + rollout + alignment)
3. Task 3 (beam/length ablation)
4. Task 4 (diversity + style steering)
5. Task 5 (toxicity/bias + mitigation)

---

## Environment Notes

- Use your main training env for model training/export if already stable.
- For CoreML conversion and most task scripts in this project, Python 3.11 env is recommended.
- Example activation:

```bash
source .venv_coreml/bin/activate
```

---

## Task 1 - BLIP Optimization

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

---

## Task 2 - Attention + Rollout Analysis

```bash
python -m tasks.task2_attention_analysis.run_attention_analysis \
  --checkpoint_dir tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224 \
  --annotation_path src/data/raw/captions_validation.jsonl \
  --image_dir src/data/raw/val2017 \
  --instances_json src/data/raw/instances_val2017.json \
  --num_samples 100 \
  --max_new_tokens 20 \
  --grid_steps 5 \
  --device cpu
```

Artifacts:
- `tasks/task2_attention_analysis/artifacts/attention_maps/`
- `tasks/task2_attention_analysis/artifacts/rollout_maps/`

---

## Task 3 - Beam Search & Length Penalty Ablation

### Full generation + evaluation
```bash
python -m tasks.task3_beam_ablation.run_beam_ablation \
  --checkpoint_dir tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224 \
  --annotation_path src/data/raw/captions_validation.jsonl \
  --image_dir src/data/raw/val2017 \
  --num_samples 500 \
  --beam_sizes 1,3,5 \
  --length_penalties 0.8,1.0,1.2 \
  --device cpu
```

### Recompute metrics/heatmap from saved artifacts
```bash
python -m tasks.task3_beam_ablation.run_beam_ablation \
  --num_samples 500 \
  --device cpu \
  --reuse_artifacts
```

---

## Task 4 - Diversity + Style Steering

```bash
python -m tasks.task4_style_steering.run_style_steering \
  --checkpoint_dir tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224 \
  --annotation_path src/data/raw/captions_validation.jsonl \
  --image_dir src/data/raw/val2017 \
  --style_annotation_path src/data/processed/subset_10k.jsonl \
  --style_image_dir src/data/raw/train2017 \
  --num_images 200 \
  --num_steer_images 80 \
  --num_captions_per_image 5 \
  --top_p 0.9 \
  --device cpu
```

---

## Task 5 - Toxicity/Bias Audit + Mitigation

```bash
python -m tasks.task5_fairness_safety.run_fairness_audit \
  --checkpoint_dir tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224 \
  --annotation_path src/data/raw/captions_validation.jsonl \
  --image_dir src/data/raw/val2017 \
  --num_images 1000 \
  --num_beams 3 \
  --max_new_tokens 20 \
  --device cpu
```

---

## Final Outputs Checklist

- Task 1 benchmark: `tasks/task1_blip_optimization/results/`
- Task 2 reports + figures: `tasks/task2_attention_analysis/results/`
- Task 3 ablation metrics + heatmap: `tasks/task3_beam_ablation/results/`
- Task 4 steering summary + chart: `tasks/task4_style_steering/results/`
- Task 5 fairness report + audit chart: `tasks/task5_fairness_safety/results/`

