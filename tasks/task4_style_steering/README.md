# Task 4 - Caption Diversity and Concept Activation Vector Style Steering

This task is isolated in `tasks/task4_style_steering/`.

## What this task does

1. Generates multiple captions per image using nucleus sampling (`top_p=0.9`).
2. Computes caption diversity:
   - `unique ngrams / total ngrams`
3. Computes pre-beam hidden-state diversity (top-token embedding spread).
4. Learns concept directions from hidden-state means:
   - `short -> long`
   - `short -> detailed`
5. Applies steering during decoding:
   - `h_steered = h + λ * direction`
6. Measures style shift by caption length/word count before vs after steering.

## Step-by-step run

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

## Outputs

- Diversity artifacts: `tasks/task4_style_steering/artifacts/captions/diversity_captions.jsonl`
- Steering vectors: `tasks/task4_style_steering/artifacts/steering/dir_short_to_*.npy`
- Steering table: `tasks/task4_style_steering/results/reports/steering_outputs.csv`
- Summary markdown: `tasks/task4_style_steering/results/reports/style_steering_summary.md`
- Summary json: `tasks/task4_style_steering/results/reports/style_steering_summary.json`
- Figure: `tasks/task4_style_steering/results/figures/style_length_shift.png`

## Notes

- Direction sign is auto-calibrated on a small subset so "long" steering is more likely to increase output length.
- This task focuses on diversity and controllability; METEOR/CIDEr are reported in Task 3/Task 5 pipelines.

