# Task 2 - Attention Visualization and Cross-Attention Rollout

This task is isolated in `tasks/task2_attention_analysis/` (outside `src`).

## What this task does

1. Generates captions and extracts decoder cross-attention per decoding step.
2. Builds per-step attention maps and cross-layer rollout maps.
3. Produces 2x5 visualization grids (original image + token-step heatmaps).
4. Computes IoU-based alignment against COCO object boxes.
5. Summarizes `caption_length -> mean_alignment_iou`.

## Folder layout

- `run_attention_analysis.py` - end-to-end attention analysis runner
- `attention_utils.py` - rollout, heatmaps, IoU utilities
- `artifacts/attention_maps/` - per-step per-layer maps (`.npz`)
- `artifacts/rollout_maps/` - per-step rollout maps (`.npy`)
- `results/figures/` - rendered attention grids
- `results/reports/` - CSV + markdown summaries

## Step-by-step run

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

## Important output files

- Detail CSV: `tasks/task2_attention_analysis/results/reports/alignment_details.csv`
- Length summary CSV: `tasks/task2_attention_analysis/results/reports/caption_length_alignment_summary.csv`
- Summary markdown: `tasks/task2_attention_analysis/results/reports/attention_summary.md`
- Per-step maps:
  - `tasks/task2_attention_analysis/artifacts/attention_maps/img_XXXX_step_YY_layers.npz`
  - `tasks/task2_attention_analysis/artifacts/rollout_maps/img_XXXX_step_YY_rollout.npy`

## Notes

- Default `grid_steps` is `5` to match the required "image + 5 caption steps" analysis view.
- If `instances_val2017.json` is missing, visualizations still generate, but IoU alignment values will be weak/zero.

