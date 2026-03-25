# Task 3 - Beam Search and Length Penalty Ablation

This task is isolated in `tasks/task3_beam_ablation/`.

## What this task does

Runs 9 decoding configurations:
- `beam_size in {1, 3, 5}`
- `length_penalty in {0.8, 1.0, 1.2}`

For each configuration:
- generates captions
- computes BLEU-4, METEOR, CIDEr, ROUGE-L
- computes mean caption length
- computes mean and p95 latency

Then generates:
- CIDEr heatmap (`beam_size x length_penalty`)
- quality-vs-speed summary

## Step-by-step run

### Full run (generate + evaluate)
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

### Recompute metrics/heatmap only (reuse saved captions)
```bash
python -m tasks.task3_beam_ablation.run_beam_ablation \
  --num_samples 500 \
  --device cpu \
  --reuse_artifacts
```

## Outputs

- Metrics CSV: `tasks/task3_beam_ablation/results/reports/beam_length_ablation_metrics.csv`
- Summary MD: `tasks/task3_beam_ablation/results/reports/ablation_summary.md`
- Summary JSON: `tasks/task3_beam_ablation/results/reports/ablation_summary.json`
- Heatmap PNG: `tasks/task3_beam_ablation/results/figures/cider_heatmap.png`
- Per-config caption artifacts: `tasks/task3_beam_ablation/artifacts/captions/`

## Notes

- If official METEOR/CIDEr scorer backends are unavailable locally, the code uses deterministic proxy fallbacks (non-zero, comparable across configs).
- `length_penalty` is only applied when `beam_size > 1` (HF behavior).

