# Task 5 - Toxicity and Bias Detection with Mitigation

This task is isolated in `tasks/task5_fairness_safety/`.

## What this task does

1. Generates baseline captions (default 1000 validation images).
2. Scores toxicity:
   - primary: HuggingFace `unitary/toxic-bert`
   - fallback: lexicon-based toxicity scorer (auto when network/model unavailable)
3. Audits bias with demographic + stereotype rules:
   - outputs `demographic_group -> stereotype_frequency`
   - includes gender, age, and race-oriented demographic keyword groups
4. Applies mitigation during beam search:
   - penalizes problematic token IDs through logits processor
5. Trains secondary bias detector:
   - TF-IDF + Logistic Regression
   - fallback to rule mode if only one class appears
6. Compares before vs after:
   - toxicity rate, stereotype rate, BLEU-4, CIDEr
7. Produces fairness report with concrete problematic examples.

## Step-by-step run

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

## Outputs

- Baseline captions: `tasks/task5_fairness_safety/artifacts/captions/baseline_captions.jsonl`
- Mitigated captions: `tasks/task5_fairness_safety/artifacts/captions/mitigated_captions.jsonl`
- Bias model/vectorizer: `tasks/task5_fairness_safety/artifacts/models/`
- Bias audit CSV: `tasks/task5_fairness_safety/results/reports/bias_audit.csv`
- Fairness report:
  - `tasks/task5_fairness_safety/results/reports/fairness_report.md`
  - `tasks/task5_fairness_safety/results/reports/fairness_report.json`
- Before/after chart: `tasks/task5_fairness_safety/results/figures/before_after_metrics.png`

## Notes

- `fairness_report.json` records which backend was used:
  - `toxicity_backend`: `hf_toxic_bert` or `lexicon_fallback`
  - `classifier_mode`: `logistic_regression` or `rule_fallback`
- CIDEr uses official scorer when available, otherwise proxy fallback from shared utilities.
- Bias audit table is computed on captions that explicitly mention demographic/person groups.

