# Task 3 Beam Search and Length Penalty Ablation

- Samples evaluated: 500
- Checkpoint: `tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224`
- Config grid size: 9

## Best by CIDEr

- beam_size=3, length_penalty=0.8, CIDEr=2.8706, latency=347.92 ms

## Best Quality/Speed Trade-off

- beam_size=1, length_penalty=1.2, CIDEr=2.7607, latency=133.68 ms

## Notes

- Higher beam size generally improves quality but increases latency.
- Length penalty affects caption verbosity and may shift CIDEr/ROUGE.
