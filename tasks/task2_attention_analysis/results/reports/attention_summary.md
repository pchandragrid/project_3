# Task 2 Attention Analysis Summary

- Images analyzed: 100
- Checkpoint: `tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224`
- Caption generation max_new_tokens: 20

## Caption Length -> Mean Alignment IoU

| Caption Length | Mean Alignment IoU | Word Matches |
|---:|---:|---:|
| 7 | 0.0000 | 1 |
| 8 | 0.1721 | 16 |
| 9 | 0.0919 | 29 |
| 10 | 0.1391 | 30 |
| 11 | 0.1610 | 11 |
| 12 | 0.0031 | 8 |
| 13 | 0.0000 | 1 |
| 14 | 0.0000 | 3 |
| 15 | 0.0000 | 1 |

## Interpretation Notes

- Higher IoU indicates better grounding of words to object regions.
- Very long captions may show lower mean IoU due to attention diffusion.
- Check low-IoU words as potential hallucination candidates.
