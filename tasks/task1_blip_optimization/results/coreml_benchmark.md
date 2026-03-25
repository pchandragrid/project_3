# Task 1 CoreML Benchmark

Evaluated on 100 samples.

| Model | Mean Latency (ms) | Std (ms) | BLEU-4 |
|---|---:|---:|---:|
| PyTorch (full precision) | 258.67 | 347.66 | 0.1814 |
| CoreML (4-bit quantized) | 178.71 | 11.05 | 0.1613 |
