<div align="center">

# Image Captioning with BLIP, ViT‑GPT2 & GIT

[![Project Documentation](https://img.shields.io/badge/DOCUMENTATION-blue?style=for-the-badge&logo=github)](https://pchandragrid.github.io/ML_Report/index.html)
[![Hugging Face Space](https://img.shields.io/badge/HUGGING_FACE-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/pchandragrid/image_captioning)

**End‑to‑end project to generate natural language descriptions of images, compare architectures, and deploy a public demo.**

</div>

---

## 📖 Overview

This project focuses on the **Image Captioning** task—generating natural language descriptions for given images (e.g., *"a brown dog running with a tennis ball in the grass"*). We leverage and fine-tune state-of-the-art transformer-based vision-language models to achieve high-quality captions.

The primary objectives are:
- **Performance**: Improve **CIDEr** score by **10%+** over baseline models.
- **Comparison**: Benchmarking **BLIP**, **ViT‑GPT2**, and **GIT** architectures.
- **Analysis**: Studying the effects of image resolution, caption length, and decoding parameters.
- **Deployment**: Providing a user-friendly **Streamlit web UI** hosted on Hugging Face Spaces.

> **Key Takeaway**: Generate natural language descriptions of images, optimize CIDEr metrics, and ensure accessibility via a simple web interface.

---

## 🛠️ Technologies Used

- **Deep Learning Framework**: PyTorch (`torch`) for training and tensor operations.
- **Models**: Hugging Face `transformers` (BLIP, ViT‑GPT2, GIT).
- **Data Processing**:
  - COCO captions via [`whyen-wang/coco_captions`](https://huggingface.co/datasets/whyen-wang/coco_captions).
  - Custom data loaders and processing with `Pillow` and `numpy`.
- **Evaluation**: `pycocoevalcap` for standard metrics (CIDEr).
- **Web Application**: `streamlit` for the UI, `matplotlib` for visualizations.

---

## 🔬 Methodology & Experiments

### 1. Training Strategy
Our training pipeline follows a progressive "recipe" to ensure stability and performance:
1.  **Data Preparation**: Utilizing a **10k–50k subset** of MS COCO captions.
2.  **Model Selection**: Fine-tuning from pre-trained checkpoints (e.g., `Salesforce/blip-image-captioning-base`).
3.  **Resolution Scaling**: Starting training at **224px** and progressively increasing to **384px**.
4.  **Optimization**: Implementing **gradient checkpointing** and mixed precision (especially for Mac MPS acceleration) to manage memory usage.

### 2. Experimental Analysis
The repository supports reproducible experiments across several dimensions:

-   **Architecture Comparison**:
    -   **BLIP**: Multimodal mixture of encoder-decoder.
    -   **ViT‑GPT2**: Vision encoder + GPT2 decoder (cross‑attention).
    -   **GIT**: Unified transformer for image‑to‑text.
-   **Data Filtering**:
    -   Analyzing impact of caption length (Short ≤8 words vs. Long >15 words).
    -   Filtering low-quality, repetitive, or non-alphabetic captions.
-   **Decoding Strategies**:
    -   Grid search on **Beam Sizes** (3, 5, 10).
    -   Tuning **Length Penalty** (0.8, 1.0, 1.2) and **Max Length**.

### 3. Mac Acceleration (MPS)
Special considerations were made for training on Apple Silicon:
-   Batch size: 4–8.
-   Enabled `model.gradient_checkpointing_enable()`.
-   Used `torch.autocast(device_type="mps", dtype=torch.float16)`.

---

## 📂 Project Structure

```text
image-captioning/
├── app/
│   └── streamlit_app.py             # Main Streamlit application
├── HuggingFaceUploads/
│   └── uploadtohf.py                # Utility to push models to HF Hub
├── outputs/                         # Checkpoints and results storage
├── src/
│   ├── data/                        # Dataset classes and processing
│   ├── evaluation/                  # Evaluation scripts (CIDEr, Beam Search)
│   ├── plot/                        # Visualization scripts
│   ├── training/                    # Training loops for different models
│   └── utils/                       # Helper utilities
├── tasks/
│   ├── report/                      # Interactive HTML report (GitHub Pages)
│   ├── task1_blip_optimization/     # Training, ONNX, CoreML, benchmarking
│   ├── task2_attention_analysis/    # Attention maps, rollout, IoU grounding
│   ├── task3_beam_ablation/         # Beam/length penalty ablation
│   ├── task4_style_steering/        # Diversity + CAV style steering
│   └── task5_fairness_safety/       # Toxicity/bias audit + mitigation
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
```

---

## 🧩 Project 3 Task Modules

For Project 3 continuation work, each task is implemented in a dedicated folder under `tasks/`.
All tasks are run as Python modules from the repository root.

📊 **[View Full Interactive Report](https://pchandragrid.github.io/project_3/)**

### Recommended Execution Order

1. **Task 1** — Train / export / quantize / benchmark BLIP
2. **Task 2** — Attention visualization + rollout + grounding alignment
3. **Task 3** — Beam search + length penalty ablation (quality vs latency)
4. **Task 4** — Caption diversity + CAV-style steering
5. **Task 5** — Toxicity/bias audit + mitigation + fairness report

> **Prerequisite:** Tasks 2–5 expect a BLIP checkpoint produced by Task 1, located at:
> `tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224`

### Environment Notes

- Use your main training env for model training/export if already stable.
- For CoreML conversion and most task scripts, a **Python 3.11** environment is recommended.

---

### Task 1 — BLIP Optimization (Memory-Efficient Training + CoreML Export)

**What it does:**
1. Fine-tunes BLIP on 10k COCO captions with gradient checkpointing + mixed precision (fp16 forward, fp32 loss) at 224px resolution.
2. Exports encoder and decoder to ONNX with dynamic axes.
3. Converts to CoreML (`CPU_AND_NE`) and applies 4-bit quantization.
4. Benchmarks PyTorch (full precision) vs quantized CoreML on latency + BLEU-4.

**Folder layout:**
```text
tasks/task1_blip_optimization/
  train_blip_memory_efficient.py   # Memory-efficient fine-tuning
  export_onnx.py                   # ONNX export (encoder + decoder)
  convert_coreml.py                # CoreML conversion + 4-bit quantization
  benchmark.py                     # Latency/quality benchmark
  checkpoints/                     # Trained BLIP checkpoints
  exports/onnx/                    # ONNX artifacts
  exports/coreml/                  # CoreML artifacts
  results/                         # Benchmark outputs
```

**Step-by-step:**

```bash
# 1) Train
python -m tasks.task1_blip_optimization.train_blip_memory_efficient \
  --annotation_path src/data/processed/subset_10k.jsonl \
  --image_dir src/data/raw/train2017 \
  --batch_size 4 \
  --image_size 224

# 2) Export ONNX
python -m tasks.task1_blip_optimization.export_onnx

# 3) Convert to CoreML + quantize
python -m tasks.task1_blip_optimization.convert_coreml \
  --compute_units CPU_AND_NE \
  --conversion_mode torch

# 4) Benchmark
python -m tasks.task1_blip_optimization.benchmark \
  --compute_units CPU_AND_NE \
  --num_samples 100
```

**Key outputs:**
- Checkpoint: `tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224/`
- ONNX exports: `tasks/task1_blip_optimization/exports/onnx/`
- CoreML exports: `tasks/task1_blip_optimization/exports/coreml/`
- Benchmark: `tasks/task1_blip_optimization/results/coreml_benchmark.json` / `.md`

> **Note:** If ONNX→CoreML conversion fails locally, use `--conversion_mode torch` which handles conversion directly from model wrappers. CoreML conversion works best in Python 3.11/3.12.

---

### Task 2 — Attention Visualization and Cross-Attention Rollout

**What it does:**
1. Generates captions and extracts decoder cross-attention per decoding step.
2. Builds per-step attention maps and cross-layer rollout maps.
3. Produces 2×5 visualization grids (original image + token-step heatmaps).
4. Computes IoU-based alignment against COCO object boxes.
5. Summarizes `caption_length → mean_alignment_iou`.

**Folder layout:**
```text
tasks/task2_attention_analysis/
  run_attention_analysis.py    # End-to-end attention analysis runner
  attention_utils.py           # Rollout, heatmaps, IoU utilities
  artifacts/attention_maps/    # Per-step per-layer maps (.npz)
  artifacts/rollout_maps/      # Per-step rollout maps (.npy)
  results/figures/             # Rendered attention grids
  results/reports/             # CSV + markdown summaries
```

**Run:**
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

**Key outputs:**
- Attention grids: `tasks/task2_attention_analysis/results/figures/attention_grid_*.png` (100 grids)
- Summary: `tasks/task2_attention_analysis/results/reports/attention_summary.md`
- Alignment CSV: `tasks/task2_attention_analysis/results/reports/alignment_details.csv`
- Raw maps: `artifacts/attention_maps/` (.npz) and `artifacts/rollout_maps/` (.npy)

> **Note:** If `instances_val2017.json` is missing, visualizations still generate, but IoU alignment values will be weak/zero.

---

### Task 3 — Beam Search and Length Penalty Ablation

**What it does:**

Runs **9 decoding configurations** (`beam_size ∈ {1, 3, 5}` × `length_penalty ∈ {0.8, 1.0, 1.2}`). For each:
- Generates captions and computes BLEU-4, METEOR, CIDEr, ROUGE-L
- Records mean caption length, mean latency, and p95 latency
- Generates a CIDEr heatmap and quality-vs-speed summary

**Run:**
```bash
# Full run (generate + evaluate)
python -m tasks.task3_beam_ablation.run_beam_ablation \
  --checkpoint_dir tasks/task1_blip_optimization/checkpoints/blip_gc_mp_224 \
  --annotation_path src/data/raw/captions_validation.jsonl \
  --image_dir src/data/raw/val2017 \
  --num_samples 500 \
  --beam_sizes 1,3,5 \
  --length_penalties 0.8,1.0,1.2 \
  --device cpu

# Recompute metrics/heatmap from saved artifacts
python -m tasks.task3_beam_ablation.run_beam_ablation \
  --num_samples 500 --device cpu --reuse_artifacts
```

**Key outputs:**
- Metrics CSV: `tasks/task3_beam_ablation/results/reports/beam_length_ablation_metrics.csv`
- Summary: `tasks/task3_beam_ablation/results/reports/ablation_summary.json` / `.md`
- CIDEr heatmap: `tasks/task3_beam_ablation/results/figures/cider_heatmap.png`
- Per-config captions: `tasks/task3_beam_ablation/artifacts/captions/`

> **Note:** `length_penalty` is only applied when `beam_size > 1` (HuggingFace behavior). If official METEOR/CIDEr scorer backends are unavailable, the code uses deterministic proxy fallbacks.

---

### Task 4 — Caption Diversity and Concept Activation Vector Style Steering

**What it does:**
1. Generates multiple captions per image using nucleus sampling (`top_p=0.9`).
2. Computes caption diversity: `unique ngrams / total ngrams`.
3. Computes pre-beam hidden-state diversity (top-token embedding spread).
4. Learns concept directions from hidden-state means: `short → long` and `short → detailed`.
5. Applies steering during decoding: `h_steered = h + λ * direction`.
6. Measures style shift by caption length before vs after steering.

**Run:**
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

**Key outputs:**
- Diversity captions: `tasks/task4_style_steering/artifacts/captions/diversity_captions.jsonl`
- Steering vectors: `tasks/task4_style_steering/artifacts/steering/dir_short_to_*.npy`
- Steering table: `tasks/task4_style_steering/results/reports/steering_outputs.csv`
- Summary: `tasks/task4_style_steering/results/reports/style_steering_summary.json` / `.md`
- Figure: `tasks/task4_style_steering/results/figures/style_length_shift.png`

> **Note:** Direction sign is auto-calibrated on a small subset so "long" steering increases output length. This task focuses on diversity and controllability.

---

### Task 5 — Toxicity and Bias Detection with Mitigation

**What it does:**
1. Generates baseline captions on 1000 validation images.
2. Scores toxicity using HuggingFace `unitary/toxic-bert` (with lexicon-based fallback).
3. Audits bias with demographic + stereotype rules (`demographic_group → stereotype_frequency`), covering gender, age, and race-oriented keyword groups.
4. Applies mitigation during beam search by penalizing problematic token IDs via logits processor.
5. Trains a secondary bias detector (TF-IDF + Logistic Regression; rule fallback if only one class).
6. Compares before vs after: toxicity rate, stereotype rate, BLEU-4, CIDEr.
7. Produces a fairness report with concrete problematic examples.

**Run:**
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

**Key outputs:**
- Baseline captions: `tasks/task5_fairness_safety/artifacts/captions/baseline_captions.jsonl`
- Mitigated captions: `tasks/task5_fairness_safety/artifacts/captions/mitigated_captions.jsonl`
- Bias model: `tasks/task5_fairness_safety/artifacts/models/`
- Bias audit CSV: `tasks/task5_fairness_safety/results/reports/bias_audit.csv`
- Fairness report: `tasks/task5_fairness_safety/results/reports/fairness_report.md` / `.json`
- Before/after chart: `tasks/task5_fairness_safety/results/figures/before_after_metrics.png`

> **Note:** `fairness_report.json` records which backend was used (`hf_toxic_bert` or `lexicon_fallback`) and classifier mode (`logistic_regression` or `rule_fallback`).

---

### Final Outputs Checklist

| Task | Key Output Location |
|------|-------------------|
| Task 1 | `tasks/task1_blip_optimization/results/` |
| Task 2 | `tasks/task2_attention_analysis/results/` |
| Task 3 | `tasks/task3_beam_ablation/results/` |
| Task 4 | `tasks/task4_style_steering/results/` |
| Task 5 | `tasks/task5_fairness_safety/results/` |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/pchandragrid/ML-Image-Captioning.git
    cd ML-Image-Captioning
    ```

2.  **Set up the environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

### Running the App Locally

To launch the main interface with model comparison:

```bash
streamlit run app.py
```

Access the app at `http://localhost:8501`. You can:
1.  Upload an image.
2.  Select models (BLIP, ViT-GPT2, GIT) from the sidebar.
3.  Adjust generation parameters (Beam Size, etc.).
4.  Generate and compare captions.

> **Note**: Models will be downloaded from Hugging Face on the first run.

---

## 📦 Model Management

To keep the repository lightweight, fine-tuned models are hosted on Hugging Face Hub and loaded dynamically:

-   **BLIP**: `pchandragrid/blip-caption-model`
-   **ViT‑GPT2**: `pchandragrid/vit-gpt2-caption-model`
-   **GIT**: `pchandragrid/git-caption-model`

The `app.py` script automatically checks for local models in `outputs/` before falling back to these remote repositories.

---

## ☁️ Deployment

This project is designed for easy deployment to **Hugging Face Spaces**.

1.  **Create a Space**: Select "Streamlit" as the SDK.
2.  **Push Code**:
    ```bash
    git remote add space https://huggingface.co/spaces/pchandragrid/image_captioning
    git push space main
    ```
3.  **Configuration**: The app defaults to public model paths. For private models, set `HF_TOKEN` in the Space's secrets.

---

## 🏆 Key Achievements

-   Successfully built an end-to-end **image captioning system**.
-   Fine-tuned and compared **BLIP**, **ViT‑GPT2**, and **GIT** on COCO data.
-   Achieved **>10% improvement in CIDEr scores** over baselines.
-   Optimized training pipelines for **Mac MPS** hardware.
-   Deployed a fully functional **public web demo** accessible to anyone.

---
