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
├── apps/
│   └── app.py                     # Main Streamlit application
├── docs/
│   └── index.html                 # Project report and visualization
├── HuggingFaceUploads/
│   └── uploadtohf.py              # Utility to push models to HF Hub
├── outputs/                       # Checkpoints and results storage
├── src/
│   ├── data/                      # Dataset classes and processing
│   ├── evaluation/                # Evaluation scripts (CIDEr, Beam Search)
│   ├── plot/                      # Visualization scripts
│   ├── training/                  # Training loops for different models
│   └── utils/                     # Helper utilities
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🧩 Project 3 Task Modules

For Project 3 continuation work, each task is implemented in a dedicated folder.  
Task 1 is available in:

- `tasks/task1_blip_optimization/` (training, ONNX export, CoreML conversion, quantized benchmarking)
- `tasks/task1_blip_optimization/checkpoints/` (fine-tuned checkpoints)
- `tasks/task1_blip_optimization/exports/` (ONNX and CoreML artifacts)
- `tasks/task1_blip_optimization/results/` (benchmark reports)
- `tasks/task2_attention_analysis/` (cross-attention visualization, rollout, alignment analysis)
- `tasks/task3_beam_ablation/` (beam size and length penalty ablation with quality/speed trade-off)
- `tasks/task4_style_steering/` (caption diversity analysis and concept activation vector style steering)
- `tasks/task5_fairness_safety/` (toxicity/bias detection, mitigation, and fairness report)

**For a complete step-by-step execution guide for all tasks, see:**
👉 [`tasks/README.md`](tasks/README.md)

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
