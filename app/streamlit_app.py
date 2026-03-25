import os
import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    GitProcessor,
    GitForCausalLM
)

from PIL import Image


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = _get_device()
_TORCH_DTYPE = torch.float16 if device.type in {"cuda", "mps"} else torch.float32


def _resolve_source(local_dir: str, hub_id: str) -> str:
    """
    Prefer a local directory if it exists; otherwise use a Hugging Face Hub repo id.
    """
    if local_dir and os.path.isdir(local_dir):
        return local_dir
    return hub_id


# ================================
# EXPERIMENT GRAPH FUNCTIONS
# ================================

def plot_beam_experiment():

    beam_sizes = [1,3,5,10]

    blip_scores = [0.52,0.59,0.61,0.60]
    vit_scores = [0.50,0.56,0.60,0.58]
    git_scores = [0.12,0.16,0.17,0.16]

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(beam_sizes, blip_scores, marker='o', linewidth=3, label="BLIP")
    ax.plot(beam_sizes, vit_scores, marker='o', linewidth=3, label="ViT-GPT2")
    ax.plot(beam_sizes, git_scores, marker='o', linewidth=3, label="GIT")

    ax.set_xlabel("Beam Size")
    ax.set_ylabel("CIDEr Score")
    ax.set_title("Beam Size vs Caption Quality")

    ax.legend()
    ax.grid(True)

    return fig


def plot_caption_length():

    labels = ["Short","Medium","Long"]

    blip = [0.71,0.60,0.48]
    vit = [0.65,0.59,0.42]
    git = [0.30,0.18,0.11]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(x - width, blip, width, label="BLIP")
    ax.bar(x, vit, width, label="ViT-GPT2")
    ax.bar(x + width, git, width, label="GIT")

    ax.set_xlabel("Caption Length Category")
    ax.set_ylabel("CIDEr Score")
    ax.set_title("Model Performance vs Caption Length")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    return fig


# ================================
# UI STYLE
# ================================

st.markdown("""
<style>

.main-title{
text-align:center;
font-size:42px;
font-weight:bold;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
margin-bottom:30px;
}

.caption-box{
background-color:white;
padding:20px;
border-radius:14px;
text-align:center;
font-size:18px;
min-height:120px;
display:flex;
align-items:center;
justify-content:center;
color:black;
font-weight:500;
box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.model-title{
text-align:center;
font-size:22px;
font-weight:bold;
margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)


# ================================
# LOAD MODELS
# ================================

@st.cache_resource
def load_blip():
    source = _resolve_source(
        os.getenv("BLIP_LOCAL_DIR", "saved_model_phase2"),
        os.getenv("BLIP_MODEL_ID", "pchandragrid/blip-caption-model"),
    )

    model = BlipForConditionalGeneration.from_pretrained(
        source,
        torch_dtype=_TORCH_DTYPE,
        low_cpu_mem_usage=True,
    )
    processor = BlipProcessor.from_pretrained(source)
    model.to(device)
    model.eval()
    return model, processor


@st.cache_resource
def load_vit_gpt2():
    source = _resolve_source(
        os.getenv("VITGPT2_LOCAL_DIR", "saved_vit_gpt2"),
        os.getenv("VITGPT2_MODEL_ID", "pchandragrid/vit-gpt2-caption-model"),
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        source,
        torch_dtype=_TORCH_DTYPE,
        low_cpu_mem_usage=True,
    )
    processor = ViTImageProcessor.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model.to(device)
    model.eval()
    return model, processor, tokenizer


@st.cache_resource
def load_git():
    source = _resolve_source(
        os.getenv("GIT_LOCAL_DIR", "saved_git_model"),
        os.getenv("GIT_MODEL_ID", "pchandragrid/git-caption-model"),
    )

    processor = GitProcessor.from_pretrained(source)
    model = GitForCausalLM.from_pretrained(
        source,
        torch_dtype=_TORCH_DTYPE,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, processor


# ================================
# HEADER
# ================================

st.markdown('<div class="main-title">🖼️ Image Captioning</div>', unsafe_allow_html=True)

st.markdown(
'<div class="subtitle">Compare BLIP vs ViT-GPT2 vs GIT on the same image</div>',
unsafe_allow_html=True
)


st.markdown("""
### 📌 Project Overview

This project focuses on **automatic image caption generation using transformer-based vision-language models**.

The system takes an input image and generates a natural language description of the scene.

Three architectures are evaluated:

• **BLIP (Bootstrapping Language Image Pretraining)** – multimodal transformer designed specifically for vision-language tasks  
• **ViT-GPT2** – Vision Transformer encoder combined with GPT2 text decoder  
• **GIT (Generative Image-to-Text Transformer)** – unified transformer architecture for image-to-text generation  

The goal of this project is to **compare model architectures, caption quality, and generation performance** using the COCO dataset.

---

### 🎯 Project Objective

Improve caption generation performance through **fine-tuning and decoding optimization**.

Training pipeline:

**Step 1 — Dataset Preparation**
- Use **MS COCO captions dataset**
- Train on a **10k–50k image-caption subset**

**Step 2 — Model Fine-Tuning**
- Fine-tune **BLIP or VisionEncoderDecoder models**

**Step 3 — Training Configuration**
- Train with image resolution **224–384 px**
- Train for **3 epochs**

**Step 4 — Memory Optimization**
- Use **gradient checkpointing** to reduce GPU memory usage

**Step 5 — Target Performance**
- Achieve **10%+ improvement in CIDEr score** compared to baseline models

These steps allow the system to learn stronger **image-text alignment and caption generation capability**.
""")


# ================================
# SIDEBAR
# ================================

st.sidebar.header("⚙️ Generation Settings")

st.sidebar.subheader("Models to run")
run_blip = st.sidebar.checkbox("BLIP", value=True)
run_vit = st.sidebar.checkbox("ViT-GPT2", value=False)
run_git = st.sidebar.checkbox("GIT", value=False)

num_beams = st.sidebar.slider("Beam Size",1,10,5)
max_length = st.sidebar.slider("Max Length",10,50,20)
length_penalty = st.sidebar.slider("Length Penalty",0.5,2.0,1.0,step=0.1)


uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])


# ================================
# IMAGE DISPLAY
# ================================

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown(
    """
    <div style="text-align:center;font-size:22px;font-weight:bold;margin-bottom:10px;">
    Uploaded Image
    </div>
    """,
    unsafe_allow_html=True
    )

    st.image(image, use_container_width=True)

    if st.button("Generate Captions"):

        with st.spinner("Running models..."):

            if not any([run_blip, run_vit, run_git]):
                st.warning("Select at least one model in the sidebar.")
                st.stop()

            results = []
            blip_inputs = None

            if run_blip:
                blip_model, blip_processor = load_blip()
                start = time.time()
                blip_inputs = blip_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    blip_ids = blip_model.generate(
                        **blip_inputs,
                        num_beams=num_beams,
                        max_length=max_length,
                        length_penalty=length_penalty,
                    )
                blip_caption = blip_processor.decode(blip_ids[0], skip_special_tokens=True)
                results.append(("BLIP", blip_caption, time.time() - start))

            if run_vit:
                vit_model, vit_processor, vit_tokenizer = load_vit_gpt2()
                start = time.time()
                pixel_values = vit_processor(images=image, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    vit_ids = vit_model.generate(
                        pixel_values=pixel_values,
                        num_beams=num_beams,
                        max_length=max_length,
                    )
                vit_caption = vit_tokenizer.decode(vit_ids[0], skip_special_tokens=True)
                results.append(("ViT-GPT2", vit_caption, time.time() - start))

            if run_git:
                git_model, git_processor = load_git()
                start = time.time()
                git_inputs = git_processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    git_ids = git_model.generate(
                        **git_inputs,
                        num_beams=num_beams,
                        max_length=max_length,
                    )
                git_caption = git_processor.batch_decode(git_ids, skip_special_tokens=True)[0]
                results.append(("GIT", git_caption, time.time() - start))


        st.divider()

        st.subheader("Model Comparison")

        st.markdown("""
Each model generates a caption describing the uploaded image.

This comparison highlights differences in:

• caption quality  
• inference speed  
• architectural design
""")

        cols = st.columns(len(results))
        for col, (name, caption, seconds) in zip(cols, results):
            with col:
                st.markdown(f'<div class="model-title">{name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="caption-box">{caption}</div>', unsafe_allow_html=True)
                st.caption(f"Inference: {seconds:.2f}s")


        st.divider()


        # ================================
        # ATTENTION HEATMAP
        # ================================

        if run_blip and blip_inputs is not None:
            blip_model, _ = load_blip()
            with torch.no_grad():
                vision_outputs = blip_model.vision_model(
                    blip_inputs["pixel_values"],
                    output_attentions=True,
                    return_dict=True,
                )

            attentions = vision_outputs.attentions[-1]

            attn = attentions[0].mean(0)
            cls_attn = attn[0, 1:]

            attn_map = cls_attn.cpu().numpy()
            attn_map = attn_map / attn_map.max()

            size = int(np.sqrt(len(attn_map)))

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.imshow(attn_map.reshape(size, size), cmap="viridis")
            ax.set_title("BLIP Vision Attention")
            ax.axis("off")

            st.pyplot(fig, use_container_width=True)

            st.markdown("""
### 🔍 Attention Visualization

The attention heatmap highlights **which regions of the image the model focused on while generating the caption**.

Brighter regions indicate higher importance for the caption generation process.
""")


# ================================
# ARCHITECTURE COMPARISON TABLE
# ================================

st.divider()
st.header("📊 Model Architecture Comparison")

data = {
"Model":["BLIP","ViT-GPT2","GIT"],
"Architecture":[
"Vision Transformer + Text Decoder",
"ViT Encoder + GPT2 Decoder",
"Unified Transformer"
],
"Parameters":["~224M","~210M","~150M"],
"Training Time":["~1h 34m / epoch","~1h 20m / epoch","~11 min / epoch"],
"CIDEr Score":["0.61","0.60","0.17"]
}

df = pd.DataFrame(data)

st.table(df)


# ================================
# EXPERIMENT GRAPHS
# ================================

st.divider()
st.header("📊 Experiment Analysis")

st.subheader("Beam Size vs Caption Quality")

fig1 = plot_beam_experiment()
st.pyplot(fig1, use_container_width=True)

st.markdown("""
Beam search controls how many candidate captions are explored during generation.
Increasing beam size improves caption quality initially but eventually leads to diminishing returns.
""")


st.divider()

st.subheader("Caption Length vs Model Performance")

fig2 = plot_caption_length()
st.pyplot(fig2, use_container_width=True)

st.markdown("""
Caption length impacts performance because longer captions require more detailed reasoning about the scene.
Models generally perform better on shorter captions.
""")
