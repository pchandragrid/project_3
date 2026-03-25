import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def parse_image_id_from_filename(file_name: str) -> int:
    match = re.search(r"(\d+)", os.path.basename(file_name))
    if not match:
        return -1
    return int(match.group(1))


def normalize_map(attn_map: np.ndarray) -> np.ndarray:
    attn_map = attn_map.astype(np.float32)
    attn_map = attn_map - float(attn_map.min())
    denom = float(attn_map.max()) + 1e-8
    return attn_map / denom


def vector_to_grid(attn_vec: np.ndarray, image_size: Tuple[int, int], patch_size: int = 16) -> np.ndarray:
    h, w = image_size
    gh = max(1, h // patch_size)
    gw = max(1, w // patch_size)

    vec = np.asarray(attn_vec).reshape(-1)
    if vec.size == gh * gw + 1:
        vec = vec[1:]  # remove CLS-like token if present
    if vec.size != gh * gw:
        # Best effort fallback for slight shape mismatches.
        size = min(vec.size, gh * gw)
        clipped = np.zeros((gh * gw,), dtype=np.float32)
        clipped[:size] = vec[:size]
        vec = clipped
    grid = vec.reshape(gh, gw)

    grid_img = Image.fromarray((normalize_map(grid) * 255.0).astype(np.uint8))
    grid_img = grid_img.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(grid_img).astype(np.float32) / 255.0


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = np.asarray(image).astype(np.float32) / 255.0
    hm = normalize_map(heatmap)
    # Lightweight "jet-like" colormap without matplotlib dependency.
    colored = np.stack(
        [
            np.clip(1.5 - np.abs(4.0 * hm - 3.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * hm - 2.0), 0.0, 1.0),
            np.clip(1.5 - np.abs(4.0 * hm - 1.0), 0.0, 1.0),
        ],
        axis=-1,
    )
    out = (1.0 - alpha) * img + alpha * colored
    return np.clip(out, 0.0, 1.0)


def plot_2x5_attention_grid(
    image: Image.Image,
    words: Sequence[str],
    token_maps: Sequence[np.ndarray],
    output_path: str,
    title: str,
) -> None:
    base = image.convert("RGB")
    cell_w, cell_h = base.size
    header_h = 56
    gap = 8
    cols, rows = 5, 2
    canvas_w = cols * cell_w + (cols + 1) * gap
    canvas_h = header_h + rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(22, 22, 22))
    draw = ImageDraw.Draw(canvas)

    short_title = title if len(title) <= 160 else title[:157] + "..."
    draw.text((gap, 8), short_title, fill=(235, 235, 235))

    # Panel 0: original image
    x0 = gap
    y0 = header_h + gap
    canvas.paste(base, (x0, y0))
    draw.text((x0 + 6, y0 + 6), "Original", fill=(255, 255, 255))

    usable = min(len(token_maps), 9)
    for i in range(usable):
        panel = i + 1
        row = panel // cols
        col = panel % cols
        x = gap + col * (cell_w + gap)
        y = header_h + gap + row * (cell_h + gap)

        overlay = (overlay_heatmap(base, token_maps[i]) * 255.0).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)
        canvas.paste(overlay_img, (x, y))
        word = words[i] if i < len(words) else f"step_{i+1}"
        draw.text((x + 6, y + 6), f"Step {i+1}: {word}", fill=(255, 255, 255))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def attention_rollout(attn_by_layer: Sequence[np.ndarray]) -> np.ndarray:
    """
    Recursive cross-attention rollout over decoder layers.
    Each layer attention is expected as a vector over image tokens.
    """
    if not attn_by_layer:
        raise ValueError("No layer attention available for rollout.")

    rollout = normalize_map(np.asarray(attn_by_layer[0]).reshape(-1))
    for layer_vec in attn_by_layer[1:]:
        layer_vec = normalize_map(np.asarray(layer_vec).reshape(-1))
        rollout = rollout * layer_vec
        rollout = normalize_map(rollout)
    return rollout


def binarize_heatmap(heatmap: np.ndarray, quantile: float = 0.85) -> np.ndarray:
    threshold = float(np.quantile(heatmap, quantile))
    return (heatmap >= threshold).astype(np.uint8)


def boxes_to_mask(boxes_xywh: Sequence[Sequence[float]], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes_xywh:
        x, y, bw, bh = box
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(w, int(round(x + bw)))
        y1 = min(h, int(round(y + bh)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    return mask


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


@dataclass
class CocoBoxIndex:
    image_to_annotations: Dict[int, List[Dict]]
    category_id_to_name: Dict[int, str]
    category_name_to_id: Dict[str, int]


def build_coco_box_index(instances_json_path: Optional[str]) -> Optional[CocoBoxIndex]:
    if not instances_json_path or not os.path.exists(instances_json_path):
        return None

    with open(instances_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    image_to_annotations: Dict[int, List[Dict]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        image_to_annotations[int(ann["image_id"])].append(ann)

    category_id_to_name = {}
    category_name_to_id = {}
    for cat in payload.get("categories", []):
        cat_id = int(cat["id"])
        name = str(cat["name"]).lower()
        category_id_to_name[cat_id] = name
        category_name_to_id[name] = cat_id

    return CocoBoxIndex(
        image_to_annotations=dict(image_to_annotations),
        category_id_to_name=category_id_to_name,
        category_name_to_id=category_name_to_id,
    )


def simple_word_variants(word: str) -> List[str]:
    w = word.lower().strip(".,!?;:'\"()[]{}")
    variants = {w}
    if w.endswith("s") and len(w) > 3:
        variants.add(w[:-1])
    else:
        variants.add(w + "s")
    return [v for v in variants if v]


def boxes_for_word(
    coco_index: Optional[CocoBoxIndex],
    image_id: int,
    word: str,
) -> List[List[float]]:
    if coco_index is None or image_id < 0:
        return []

    candidates = set(simple_word_variants(word))
    anns = coco_index.image_to_annotations.get(image_id, [])
    out = []
    for ann in anns:
        cat_name = coco_index.category_id_to_name.get(int(ann["category_id"]), "")
        if cat_name in candidates:
            out.append(list(ann["bbox"]))
    return out

