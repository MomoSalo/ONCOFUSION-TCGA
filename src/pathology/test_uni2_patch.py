from pathlib import Path

import numpy as np
import torch
import timm
import openslide

from PIL import Image

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from process_wsi_streaming import (
    create_tissue_mask,
    get_base_mpp,
    find_candidate_coordinates,
    PATCH_SIZE,
    TARGET_MPP,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PILOT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pathology"
    / "pilot"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Find one WSI
# ============================================================

slides = list(
    PILOT_DIR.glob("*.svs")
)

if len(slides) == 0:
    raise RuntimeError("No SVS found in pilot directory.")

slide_path = slides[0]

print("\nUsing slide:")
print(slide_path.name)


# ============================================================
# Open WSI
# ============================================================

slide = openslide.OpenSlide(
    str(slide_path)
)

print(
    "\nSlide dimensions:",
    slide.dimensions
)


# ============================================================
# Determine physical resolution
# ============================================================

base_mpp = get_base_mpp(
    slide
)

print(
    "Base MPP:",
    base_mpp
)

read_size = int(
    round(
        PATCH_SIZE
        * TARGET_MPP
        / base_mpp
    )
)

print(
    "Level-0 read size:",
    read_size
)


# ============================================================
# Tissue detection
# ============================================================

thumbnail, mask = create_tissue_mask(
    slide
)

print(
    "Tissue fraction:",
    f"{mask.mean():.2%}"
)


# ============================================================
# Find usable patches
# ============================================================

coordinates = find_candidate_coordinates(
    slide,
    mask,
    read_size
)

print(
    "Candidate patches:",
    len(coordinates)
)

if len(coordinates) == 0:
    slide.close()
    raise RuntimeError(
        "No tissue patch found."
    )


# ============================================================
# Pick one tissue patch
# ============================================================

coord = coordinates[0]

# Usually coordinates are (x, y)
x, y = coord[:2]

print(
    "\nSelected coordinate:",
    x,
    y
)


# ============================================================
# Read patch from WSI
# ============================================================

patch = slide.read_region(
    (int(x), int(y)),
    0,
    (read_size, read_size)
).convert("RGB")

slide.close()


# ============================================================
# Convert to our standard 256 x 256 physical patch
# ============================================================

patch = patch.resize(
    (PATCH_SIZE, PATCH_SIZE),
    Image.Resampling.BILINEAR
)

print(
    "Extracted patch size:",
    patch.size
)


# Optional: save the patch so we can inspect it
output_path = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_test_patch.png"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

patch.save(
    output_path
)

print(
    "\nPatch saved to:"
)

print(
    output_path
)


# ============================================================
# UNI2-h
# ============================================================

print("\nLoading UNI2-h...")

timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}

model = timm.create_model(
    "hf-hub:MahmoodLab/UNI2-h",
    pretrained=True,
    **timm_kwargs
)

model = model.to(device)
model.eval()


# ============================================================
# UNI2 preprocessing
# ============================================================

transform = create_transform(
    **resolve_data_config(
        model.pretrained_cfg,
        model=model
    )
)

x = transform(
    patch
)

print(
    "\nAfter UNI2 preprocessing:",
    x.shape
)

x = (
    x
    .unsqueeze(0)
    .to(device)
)


# ============================================================
# UNI2 inference
# ============================================================

with torch.inference_mode():

    embedding = model(
        x
    )


# ============================================================
# Results
# ============================================================

print(
    "\nUNI2-h embedding shape:",
    embedding.shape
)

print(
    "Embedding dtype:",
    embedding.dtype
)

print(
    "Embedding norm:",
    embedding.norm().item()
)

print(
    "\nFirst 10 values:"
)

print(
    embedding[
        0,
        :10
    ]
    .cpu()
    .numpy()
)


assert embedding.shape == (
    1,
    1536
)

print(
    "\nSUCCESS"
)

print(
    "Real TCGA pathology patch"
    " -> UNI2-h"
    " -> 1536-dimensional embedding"
)