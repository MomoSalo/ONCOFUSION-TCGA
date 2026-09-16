from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader
import openslide

from extract_patches import (
    create_tissue_mask,
    get_base_mpp,
    find_candidate_patches,
    sample_coordinates,
    extract_patches,
    PATCH_SIZE,
    TARGET_MPP
)

from extract_features import (
    PatchDataset,
    build_encoder,
    extract_embeddings,
    BATCH_SIZE,
    NUM_WORKERS
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

PATCH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pathology_patches"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pathology_features"
)


# ============================================================
# Patch extraction
# ============================================================

def process_slide_patches(slide_path):

    slide_name = slide_path.stem

    output_dir = (
        PATCH_ROOT
        / slide_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_patches = list(
        output_dir.glob("patch_*.png")
    )

    if existing_patches:

        print(
            f"Patches already exist: "
            f"{len(existing_patches)}"
        )

        return output_dir

    slide = openslide.OpenSlide(
        str(slide_path)
    )

    try:

        base_mpp = get_base_mpp(
            slide
        )

        read_size = int(
            round(
                PATCH_SIZE
                * TARGET_MPP
                / base_mpp
            )
        )

        print(
            "MPP:",
            base_mpp
        )

        print(
            "Read size:",
            read_size
        )

        # ----------------------------------------
        # Tissue mask
        # ----------------------------------------

        thumbnail, mask = (
            create_tissue_mask(
                slide
            )
        )

        thumbnail.save(
            output_dir
            / "thumbnail.png"
        )

        Image.fromarray(
            mask.astype(
                np.uint8
            ) * 255
        ).save(
            output_dir
            / "tissue_mask.png"
        )

        # ----------------------------------------
        # Candidate patches
        # ----------------------------------------

        candidates = (
            find_candidate_patches(
                slide,
                mask,
                read_size
            )
        )

        selected = (
            sample_coordinates(
                candidates
            )
        )

        print(
            "Selected patches:",
            len(selected)
        )

        # ----------------------------------------
        # Extract images
        # ----------------------------------------

        metadata = extract_patches(
            slide,
            selected,
            read_size,
            output_dir
        )

        metadata.to_csv(
            output_dir
            / "patch_metadata.csv",
            index=False
        )

    finally:

        slide.close()

    return output_dir


# ============================================================
# Feature extraction
# ============================================================

def process_slide_features(
    patch_dir,
    model,
    transform,
    device
):

    slide_name = (
        patch_dir.name
    )

    output_dir = (
        FEATURE_ROOT
        / slide_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_path = (
        output_dir
        / "features.npy"
    )

    if feature_path.exists():

        features = np.load(
            feature_path
        )

        print(
            "Features already exist:",
            features.shape
        )

        return

    dataset = PatchDataset(
        patch_dir,
        transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    features, filenames = (
        extract_embeddings(
            model,
            dataloader,
            device
        )
    )

    np.save(
        feature_path,
        features
    )

    # ----------------------------------------
    # Keep patch → embedding mapping
    # ----------------------------------------

    index = pd.DataFrame({
        "embedding_row":
            np.arange(
                len(filenames)
            ),

        "patch_file":
            filenames
    })

    metadata_path = (
        patch_dir
        / "patch_metadata.csv"
    )

    if metadata_path.exists():

        metadata = pd.read_csv(
            metadata_path
        )

        index = index.merge(
            metadata,
            left_on="patch_file",
            right_on="file_name",
            how="left"
        )

    index.to_csv(
        output_dir
        / "feature_index.csv",
        index=False
    )

    print(
        "Feature matrix:",
        features.shape
    )


# ============================================================
# Main
# ============================================================

def main():

    PATCH_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    slides = sorted(
        PILOT_DIR.glob("*.svs")
    )

    print(
        "Pilot slides:",
        len(slides)
    )

    if not slides:

        raise RuntimeError(
            "No pilot .svs files found."
        )

    # ========================================================
    # ResNet is loaded only once
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    model, transform = (
        build_encoder(
            device
        )
    )

    # ========================================================
    # Process slides
    # ========================================================

    for i, slide_path in enumerate(
        slides,
        start=1
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"SLIDE {i}/{len(slides)}"
        )

        print(
            slide_path.name
        )

        print(
            "=" * 70
        )

        # ----------------------------------------
        # WSI -> patches
        # ----------------------------------------

        patch_dir = (
            process_slide_patches(
                slide_path
            )
        )

        # ----------------------------------------
        # patches -> ResNet embeddings
        # ----------------------------------------

        process_slide_features(
            patch_dir,
            model,
            transform,
            device
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PILOT PROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()