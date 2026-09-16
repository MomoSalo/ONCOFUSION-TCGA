from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import requests

from PIL import Image

import openslide

import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "patient_manifest_split.csv"
)

# Only ONE raw WSI will normally live here at a time
WORK_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pathology"
    / "working"
)

# Final compact representations
FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pathology_features"
)


# ============================================================
# GDC
# ============================================================

GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


# ============================================================
# Pathology parameters
# ============================================================

PATCH_SIZE = 256

# Approximate physical resolution presented to ResNet
TARGET_MPP = 0.5

# Patch must contain at least this much detected tissue
MIN_TISSUE_FRACTION = 0.60

# Maximum patches kept per patient
DEFAULT_MAX_PATCHES = 1000

THUMBNAIL_MAX_SIZE = 2000

SATURATION_THRESHOLD = 20
VALUE_THRESHOLD = 245

BATCH_SIZE = 32

RANDOM_SEED = 42


# ============================================================
# ResNet encoder
# ============================================================

def build_encoder(device):

    weights = ResNet50_Weights.DEFAULT

    model = resnet50(
        weights=weights
    )

    # Remove ImageNet classifier
    #
    # output:
    # [batch, 2048]
    model.fc = nn.Identity()

    model = model.to(device)

    model.eval()

    transform = weights.transforms()

    return model, transform


# ============================================================
# Download one WSI
# ============================================================

def download_wsi(
    file_id,
    file_name
):

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        WORK_DIR
        / file_name
    )

    # If previous run downloaded the WSI
    # but processing failed, reuse it.
    if output_path.exists():

        print(
            "\nRaw WSI already present."
        )

        return output_path

    temp_path = Path(
        str(output_path) + ".part"
    )

    url = (
        f"{GDC_DATA_URL}/{file_id}"
    )

    print(
        "\nDownloading WSI:"
    )

    print(
        file_name
    )

    with requests.get(
        url,
        stream=True,
        timeout=(30, 600)
    ) as response:

        response.raise_for_status()

        total_bytes = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        downloaded = 0

        with open(
            temp_path,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=8 * 1024 * 1024
            ):

                if not chunk:
                    continue

                f.write(chunk)

                downloaded += len(chunk)

                if total_bytes > 0:

                    percentage = (
                        downloaded
                        / total_bytes
                        * 100
                    )

                    print(
                        f"\rDownload: "
                        f"{percentage:.1f}%",
                        end=""
                    )

    print()

    temp_path.rename(
        output_path
    )

    return output_path


# ============================================================
# Get slide MPP
# ============================================================

def get_base_mpp(slide):

    value = slide.properties.get(
        openslide.PROPERTY_NAME_MPP_X
    )

    if value is None:

        print(
            "WARNING: MPP missing. "
            "Using 0.5 µm/pixel."
        )

        return TARGET_MPP

    return float(value)


# ============================================================
# Tissue mask
# ============================================================

def create_tissue_mask(slide):

    thumbnail = slide.get_thumbnail(
        (
            THUMBNAIL_MAX_SIZE,
            THUMBNAIL_MAX_SIZE
        )
    ).convert("RGB")

    hsv = np.asarray(
        thumbnail.convert("HSV")
    )

    saturation = hsv[:, :, 1]

    brightness = hsv[:, :, 2]

    mask = (
        (saturation > SATURATION_THRESHOLD)
        &
        (brightness < VALUE_THRESHOLD)
    )

    return thumbnail, mask


# ============================================================
# Tissue fraction for candidate region
# ============================================================

def tissue_fraction(
    mask,
    x,
    y,
    read_size,
    slide_width,
    slide_height
):

    mask_height, mask_width = (
        mask.shape
    )

    x0 = int(
        x
        / slide_width
        * mask_width
    )

    x1 = int(
        (x + read_size)
        / slide_width
        * mask_width
    )

    y0 = int(
        y
        / slide_height
        * mask_height
    )

    y1 = int(
        (y + read_size)
        / slide_height
        * mask_height
    )

    # Clip to mask
    x0 = max(
        0,
        min(
            x0,
            mask_width - 1
        )
    )

    y0 = max(
        0,
        min(
            y0,
            mask_height - 1
        )
    )

    x1 = max(
        x0 + 1,
        min(
            x1,
            mask_width
        )
    )

    y1 = max(
        y0 + 1,
        min(
            y1,
            mask_height
        )
    )

    region = mask[
        y0:y1,
        x0:x1
    ]

    if region.size == 0:
        return 0.0

    return float(
        region.mean()
    )


# ============================================================
# Find candidate patches
# ============================================================

def find_candidate_coordinates(
    slide,
    mask,
    read_size
):

    slide_width, slide_height = (
        slide.dimensions
    )

    coordinates = []

    # Non-overlapping grid
    for y in range(
        0,
        slide_height - read_size + 1,
        read_size
    ):

        for x in range(
            0,
            slide_width - read_size + 1,
            read_size
        ):

            fraction = tissue_fraction(
                mask,
                x,
                y,
                read_size,
                slide_width,
                slide_height
            )

            if (
                fraction
                >= MIN_TISSUE_FRACTION
            ):

                coordinates.append(
                    (
                        x,
                        y,
                        fraction
                    )
                )

    return coordinates


# ============================================================
# Select patches
# ============================================================

def select_coordinates(
    coordinates,
    max_patches,
    seed
):

    if (
        len(coordinates)
        <= max_patches
    ):

        return coordinates

    rng = random.Random(
        seed
    )

    return rng.sample(
        coordinates,
        max_patches
    )


# ============================================================
# Encode patches WITHOUT saving PNG files
# ============================================================

def encode_slide(
    slide,
    coordinates,
    read_size,
    model,
    transform,
    device
):

    all_embeddings = []

    batch_images = []

    total = len(
        coordinates
    )

    for i, (
        x,
        y,
        fraction
    ) in enumerate(
        coordinates,
        start=1
    ):

        # ----------------------------------------------------
        # Read directly from .svs
        # ----------------------------------------------------

        patch = slide.read_region(
            (x, y),
            0,
            (
                read_size,
                read_size
            )
        ).convert("RGB")

        # ----------------------------------------------------
        # Normalize physical resolution
        # ----------------------------------------------------

        if (
            read_size
            != PATCH_SIZE
        ):

            patch = patch.resize(
                (
                    PATCH_SIZE,
                    PATCH_SIZE
                ),
                Image.Resampling.LANCZOS
            )

        # ResNet preprocessing
        tensor = transform(
            patch
        )

        batch_images.append(
            tensor
        )

        # ----------------------------------------------------
        # Run batch through ResNet
        # ----------------------------------------------------

        if (
            len(batch_images)
            == BATCH_SIZE
            or
            i == total
        ):

            batch = torch.stack(
                batch_images
            ).to(device)

            with torch.no_grad():

                embeddings = model(
                    batch
                )

            embeddings = (
                embeddings
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            all_embeddings.append(
                embeddings
            )

            batch_images = []

            print(
                f"\rEncoded patches: "
                f"{i}/{total}",
                end=""
            )

    print()

    features = np.concatenate(
        all_embeddings,
        axis=0
    )

    return features


# ============================================================
# Process one patient
# ============================================================

def process_patient(
    row,
    model,
    transform,
    device,
    max_patches
):

    case_id = row[
        "case_id"
    ]

    file_id = row[
        "wsi_file_id"
    ]

    file_name = row[
        "wsi_file_name"
    ]

    split = row[
        "split"
    ]

    # ========================================================
    # Output
    # ========================================================

    output_dir = (
        FEATURE_ROOT
        / case_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_path = (
        output_dir
        / "features.npy"
    )

    done_path = (
        output_dir
        / "DONE"
    )

    # ========================================================
    # Resume support
    # ========================================================

    if (
        feature_path.exists()
        and
        done_path.exists()
    ):

        print(
            "Already processed — skipping."
        )

        return

    # ========================================================
    # Download
    # ========================================================

    slide_path = download_wsi(
        file_id,
        file_name
    )

    success = False

    slide = None

    try:

        # ====================================================
        # Open WSI
        # ====================================================

        slide = openslide.OpenSlide(
            str(slide_path)
        )

        base_mpp = get_base_mpp(
            slide
        )

        # Number of level-0 pixels required
        # to represent 256 pixels at TARGET_MPP
        read_size = int(
            round(
                PATCH_SIZE
                * TARGET_MPP
                / base_mpp
            )
        )

        read_size = max(
            1,
            read_size
        )

        print(
            "Slide dimensions:",
            slide.dimensions
        )

        print(
            "Base MPP:",
            base_mpp
        )

        print(
            "Level-0 patch size:",
            read_size
        )

        # ====================================================
        # Tissue mask
        # ====================================================

        thumbnail, mask = (
            create_tissue_mask(
                slide
            )
        )

        print(
            "Tissue fraction:",
            f"{mask.mean():.2%}"
        )

        # ====================================================
        # Patch coordinates
        # ====================================================

        coordinates = (
            find_candidate_coordinates(
                slide,
                mask,
                read_size
            )
        )

        print(
            "Candidate tissue patches:",
            len(coordinates)
        )

        if len(coordinates) == 0:

            raise RuntimeError(
                "No valid tissue patches found."
            )

        # Deterministic patient-specific seed
        patient_seed = (
            RANDOM_SEED
            +
            sum(
                ord(c)
                for c in case_id
            )
        )

        selected = (
            select_coordinates(
                coordinates,
                max_patches,
                patient_seed
            )
        )

        print(
            "Selected patches:",
            len(selected)
        )

        # ====================================================
        # ResNet feature extraction
        # ====================================================

        features = encode_slide(
            slide,
            selected,
            read_size,
            model,
            transform,
            device
        )

        print(
            "Feature matrix:",
            features.shape
        )

        # ====================================================
        # Save compact embeddings
        # ====================================================

        np.save(
            feature_path,
            features
        )

        # ====================================================
        # Save patch coordinates
        # ====================================================

        patch_index = pd.DataFrame(

            selected,

            columns=[
                "x_level0",
                "y_level0",
                "tissue_fraction"
            ]
        )

        patch_index[
            "embedding_row"
        ] = np.arange(
            len(patch_index)
        )

        patch_index.to_csv(
            output_dir
            / "patch_index.csv",
            index=False
        )

        # ====================================================
        # Save slide metadata
        # ====================================================

        metadata = {

            "case_id":
                case_id,

            "split":
                split,

            "wsi_file_id":
                file_id,

            "wsi_file_name":
                file_name,

            "base_mpp":
                base_mpp,

            "target_mpp":
                TARGET_MPP,

            "patch_size":
                PATCH_SIZE,

            "read_size_level0":
                read_size,

            "n_candidate_patches":
                len(coordinates),

            "n_selected_patches":
                len(selected),

            "embedding_dim":
                int(
                    features.shape[1]
                )
        }

        with open(
            output_dir
            / "metadata.json",
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        # ====================================================
        # Mark patient complete
        # ====================================================

        done_path.write_text(
            "complete"
        )

        success = True

    finally:

        if slide is not None:
            slide.close()

    # ========================================================
    # DELETE RAW WSI ONLY AFTER SUCCESS
    # ========================================================

    if success:

        if slide_path.exists():

            size_gb = (
                slide_path.stat().st_size
                / (1024 ** 3)
            )

            slide_path.unlink()

            print(
                f"Deleted raw WSI "
                f"({size_gb:.2f} GB)."
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
            "all"
        ],
        default="train"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--max-patches",
        type=int,
        default=DEFAULT_MAX_PATCHES
    )

    args = parser.parse_args()

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FEATURE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    if (
        args.split
        != "all"
    ):

        manifest = manifest[
            manifest["split"]
            == args.split
        ].copy()

    manifest = (
        manifest
        .drop_duplicates(
            subset=[
                "case_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        args.limit
        is not None
    ):

        manifest = (
            manifest
            .head(
                args.limit
            )
        )

    print("=" * 70)

    print(
        "STREAMING WSI FEATURE EXTRACTION"
    )

    print("=" * 70)

    print(
        "Patients:",
        len(manifest)
    )

    print(
        "Split:",
        args.split
    )

    print(
        "Max patches / patient:",
        args.max_patches
    )

    # ========================================================
    # Device
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

    # ========================================================
    # Load ResNet ONCE
    # ========================================================

    model, transform = (
        build_encoder(
            device
        )
    )

    # ========================================================
    # Process sequentially
    # ========================================================

    for index, row in (
        manifest.iterrows()
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"PATIENT "
            f"{index + 1}/"
            f"{len(manifest)}"
        )

        print(
            row["case_id"]
        )

        print(
            "=" * 70
        )

        try:

            process_patient(
                row,
                model,
                transform,
                device,
                args.max_patches
            )

        except Exception as error:

            print(
                "\nERROR processing patient:"
            )

            print(
                row["case_id"]
            )

            print(
                error
            )

            print(
                "Continuing with next patient."
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()