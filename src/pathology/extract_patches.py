from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import openslide


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pathology_patches"
)


# ============================================================
# Configuration
# ============================================================

# Final patch given to our future neural network
PATCH_SIZE = 256

# Standard pathology resolution we want approximately
TARGET_MPP = 0.5

# A patch must contain at least 60% detected tissue
MIN_TISSUE_FRACTION = 0.60

# Pilot only
MAX_PATCHES = 200

THUMBNAIL_MAX_SIZE = 2000

SATURATION_THRESHOLD = 20
VALUE_THRESHOLD = 245

RANDOM_SEED = 42


# ============================================================
# Find one slide
# ============================================================

def get_first_slide():

    slides = list(
        PILOT_DIR.glob("*.svs")
    )

    if not slides:
        raise FileNotFoundError(
            f"No .svs files found in:\n{PILOT_DIR}"
        )

    return slides[0]


# ============================================================
# Create tissue mask
# ============================================================

def create_tissue_mask(slide):

    thumbnail = slide.get_thumbnail(
        (
            THUMBNAIL_MAX_SIZE,
            THUMBNAIL_MAX_SIZE
        )
    ).convert("RGB")

    hsv = np.array(
        thumbnail.convert("HSV")
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (
        (saturation > SATURATION_THRESHOLD)
        &
        (value < VALUE_THRESHOLD)
    )

    return thumbnail, mask


# ============================================================
# Determine resolution
# ============================================================

def get_base_mpp(slide):

    mpp = slide.properties.get(
        openslide.PROPERTY_NAME_MPP_X
    )

    if mpp is None:

        print(
            "WARNING: MPP metadata missing."
        )

        print(
            "Using 0.5 microns/pixel as fallback."
        )

        return TARGET_MPP

    return float(mpp)


# ============================================================
# Calculate tissue fraction for a candidate patch
# ============================================================

def get_tissue_fraction(
    mask,
    x,
    y,
    read_size,
    slide_width,
    slide_height
):

    mask_height, mask_width = mask.shape

    # Map Level-0 WSI coordinates
    # to thumbnail coordinates

    x0 = int(
        x / slide_width
        * mask_width
    )

    x1 = int(
        (x + read_size)
        / slide_width
        * mask_width
    )

    y0 = int(
        y / slide_height
        * mask_height
    )

    y1 = int(
        (y + read_size)
        / slide_height
        * mask_height
    )

    # Ensure at least one mask pixel
    x1 = max(
        x1,
        x0 + 1
    )

    y1 = max(
        y1,
        y0 + 1
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
# Find tissue patch coordinates
# ============================================================

def find_candidate_patches(
    slide,
    mask,
    read_size
):

    slide_width, slide_height = (
        slide.dimensions
    )

    coordinates = []

    print("\nSearching tissue regions...")

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

            tissue_fraction = (
                get_tissue_fraction(
                    mask,
                    x,
                    y,
                    read_size,
                    slide_width,
                    slide_height
                )
            )

            if (
                tissue_fraction
                >= MIN_TISSUE_FRACTION
            ):

                coordinates.append(
                    (
                        x,
                        y,
                        tissue_fraction
                    )
                )

    print(
        "Valid tissue patches found:",
        len(coordinates)
    )

    return coordinates


# ============================================================
# Sample pilot patches
# ============================================================

def sample_coordinates(
    coordinates
):

    random.seed(
        RANDOM_SEED
    )

    if (
        len(coordinates)
        <= MAX_PATCHES
    ):
        return coordinates

    return random.sample(
        coordinates,
        MAX_PATCHES
    )


# ============================================================
# Extract patches
# ============================================================

def extract_patches(
    slide,
    coordinates,
    read_size,
    output_dir
):

    records = []

    for i, (
        x,
        y,
        tissue_fraction
    ) in enumerate(
        coordinates
    ):

        # ----------------------------------------------------
        # Read original region at Level 0
        # ----------------------------------------------------

        patch = slide.read_region(
            (x, y),
            0,
            (
                read_size,
                read_size
            )
        )

        patch = patch.convert(
            "RGB"
        )

        # ----------------------------------------------------
        # Normalize resolution
        #
        # Example:
        # original = 0.25 µm/px
        #
        # read 512×512
        # resize -> 256×256
        #
        # gives approximately 0.5 µm/px
        # ----------------------------------------------------

        if read_size != PATCH_SIZE:

            patch = patch.resize(
                (
                    PATCH_SIZE,
                    PATCH_SIZE
                ),
                Image.Resampling.LANCZOS
            )

        file_name = (
            f"patch_{i:04d}.png"
        )

        patch.save(
            output_dir
            / file_name
        )

        records.append({
            "patch_id": i,

            "file_name":
                file_name,

            "x_level0":
                x,

            "y_level0":
                y,

            "read_size_level0":
                read_size,

            "output_size":
                PATCH_SIZE,

            "tissue_fraction":
                tissue_fraction
        })

        if (
            (i + 1) % 20
            == 0
        ):

            print(
                f"Extracted "
                f"{i + 1}/"
                f"{len(coordinates)}"
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# Main
# ============================================================

def main():

    slide_path = (
        get_first_slide()
    )

    slide_name = (
        slide_path.stem
    )

    slide_output_dir = (
        OUTPUT_DIR
        / slide_name
    )

    slide_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("WSI PATCH EXTRACTION")
    print("=" * 60)

    print(
        "\nSlide:\n",
        slide_path.name
    )

    slide = openslide.OpenSlide(
        str(slide_path)
    )

    try:

        # ----------------------------------------------------
        # Resolution
        # ----------------------------------------------------

        base_mpp = (
            get_base_mpp(
                slide
            )
        )

        print(
            "\nOriginal MPP:",
            base_mpp
        )

        print(
            "Target MPP:",
            TARGET_MPP
        )

        # ----------------------------------------------------
        # How many Level-0 pixels correspond to one
        # 256×256 patch at TARGET_MPP?
        # ----------------------------------------------------

        read_size = int(
            round(
                PATCH_SIZE
                *
                TARGET_MPP
                /
                base_mpp
            )
        )

        print(
            "Level-0 region size:",
            f"{read_size} x {read_size}"
        )

        print(
            "Final patch size:",
            f"{PATCH_SIZE} x {PATCH_SIZE}"
        )

        # ----------------------------------------------------
        # Tissue mask
        # ----------------------------------------------------

        thumbnail, mask = (
            create_tissue_mask(
                slide
            )
        )

        thumbnail.save(
            slide_output_dir
            / "thumbnail.png"
        )

        mask_image = (
            mask.astype(
                np.uint8
            )
            * 255
        )

        Image.fromarray(
            mask_image
        ).save(
            slide_output_dir
            / "tissue_mask.png"
        )

        # ----------------------------------------------------
        # Find tissue regions
        # ----------------------------------------------------

        candidates = (
            find_candidate_patches(
                slide,
                mask,
                read_size
            )
        )

        # ----------------------------------------------------
        # Random subset for pilot
        # ----------------------------------------------------

        selected = (
            sample_coordinates(
                candidates
            )
        )

        print(
            "\nPatches selected:",
            len(selected)
        )

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        metadata = (
            extract_patches(
                slide,
                selected,
                read_size,
                slide_output_dir
            )
        )

        metadata.to_csv(
            slide_output_dir
            / "patch_metadata.csv",
            index=False
        )

        print("\n" + "=" * 60)

        print(
            "PATCH EXTRACTION COMPLETE"
        )

        print("=" * 60)

        print(
            "\nSaved to:\n",
            slide_output_dir
        )

    finally:

        slide.close()


if __name__ == "__main__":
    main()