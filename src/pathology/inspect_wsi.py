from pathlib import Path

import numpy as np
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
    / "pathology_qc"
)


# ============================================================
# Configuration
# ============================================================

THUMBNAIL_MAX_SIZE = 2000

# Simple tissue-detection thresholds
SATURATION_THRESHOLD = 20
VALUE_THRESHOLD = 245


# ============================================================
# Find one pilot WSI
# ============================================================

def get_first_slide():

    slides = list(
        PILOT_DIR.glob("*.svs")
    )

    if len(slides) == 0:
        raise FileNotFoundError(
            f"No .svs file found in:\n{PILOT_DIR}"
        )

    return slides[0]


# ============================================================
# Print WSI information
# ============================================================

def inspect_slide(slide):

    print("=" * 60)
    print("WSI INFORMATION")
    print("=" * 60)

    print(
        "\nLevel count:",
        slide.level_count
    )

    print(
        "Full resolution:",
        slide.dimensions
    )

    print("\nPyramid:")

    for level in range(
        slide.level_count
    ):

        dimensions = (
            slide.level_dimensions[level]
        )

        downsample = (
            slide.level_downsamples[level]
        )

        print(
            f"Level {level}: "
            f"{dimensions} "
            f"downsample={downsample:.2f}x"
        )

    # --------------------------------------------------------
    # Physical resolution
    # --------------------------------------------------------

    mpp_x = slide.properties.get(
        openslide.PROPERTY_NAME_MPP_X
    )

    mpp_y = slide.properties.get(
        openslide.PROPERTY_NAME_MPP_Y
    )

    print("\nMicrons per pixel:")

    print(
        "MPP X:",
        mpp_x
    )

    print(
        "MPP Y:",
        mpp_y
    )

    # --------------------------------------------------------
    # Scanner vendor
    # --------------------------------------------------------

    vendor = slide.properties.get(
        openslide.PROPERTY_NAME_VENDOR
    )

    print(
        "\nVendor:",
        vendor
    )


# ============================================================
# Create thumbnail
# ============================================================

def create_thumbnail(slide):

    thumbnail = slide.get_thumbnail(
        (
            THUMBNAIL_MAX_SIZE,
            THUMBNAIL_MAX_SIZE
        )
    )

    thumbnail = thumbnail.convert(
        "RGB"
    )

    return thumbnail


# ============================================================
# Tissue detection
# ============================================================

def create_tissue_mask(thumbnail):
    """
    Basic H&E tissue detection.

    Background is generally:
        - very bright
        - low saturation

    Tissue generally has:
        - stronger color
        - lower brightness
    """

    hsv = thumbnail.convert(
        "HSV"
    )

    hsv_array = np.array(
        hsv
    )

    # PIL HSV:
    # channel 0 = Hue
    # channel 1 = Saturation
    # channel 2 = Value / brightness

    saturation = hsv_array[:, :, 1]

    value = hsv_array[:, :, 2]

    tissue_mask = (
        (saturation > SATURATION_THRESHOLD)
        &
        (value < VALUE_THRESHOLD)
    )

    return tissue_mask


# ============================================================
# Save mask
# ============================================================

def save_mask(mask, path):

    mask_image = (
        mask.astype(
            np.uint8
        )
        * 255
    )

    mask_image = Image.fromarray(
        mask_image
    )

    mask_image.save(
        path
    )


# ============================================================
# Tissue statistics
# ============================================================

def print_tissue_statistics(mask):

    total_pixels = mask.size

    tissue_pixels = mask.sum()

    tissue_fraction = (
        tissue_pixels
        / total_pixels
    )

    print("\n" + "=" * 60)
    print("TISSUE DETECTION")
    print("=" * 60)

    print(
        f"Tissue pixels: "
        f"{tissue_pixels:,}"
    )

    print(
        f"Total thumbnail pixels: "
        f"{total_pixels:,}"
    )

    print(
        f"Tissue fraction: "
        f"{tissue_fraction:.2%}"
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Select WSI
    # --------------------------------------------------------

    slide_path = get_first_slide()

    print(
        "\nOpening:\n",
        slide_path
    )

    # --------------------------------------------------------
    # Open WSI
    # --------------------------------------------------------

    slide = openslide.OpenSlide(
        str(slide_path)
    )

    try:

        # ----------------------------------------------------
        # Inspect pyramid
        # ----------------------------------------------------

        inspect_slide(
            slide
        )

        # ----------------------------------------------------
        # Thumbnail
        # ----------------------------------------------------

        thumbnail = create_thumbnail(
            slide
        )

        thumbnail_path = (
            OUTPUT_DIR
            / "wsi_thumbnail.png"
        )

        thumbnail.save(
            thumbnail_path
        )

        print(
            "\nThumbnail size:",
            thumbnail.size
        )

        # ----------------------------------------------------
        # Tissue mask
        # ----------------------------------------------------

        mask = create_tissue_mask(
            thumbnail
        )

        mask_path = (
            OUTPUT_DIR
            / "wsi_tissue_mask.png"
        )

        save_mask(
            mask,
            mask_path
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print_tissue_statistics(
            mask
        )

        print(
            "\nThumbnail saved to:\n",
            thumbnail_path
        )

        print(
            "\nTissue mask saved to:\n",
            mask_path
        )

    finally:

        slide.close()


if __name__ == "__main__":
    main()