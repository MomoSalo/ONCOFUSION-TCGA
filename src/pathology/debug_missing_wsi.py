from pathlib import Path

import pandas as pd
import openslide

from process_wsi_streaming import (
    create_tissue_mask,
    get_base_mpp,
    find_candidate_coordinates,
    PATCH_SIZE,
    TARGET_MPP
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "patient_manifest_split.csv"
)

WORK_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "pathology"
    / "working"
)

WSI_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "wsi_files_luad.csv"
)


CASE_ID = "8b119d1c-6d21-4bbd-8a00-12da7b97d6c4"


def main():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    row = manifest[
        manifest["case_id"] == CASE_ID
    ].iloc[0]

    file_name = row["wsi_file_name"]
    file_id = row["wsi_file_id"]

    slide_path = (
        WORK_DIR
        / file_name
    )

    print("=" * 70)
    print("DETAILED WSI DEBUG")
    print("=" * 70)

    print("\nCase:")
    print(CASE_ID)

    print("\nFile:")
    print(file_name)

    # ========================================================
    # Compare expected size with actual size
    # ========================================================

    metadata = pd.read_csv(
        WSI_METADATA_PATH
    )

    metadata_row = metadata[
        metadata["wsi_file_id"] == file_id
    ]

    actual_bytes = (
        slide_path.stat().st_size
    )

    print(
        "\nActual file size:",
        f"{actual_bytes / (1024**2):.2f} MB"
    )

    if not metadata_row.empty:

        expected_bytes = int(
            metadata_row.iloc[0][
                "wsi_file_size"
            ]
        )

        print(
            "Expected file size:",
            f"{expected_bytes / (1024**2):.2f} MB"
        )

        print(
            "Size match:",
            actual_bytes == expected_bytes
        )

    # ========================================================
    # Try OpenSlide
    # ========================================================

    print("\nTrying to open slide...")

    try:

        slide = openslide.OpenSlide(
            str(slide_path)
        )

    except Exception as error:

        print("\nOPENSLIDE ERROR:")
        print(error)

        return

    print("OpenSlide opened successfully.")

    try:

        # ====================================================
        # Slide dimensions
        # ====================================================

        print(
            "\nDimensions:",
            slide.dimensions
        )

        print(
            "Levels:",
            slide.level_count
        )

        for i in range(
            slide.level_count
        ):

            print(
                f"Level {i}:",
                slide.level_dimensions[i]
            )

        # ====================================================
        # MPP
        # ====================================================

        base_mpp = get_base_mpp(
            slide
        )

        print(
            "\nBase MPP:",
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
            "\nThumbnail size:",
            thumbnail.size
        )

        print(
            "Detected tissue fraction:",
            f"{mask.mean():.4%}"
        )

        # ====================================================
        # Candidate patches
        # ====================================================

        coordinates = (
            find_candidate_coordinates(
                slide,
                mask,
                read_size
            )
        )

        print(
            "\nCandidate patches:",
            len(coordinates)
        )

        if len(coordinates) == 0:

            print(
                "\nThe file is readable, but our "
                "current tissue detector found no "
                "usable patches."
            )

        else:

            print(
                "\nSlide appears processable."
            )

    finally:

        slide.close()


if __name__ == "__main__":
    main()