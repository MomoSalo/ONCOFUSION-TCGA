from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_feature_matches.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uni2_compact"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_compact_manifest.csv"
)


# ============================================================
# Configuration
# ============================================================

MAX_PATCHES = 2048


# ============================================================
# Deterministic spatial subsampling
# ============================================================

def select_indices(n_patches, max_patches):
    """
    Preserve the original spatial sequence while selecting
    approximately evenly spaced patches across the slide.
    """

    if n_patches <= max_patches:
        return np.arange(
            n_patches,
            dtype=np.int64
        )

    indices = np.linspace(
        0,
        n_patches - 1,
        max_patches,
        dtype=np.int64
    )

    return indices


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    matches = pd.read_csv(
        MATCH_PATH
    )

    matches = matches[
        matches["uni2_available"] == True
    ].copy()

    print("=" * 70)
    print("COMPACT UNI2-h FEATURES")
    print("=" * 70)

    print(
        "\nMatched slides:",
        len(matches)
    )

    print(
        "Maximum patches / slide:",
        MAX_PATCHES
    )

    records = []

    total_original_patches = 0
    total_kept_patches = 0

    for i, row in matches.iterrows():

        case_id = row[
            "case_id"
        ]

        input_path = Path(
            row[
                "uni2_h5_path"
            ]
        )

        output_path = (
            OUTPUT_ROOT
            / f"{case_id}.h5"
        )

        # ====================================================
        # Load original file
        # ====================================================

        with h5py.File(
            input_path,
            "r"
        ) as f:

            features = f[
                "features"
            ][:]

            coords = f[
                "coords"
            ][:]

        # Mahmood Lab format:
        #
        # features: [1, N, 1536]
        # coords:   [1, N, 2]

        if (
            features.ndim == 3
            and
            features.shape[0] == 1
        ):
            features = features[0]

        if (
            coords.ndim == 3
            and
            coords.shape[0] == 1
        ):
            coords = coords[0]

        n_original = (
            features.shape[0]
        )

        total_original_patches += (
            n_original
        )

        # ====================================================
        # Select patches
        # ====================================================

        indices = select_indices(
            n_original,
            MAX_PATCHES
        )

        features = features[
            indices
        ]

        coords = coords[
            indices
        ]

        n_kept = (
            features.shape[0]
        )

        total_kept_patches += (
            n_kept
        )

        # ====================================================
        # Reduce storage
        # ====================================================

        features = features.astype(
            np.float16
        )

        coords = coords.astype(
            np.int32
        )

        # ====================================================
        # Save compact H5
        # ====================================================

        with h5py.File(
            output_path,
            "w"
        ) as f:

            f.create_dataset(
                "features",
                data=features,
                compression="gzip",
                compression_opts=4
            )

            f.create_dataset(
                "coords",
                data=coords,
                compression="gzip",
                compression_opts=4
            )

            f.attrs[
                "case_id"
            ] = case_id

            f.attrs[
                "original_n_patches"
            ] = n_original

            f.attrs[
                "kept_n_patches"
            ] = n_kept

        records.append(
            {
                "case_id":
                    case_id,

                "split":
                    row["split"],

                "original_n_patches":
                    n_original,

                "kept_n_patches":
                    n_kept,

                "compact_path":
                    str(
                        output_path
                    )
            }
        )

        if (
            (i + 1) % 25 == 0
            or
            (i + 1) == len(matches)
        ):

            print(
                f"{i + 1}/{len(matches)}"
            )

    # ========================================================
    # Save compact manifest
    # ========================================================

    compact_manifest = (
        pd.DataFrame(
            records
        )
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    compact_manifest.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    # ========================================================
    # Statistics
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPACTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOriginal total patches:",
        total_original_patches
    )

    print(
        "Kept total patches:",
        total_kept_patches
    )

    print(
        "Fraction retained:",
        f"{total_kept_patches / total_original_patches:.2%}"
    )

    print(
        "\nCompact dataset:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()