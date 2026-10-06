from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCH_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_feature_matches.csv"
)


# ============================================================
# Sequence reordering
# ============================================================

def sequence_reorder(x, R):
    """
    x shape:
        [L, D]

    Split sequence into segments and reorder them
    using the transpose-like operation described
    in MambaMIL.

    Returns:
        reordered_x
        original_length
        padded_length
    """

    L, D = x.shape

    # --------------------------------------------------------
    # Number of columns / segments
    # --------------------------------------------------------

    N = int(
        np.ceil(
            L / R
        )
    )

    padded_length = (
        R * N
    )

    # --------------------------------------------------------
    # Pad with zero feature vectors
    # --------------------------------------------------------

    if padded_length > L:

        padding = np.zeros(
            (
                padded_length - L,
                D
            ),
            dtype=x.dtype
        )

        x_padded = np.concatenate(
            [
                x,
                padding
            ],
            axis=0
        )

    else:

        x_padded = x

    # --------------------------------------------------------
    # Original:
    #
    # [L, D]
    #
    # reshape:
    #
    # [R, N, D]
    # --------------------------------------------------------

    x_2d = x_padded.reshape(
        R,
        N,
        D
    )

    # --------------------------------------------------------
    # Reorder by exchanging R and N
    #
    # [R, N, D]
    #      ↓
    # [N, R, D]
    # --------------------------------------------------------

    x_reordered = (
        x_2d
        .transpose(
            1,
            0,
            2
        )
        .reshape(
            padded_length,
            D
        )
    )

    return (
        x_reordered,
        L,
        padded_length
    )


# ============================================================
# Restoration
# ============================================================

def sequence_restore(
    x_reordered,
    R,
    original_length
):
    """
    Reverse the sequence-reordering operation.
    """

    padded_length, D = (
        x_reordered.shape
    )

    N = (
        padded_length
        // R
    )

    x_restored = (
        x_reordered
        .reshape(
            N,
            R,
            D
        )
        .transpose(
            1,
            0,
            2
        )
        .reshape(
            padded_length,
            D
        )
    )

    # remove padding
    return x_restored[
        :original_length
    ]


# ============================================================
# Main
# ============================================================

def main():

    matches = pd.read_csv(
        MATCH_PATH
    )

    matches = matches[
        matches[
            "uni2_available"
        ] == True
    ]

    # --------------------------------------------------------
    # Take one UNI2-h slide
    # --------------------------------------------------------

    row = matches.iloc[0]

    h5_path = Path(
        row[
            "uni2_h5_path"
        ]
    )

    print("=" * 70)
    print("MambaMIL SEQUENCE REORDERING TEST")
    print("=" * 70)

    print(
        "\nSlide:"
    )

    print(
        h5_path.name
    )

    # --------------------------------------------------------
    # Load features + coordinates
    # --------------------------------------------------------

    with h5py.File(
        h5_path,
        "r"
    ) as f:

        features = (
            f[
                "features"
            ][0]
        )

        coords = (
            f[
                "coords"
            ][0]
        )

    print(
        "\nFeatures:",
        features.shape
    )

    print(
        "Coordinates:",
        coords.shape
    )

    print(
        "\nFirst 10 original coordinates:"
    )

    print(
        coords[
            :10
        ]
    )

    # ========================================================
    # Use tiny artificial indices first
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOY EXAMPLE"
    )

    print(
        "=" * 70
    )

    toy = np.arange(
        10,
        dtype=np.float32
    ).reshape(
        10,
        1
    )

    R = 3

    reordered, L, padded = (
        sequence_reorder(
            toy,
            R
        )
    )

    restored = (
        sequence_restore(
            reordered,
            R,
            L
        )
    )

    print(
        "\nOriginal:"
    )

    print(
        toy[:, 0]
    )

    print(
        "\nReordered:"
    )

    print(
        reordered[:, 0]
    )

    print(
        "\nRestored:"
    )

    print(
        restored[:, 0]
    )

    print(
        "\nRestoration correct:",
        np.allclose(
            toy,
            restored
        )
    )

    # ========================================================
    # Test on actual UNI2-h embeddings
    # ========================================================

    R = 16

    reordered, L, padded = (
        sequence_reorder(
            features,
            R
        )
    )

    restored = (
        sequence_restore(
            reordered,
            R,
            L
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "REAL UNI2-h BAG"
    )

    print(
        "=" * 70
    )

    print(
        "\nOriginal length:",
        L
    )

    print(
        "Padded length:",
        padded
    )

    print(
        "Reordered shape:",
        reordered.shape
    )

    print(
        "Restored shape:",
        restored.shape
    )

    print(
        "Restoration error:",
        np.max(
            np.abs(
                features
                -
                restored
            )
        )
    )


if __name__ == "__main__":
    main()