from pathlib import Path

import numpy as np
import pandas as pd


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

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pathology_features"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "feature_qc.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    manifest = pd.read_csv(MANIFEST_PATH)

    records = []

    total_size = 0

    for _, row in manifest.iterrows():

        case_id = row["case_id"]
        split = row["split"]

        patient_dir = FEATURE_ROOT / case_id

        feature_path = (
            patient_dir
            / "features.npy"
        )

        done_path = (
            patient_dir
            / "DONE"
        )

        exists = feature_path.exists()

        complete = (
            exists
            and done_path.exists()
        )

        n_patches = None
        embedding_dim = None
        valid = False
        size_mb = 0

        if exists:

            try:

                features = np.load(
                    feature_path,
                    mmap_mode="r"
                )

                if len(features.shape) == 2:

                    n_patches = (
                        features.shape[0]
                    )

                    embedding_dim = (
                        features.shape[1]
                    )

                    valid = (
                        n_patches > 0
                        and
                        embedding_dim == 2048
                    )

                size_bytes = (
                    feature_path
                    .stat()
                    .st_size
                )

                total_size += (
                    size_bytes
                )

                size_mb = (
                    size_bytes
                    / (1024 ** 2)
                )

            except Exception as error:

                print(
                    f"Problem reading "
                    f"{case_id}: {error}"
                )

        records.append({

            "case_id":
                case_id,

            "split":
                split,

            "complete":
                complete,

            "valid":
                valid,

            "n_patches":
                n_patches,

            "embedding_dim":
                embedding_dim,

            "size_mb":
                size_mb
        })

    qc = pd.DataFrame(
        records
    )

    # ========================================================
    # Summary
    # ========================================================

    print("=" * 70)
    print("PATHOLOGY FEATURE QC")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test"
    ]:

        subset = qc[
            qc["split"] == split
        ]

        print(
            f"\n{split.upper()}"
        )

        print(
            "Expected:",
            len(subset)
        )

        print(
            "Complete:",
            subset[
                "complete"
            ].sum()
        )

        print(
            "Valid:",
            subset[
                "valid"
            ].sum()
        )

        print(
            "Missing:",
            (
                ~subset["complete"]
            ).sum()
        )

    # ========================================================
    # Global information
    # ========================================================

    valid = qc[
        qc["valid"]
    ]

    print(
        "\nTotal valid patients:",
        len(valid)
    )

    if len(valid) > 0:

        print(
            "Mean patches / patient:",
            round(
                valid[
                    "n_patches"
                ].mean(),
                1
            )
        )

        print(
            "Min patches:",
            int(
                valid[
                    "n_patches"
                ].min()
            )
        )

        print(
            "Max patches:",
            int(
                valid[
                    "n_patches"
                ].max()
            )
        )

    print(
        "\nFeature storage:",
        f"{total_size / (1024 ** 3):.2f} GB"
    )

    # ========================================================
    # Save report
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    qc.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nQC saved to:\n"
        f"{OUTPUT_PATH}"
    )

    # ========================================================
    # Missing patients
    # ========================================================

    missing = qc[
        ~qc["complete"]
    ]

    if len(missing) > 0:

        print(
            "\nStill missing:",
            len(missing)
        )

        print(
            missing[
                [
                    "case_id",
                    "split"
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "\nAll pathology features "
            "are ready."
        )


if __name__ == "__main__":
    main()