from pathlib import Path

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

QC_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "feature_qc.csv"
)

UNI_ROOT = (
    PROJECT_ROOT
    / "UNI2-h_features"
    / "TCGA"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_feature_matches.csv"
)


MIN_PATCHES = 100


# ============================================================
# Helper
# ============================================================

def slide_stem(filename):
    """
    Remove .svs or .h5 while preserving the entire TCGA slide ID.
    """

    name = Path(filename).name

    if name.lower().endswith(".svs"):
        return name[:-4]

    if name.lower().endswith(".h5"):
        return name[:-3]

    return Path(name).stem


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load original manifest
    # --------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    qc = pd.read_csv(
        QC_PATH
    )

    # Same WSI QC used in our previous experiments
    qc = qc[
        (qc["complete"] == True)
        &
        (qc["valid"] == True)
        &
        (qc["n_patches"] >= MIN_PATCHES)
    ].copy()

    cohort = manifest.merge(
        qc[
            [
                "case_id",
                "n_patches"
            ]
        ],
        on="case_id",
        how="inner"
    )

    print("=" * 70)
    print("UNI2-h FEATURE MATCHING")
    print("=" * 70)

    print(
        "\nUsable WSI cohort:",
        len(cohort)
    )

    # --------------------------------------------------------
    # Find all UNI2-h H5 files
    # --------------------------------------------------------

    h5_files = list(
        UNI_ROOT.rglob("*.h5")
    )

    print(
        "Available UNI2-h H5 slides:",
        len(h5_files)
    )

    # --------------------------------------------------------
    # Build lookup:
    #
    # slide stem -> H5 path
    # --------------------------------------------------------

    uni_lookup = {}

    for path in h5_files:

        stem = slide_stem(
            path.name
        )

        if stem in uni_lookup:

            print(
                "WARNING duplicate UNI slide:",
                stem
            )

        uni_lookup[stem] = path

    # --------------------------------------------------------
    # Match exact selected slide
    # --------------------------------------------------------

    records = []

    for _, row in cohort.iterrows():

        case_id = row["case_id"]
        wsi_name = row["wsi_file_name"]

        selected_stem = slide_stem(
            wsi_name
        )

        matched_path = (
            uni_lookup.get(
                selected_stem
            )
        )

        records.append(
            {
                "case_id":
                    case_id,

                "split":
                    row["split"],

                "wsi_file_name":
                    wsi_name,

                "slide_stem":
                    selected_stem,

                "resnet_n_patches":
                    row["n_patches"],

                "uni2_available":
                    matched_path
                    is not None,

                "uni2_h5_path":
                    (
                        str(matched_path)
                        if matched_path
                        is not None
                        else None
                    ),
            }
        )

    matches = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total = len(matches)

    matched = int(
        matches[
            "uni2_available"
        ].sum()
    )

    missing = (
        total - matched
    )

    print("\n" + "=" * 70)
    print("MATCH RESULTS")
    print("=" * 70)

    print(
        "\nSelected WSI patients:",
        total
    )

    print(
        "Exact UNI2-h slide matches:",
        matched
    )

    print(
        "Missing exact matches:",
        missing
    )

    if total > 0:

        print(
            "Coverage:",
            f"{matched / total:.2%}"
        )

    # --------------------------------------------------------
    # By split
    # --------------------------------------------------------

    print("\nBy split:")

    for split in [
        "train",
        "val",
        "test"
    ]:

        subset = matches[
            matches["split"]
            == split
        ]

        n = len(subset)

        m = int(
            subset[
                "uni2_available"
            ].sum()
        )

        print(
            f"{split:5s}: "
            f"{m}/{n} "
            f"({m/n:.2%})"
            if n > 0
            else f"{split}: 0"
        )

    # --------------------------------------------------------
    # Missing examples
    # --------------------------------------------------------

    missing_df = matches[
        ~matches[
            "uni2_available"
        ]
    ]

    if len(missing_df) > 0:

        print(
            "\nMissing selected slides:"
        )

        print(
            missing_df[
                [
                    "case_id",
                    "split",
                    "wsi_file_name"
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    matches.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()