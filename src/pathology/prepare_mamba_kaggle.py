from pathlib import Path
import shutil
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPACT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uni2_compact"
)

COMPACT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_compact_manifest.csv"
)

PATIENT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "patient_manifest_split.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "kaggle"
    / "mambamil_tcga_luad"
)

OUTPUT_FEATURES = (
    OUTPUT_ROOT
    / "features"
)


def main():

    print("=" * 70)
    print("PREPARE MambaMIL KAGGLE DATASET")
    print("=" * 70)

    OUTPUT_FEATURES.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Load manifests
    # ========================================================

    compact = pd.read_csv(
        COMPACT_MANIFEST
    )

    patients = pd.read_csv(
        PATIENT_MANIFEST
    )

    print("\nCompact patients:", len(compact))
    print("Patient manifest:", len(patients))

    print("\nPatient manifest columns:")
    print(list(patients.columns))

    # ========================================================
    # Detect survival columns
    # ========================================================

    time_candidates = [
        "survival_time",
        "time",
        "os_time",
        "OS.time",
    ]

    event_candidates = [
        "event",
        "survival_event",
        "os_event",
        "OS",
    ]

    time_col = next(
        (
            c for c in time_candidates
            if c in patients.columns
        ),
        None
    )

    event_col = next(
        (
            c for c in event_candidates
            if c in patients.columns
        ),
        None
    )

    if time_col is None:
        raise ValueError(
            "Could not detect survival-time column."
        )

    if event_col is None:
        raise ValueError(
            "Could not detect event column."
        )

    print(
        "\nUsing survival time column:",
        time_col
    )

    print(
        "Using event column:",
        event_col
    )

    # ========================================================
    # Merge compact feature list with survival information
    # ========================================================

    merged = compact.merge(
        patients[
            [
                "case_id",
                time_col,
                event_col
            ]
        ],
        on="case_id",
        how="left"
    )

    # Normalize column names
    merged = merged.rename(
        columns={
            time_col: "survival_time",
            event_col: "event"
        }
    )

    # ========================================================
    # Validate labels
    # ========================================================

    if merged["survival_time"].isna().any():
        raise ValueError(
            "Missing survival times after merge."
        )

    if merged["event"].isna().any():
        raise ValueError(
            "Missing event labels after merge."
        )

    # ========================================================
    # Copy compact H5 files
    # ========================================================

    kaggle_paths = []

    for i, row in merged.iterrows():

        case_id = str(
            row["case_id"]
        )

        source = (
            COMPACT_DIR
            / f"{case_id}.h5"
        )

        destination = (
            OUTPUT_FEATURES
            / f"{case_id}.h5"
        )

        if not source.exists():
            raise FileNotFoundError(
                source
            )

        if not destination.exists():
            shutil.copy2(
                source,
                destination
            )

        # Path that will be used inside Kaggle
        kaggle_paths.append(
            f"features/{case_id}.h5"
        )

        if (i + 1) % 50 == 0:
            print(
                f"Copied {i + 1}/{len(merged)}"
            )

    merged["feature_file"] = (
        kaggle_paths
    )

    # ========================================================
    # Keep only fields needed by training
    # ========================================================

    output_manifest = merged[
        [
            "case_id",
            "split",
            "survival_time",
            "event",
            "original_n_patches",
            "kept_n_patches",
            "feature_file"
        ]
    ].copy()

    output_manifest.to_csv(
        OUTPUT_ROOT / "manifest.csv",
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("KAGGLE DATASET READY")
    print("=" * 70)

    print(
        "\nPatients:",
        len(output_manifest)
    )

    print(
        "\nSplit counts:"
    )

    print(
        output_manifest[
            "split"
        ].value_counts()
    )

    print(
        "\nEvent counts:"
    )

    print(
        output_manifest.groupby(
            "split"
        )["event"].agg(
            ["count", "sum"]
        )
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()