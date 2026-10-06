from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPACT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uni2_compact"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_compact_manifest.csv"
)

EXPECTED_PATIENTS = 451
EXPECTED_FEATURE_DIM = 1536
MAX_PATCHES = 2048


def main():

    print("=" * 70)
    print("VALIDATE COMPACT UNI2-h DATASET")
    print("=" * 70)

    # ========================================================
    # Manifest
    # ========================================================

    manifest = pd.read_csv(MANIFEST_PATH)

    print("\nManifest rows:", len(manifest))

    print("\nSplit counts:")
    print(manifest["split"].value_counts())

    # ========================================================
    # Files
    # ========================================================

    files = sorted(
        COMPACT_DIR.glob("*.h5")
    )

    print("\nH5 files:", len(files))

    errors = []
    stats = []

    # ========================================================
    # Validate every patient
    # ========================================================

    for i, path in enumerate(files, start=1):

        try:

            with h5py.File(path, "r") as f:

                # --------------------------------------------
                # Required datasets
                # --------------------------------------------

                if "features" not in f:
                    errors.append(
                        (path.name, "missing features")
                    )
                    continue

                if "coords" not in f:
                    errors.append(
                        (path.name, "missing coords")
                    )
                    continue

                features = f["features"][:]
                coords = f["coords"][:]

                # --------------------------------------------
                # Shapes
                # --------------------------------------------

                if features.ndim != 2:
                    errors.append(
                        (
                            path.name,
                            f"features ndim={features.ndim}"
                        )
                    )

                if coords.ndim != 2:
                    errors.append(
                        (
                            path.name,
                            f"coords ndim={coords.ndim}"
                        )
                    )

                if features.shape[1] != EXPECTED_FEATURE_DIM:
                    errors.append(
                        (
                            path.name,
                            f"feature dim={features.shape[1]}"
                        )
                    )

                if coords.shape[1] != 2:
                    errors.append(
                        (
                            path.name,
                            f"coords dim={coords.shape}"
                        )
                    )

                if features.shape[0] != coords.shape[0]:
                    errors.append(
                        (
                            path.name,
                            "features/coords length mismatch"
                        )
                    )

                # --------------------------------------------
                # Number of patches
                # --------------------------------------------

                n_patches = features.shape[0]

                if n_patches == 0:
                    errors.append(
                        (
                            path.name,
                            "zero patches"
                        )
                    )

                if n_patches > MAX_PATCHES:
                    errors.append(
                        (
                            path.name,
                            f"{n_patches} > {MAX_PATCHES}"
                        )
                    )

                # --------------------------------------------
                # Dtypes
                # --------------------------------------------

                if features.dtype != np.float16:
                    errors.append(
                        (
                            path.name,
                            f"feature dtype={features.dtype}"
                        )
                    )

                if coords.dtype != np.int32:
                    errors.append(
                        (
                            path.name,
                            f"coords dtype={coords.dtype}"
                        )
                    )

                # --------------------------------------------
                # Numerical validity
                # --------------------------------------------

                if not np.isfinite(features).all():
                    errors.append(
                        (
                            path.name,
                            "NaN or Inf in features"
                        )
                    )

                if not np.isfinite(coords).all():
                    errors.append(
                        (
                            path.name,
                            "NaN or Inf in coords"
                        )
                    )

                # --------------------------------------------
                # Metadata
                # --------------------------------------------

                original_n = f.attrs.get(
                    "original_n_patches",
                    np.nan
                )

                kept_n = f.attrs.get(
                    "kept_n_patches",
                    np.nan
                )

                if not np.isnan(kept_n):
                    if int(kept_n) != n_patches:
                        errors.append(
                            (
                                path.name,
                                "kept_n_patches attr mismatch"
                            )
                        )

                stats.append(
                    {
                        "file": path.name,
                        "n_patches": n_patches,
                        "original_n_patches": original_n,
                        "feature_dtype": str(features.dtype),
                        "coords_dtype": str(coords.dtype),
                    }
                )

        except Exception as e:

            errors.append(
                (
                    path.name,
                    f"FAILED TO OPEN: {e}"
                )
            )

        if i % 50 == 0:
            print(
                f"Checked {i}/{len(files)}"
            )

    # ========================================================
    # Summary
    # ========================================================

    stats = pd.DataFrame(stats)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        "\nPatients checked:",
        len(stats)
    )

    if len(stats) > 0:

        print(
            "\nPatch count:"
        )

        print(
            stats["n_patches"].describe()
        )

        print(
            "\nPatients with exactly 2048 patches:",
            (
                stats["n_patches"]
                == MAX_PATCHES
            ).sum()
        )

        print(
            "\nPatients with <2048 patches:",
            (
                stats["n_patches"]
                < MAX_PATCHES
            ).sum()
        )

    # ========================================================
    # Manifest ↔ file consistency
    # ========================================================

    manifest_case_ids = set(
        manifest["case_id"].astype(str)
    )

    file_case_ids = set(
        p.stem
        for p in files
    )

    missing_files = (
        manifest_case_ids
        -
        file_case_ids
    )

    unexpected_files = (
        file_case_ids
        -
        manifest_case_ids
    )

    print(
        "\nMissing files from manifest:",
        len(missing_files)
    )

    print(
        "Unexpected files:",
        len(unexpected_files)
    )

    # ========================================================
    # Final status
    # ========================================================

    print("\n" + "=" * 70)

    if (
        len(files) == EXPECTED_PATIENTS
        and
        len(manifest) == EXPECTED_PATIENTS
        and
        len(errors) == 0
        and
        len(missing_files) == 0
        and
        len(unexpected_files) == 0
    ):

        print("VALIDATION PASSED ✅")

        print(
            "\nCompact UNI2-h dataset is ready for MambaMIL."
        )

    else:

        print("VALIDATION FAILED ❌")

        print(
            "\nNumber of errors:",
            len(errors)
        )

        for error in errors[:30]:
            print(error)

        if len(errors) > 30:
            print("...")

        if missing_files:
            print(
                "\nMissing:",
                sorted(missing_files)[:10]
            )

        if unexpected_files:
            print(
                "\nUnexpected:",
                sorted(unexpected_files)[:10]
            )


if __name__ == "__main__":
    main()