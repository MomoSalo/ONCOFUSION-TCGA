from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

QC_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "feature_qc.csv"
)


def main():

    qc = pd.read_csv(QC_PATH)

    small_bags = qc[
        (qc["valid"] == True)
        &
        (qc["n_patches"] < 100)
    ][
        [
            "case_id",
            "split",
            "n_patches"
        ]
    ].sort_values(
        "n_patches"
    )

    print("=" * 60)
    print("PATIENTS WITH FEWER THAN 100 PATCHES")
    print("=" * 60)

    print(small_bags.to_string(index=False))

    print(
        "\nNumber of patients:",
        len(small_bags)
    )


if __name__ == "__main__":
    main()