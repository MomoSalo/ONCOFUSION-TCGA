from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ATTENTION_DIR = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_attention_mil"
    / "attention"
)


def entropy(weights):

    weights = np.asarray(
        weights,
        dtype=np.float64
    )

    weights = np.clip(
        weights,
        1e-12,
        None
    )

    return -np.sum(
        weights * np.log(weights)
    )


def main():

    files = list(
        ATTENTION_DIR.glob("*.csv")
    )

    print(
        "Number of test patients:",
        len(files)
    )

    rows = []

    for path in files:

        df = pd.read_csv(path)

        weights = (
            df["attention"]
            .values
        )

        # Ensure normalized
        weights = (
            weights
            / weights.sum()
        )

        n = len(weights)

        # ----------------------------------------------------
        # Top-k attention concentration
        # ----------------------------------------------------

        sorted_weights = np.sort(
            weights
        )[::-1]

        top1 = sorted_weights[0]

        top10 = (
            sorted_weights[
                :min(10, n)
            ].sum()
        )

        top50 = (
            sorted_weights[
                :min(50, n)
            ].sum()
        )

        # ----------------------------------------------------
        # Normalized entropy
        #
        # 1 -> nearly uniform attention
        # 0 -> highly concentrated attention
        # ----------------------------------------------------

        h = entropy(
            weights
        )

        max_entropy = np.log(n)

        normalized_entropy = (
            h / max_entropy
            if n > 1
            else 0
        )

        rows.append(
            {
                "case_id":
                    path.stem,

                "n_patches":
                    n,

                "max_attention":
                    top1,

                "top10_attention":
                    top10,

                "top50_attention":
                    top50,

                "normalized_entropy":
                    normalized_entropy
            }
        )

    results = pd.DataFrame(
        rows
    )

    print(
        "\nSummary:"
    )

    print(
        results[
            [
                "n_patches",
                "max_attention",
                "top10_attention",
                "top50_attention",
                "normalized_entropy"
            ]
        ]
        .describe()
        .to_string()
    )

    print(
        "\nMost selective slides:"
    )

    print(
        results
        .sort_values(
            "normalized_entropy"
        )
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()