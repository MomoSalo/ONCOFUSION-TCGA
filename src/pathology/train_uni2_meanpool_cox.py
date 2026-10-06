from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


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

MATCH_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_feature_matches.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "uni2_meanpool_cox"
)


# ============================================================
# Configuration
# ============================================================

PCA_COMPONENTS = [
    5,
    10,
    20,
    50,
    100
]

PENALIZERS = [
    0.001,
    0.01,
    0.1,
    0.5,
    1.0,
    5.0,
    10.0
]

N_FOLDS = 5
RANDOM_STATE = 42


# ============================================================
# Load cohort
# ============================================================

def load_cohort():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    matches = pd.read_csv(
        MATCH_PATH
    )

    matches = matches[
        matches["uni2_available"] == True
    ].copy()

    cohort = manifest.merge(
        matches[
            [
                "case_id",
                "uni2_h5_path"
            ]
        ],
        on="case_id",
        how="inner"
    )

    cohort = cohort[
        cohort["survival_time"].notna()
    ].copy()

    cohort = cohort[
        cohort["survival_time"] > 0
    ].copy()

    cohort = cohort[
        cohort["event"].isin([0, 1])
    ].copy()

    cohort["event"] = (
        cohort["event"]
        .astype(int)
    )

    cohort = (
        cohort
        .drop_duplicates("case_id")
        .reset_index(drop=True)
    )

    return cohort


# ============================================================
# Mean pooling
# ============================================================

def mean_pool_uni2(h5_path):

    with h5py.File(
        h5_path,
        "r"
    ) as f:

        features = (
            f["features"][:]
        )

    # Mahmood Lab format:
    #
    # [1, N, 1536]
    #
    # Convert to:
    #
    # [N, 1536]

    if (
        features.ndim == 3
        and
        features.shape[0] == 1
    ):

        features = features[0]

    if (
        features.ndim != 2
        or
        features.shape[1] != 1536
    ):

        raise ValueError(
            f"Unexpected UNI2 shape: "
            f"{features.shape}"
        )

    features = features.astype(
        np.float32
    )

    # --------------------------------------------------------
    # L2 normalize each patch embedding
    # --------------------------------------------------------

    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(
        norms,
        1e-12
    )

    features = (
        features
        / norms
    )

    # --------------------------------------------------------
    # Mean pooling
    # --------------------------------------------------------

    vector = features.mean(
        axis=0
    )

    return vector


# ============================================================
# Build WSI matrix
# ============================================================

def build_matrix(cohort):

    vectors = []

    case_ids = []

    total = len(cohort)

    print(
        "\nBuilding UNI2-h "
        "mean-pooled vectors..."
    )

    for i, row in cohort.iterrows():

        vector = mean_pool_uni2(
            row["uni2_h5_path"]
        )

        vectors.append(
            vector
        )

        case_ids.append(
            row["case_id"]
        )

        if (
            (i + 1) % 25 == 0
            or
            (i + 1) == total
        ):

            print(
                f"{i + 1}/{total}"
            )

    X = np.stack(
        vectors,
        axis=0
    )

    X = pd.DataFrame(
        X,
        index=case_ids
    )

    print(
        "\nUNI2-h patient matrix:",
        X.shape
    )

    return X


# ============================================================
# Cox dataframe
# ============================================================

def make_cox_df(
    X,
    times,
    events
):

    columns = [
        f"PC{i + 1}"
        for i in range(
            X.shape[1]
        )
    ]

    df = pd.DataFrame(
        X,
        columns=columns
    )

    df["survival_time"] = (
        np.asarray(times)
    )

    df["event"] = (
        np.asarray(events)
    )

    return df


# ============================================================
# One CV configuration
# ============================================================

def evaluate_configuration(
    X,
    times,
    events,
    n_components,
    penalizer
):

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = []

    for train_idx, val_idx in (
        splitter.split(
            X,
            events
        )
    ):

        X_train = X.iloc[
            train_idx
        ]

        X_val = X.iloc[
            val_idx
        ]

        y_time_train = times.iloc[
            train_idx
        ]

        y_time_val = times.iloc[
            val_idx
        ]

        y_event_train = events.iloc[
            train_idx
        ]

        y_event_val = events.iloc[
            val_idx
        ]

        # ----------------------------------------------------
        # Standardization
        # fitted only on fold training set
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_val_scaled = (
            scaler.transform(
                X_val
            )
        )

        # ----------------------------------------------------
        # PCA
        # fitted only on fold training set
        # ----------------------------------------------------

        pca = PCA(
            n_components=
                n_components
        )

        X_train_pca = (
            pca.fit_transform(
                X_train_scaled
            )
        )

        X_val_pca = (
            pca.transform(
                X_val_scaled
            )
        )

        train_df = make_cox_df(
            X_train_pca,
            y_time_train,
            y_event_train
        )

        val_df = make_cox_df(
            X_val_pca,
            y_time_val,
            y_event_val
        )

        model = CoxPHFitter(
            penalizer=penalizer
        )

        model.fit(
            train_df,
            duration_col=
                "survival_time",
            event_col=
                "event",
            show_progress=False
        )

        score = model.score(
            val_df,
            scoring_method=
                "concordance_index"
        )

        scores.append(
            score
        )

    return {

        "n_components":
            n_components,

        "penalizer":
            penalizer,

        "mean_cv_cindex":
            np.mean(scores),

        "std_cv_cindex":
            np.std(scores)
    }


# ============================================================
# CV search
# ============================================================

def cross_validation_search(
    X,
    times,
    events
):

    results = []

    total = (
        len(PCA_COMPONENTS)
        *
        len(PENALIZERS)
    )

    counter = 0

    for n_components in (
        PCA_COMPONENTS
    ):

        for penalizer in (
            PENALIZERS
        ):

            counter += 1

            print(
                f"[{counter}/{total}] "
                f"PCA={n_components}, "
                f"penalizer={penalizer}"
            )

            try:

                result = (
                    evaluate_configuration(
                        X,
                        times,
                        events,
                        n_components,
                        penalizer
                    )
                )

                results.append(
                    result
                )

                print(
                    f"    CV C-index = "
                    f"{result['mean_cv_cindex']:.4f}"
                    f" ± "
                    f"{result['std_cv_cindex']:.4f}"
                )

            except Exception as error:

                print(
                    "    FAILED:",
                    error
                )

    results = pd.DataFrame(
        results
    )

    return (
        results
        .sort_values(
            "mean_cv_cindex",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("UNI2-h MEAN POOLING + COX")
    print("=" * 70)

    cohort = load_cohort()

    print(
        "\nTotal usable UNI2 patients:",
        len(cohort)
    )

    for split in [
        "train",
        "val",
        "test"
    ]:

        subset = cohort[
            cohort["split"]
            == split
        ]

        print(
            f"\n{split.upper()}:",
            len(subset)
        )

        print(
            "Deaths:",
            int(
                subset["event"].sum()
            )
        )

    # ========================================================
    # Build patient embeddings
    # ========================================================

    X = build_matrix(
        cohort
    )

    # ========================================================
    # Dev / test
    # ========================================================

    dev = cohort[
        cohort["split"].isin(
            [
                "train",
                "val"
            ]
        )
    ].copy()

    test = cohort[
        cohort["split"]
        == "test"
    ].copy()

    X_dev = (
        X.loc[
            dev["case_id"]
        ]
        .reset_index(drop=True)
    )

    X_test = (
        X.loc[
            test["case_id"]
        ]
        .reset_index(drop=True)
    )

    time_dev = (
        dev["survival_time"]
        .reset_index(drop=True)
    )

    event_dev = (
        dev["event"]
        .reset_index(drop=True)
    )

    time_test = (
        test["survival_time"]
        .reset_index(drop=True)
    )

    event_test = (
        test["event"]
        .reset_index(drop=True)
    )

    print(
        "\nDevelopment patients:",
        len(dev)
    )

    print(
        "Development deaths:",
        int(
            event_dev.sum()
        )
    )

    print(
        "\nTest patients:",
        len(test)
    )

    print(
        "Test deaths:",
        int(
            event_test.sum()
        )
    )

    # ========================================================
    # CV
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CROSS-VALIDATION"
    )

    print(
        "=" * 70
    )

    results = (
        cross_validation_search(
            X_dev,
            time_dev,
            event_dev
        )
    )

    results.to_csv(
        RESULTS_DIR
        / "cv_results.csv",
        index=False
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOP CONFIGURATIONS"
    )

    print(
        "=" * 70
    )

    print(
        results
        .head(10)
        .to_string(
            index=False
        )
    )

    best = results.iloc[0]

    n_components = int(
        best["n_components"]
    )

    penalizer = float(
        best["penalizer"]
    )

    print(
        "\nBest PCA components:",
        n_components
    )

    print(
        "Best penalizer:",
        penalizer
    )

    print(
        "Best CV C-index:",
        f"{best['mean_cv_cindex']:.4f}"
        f" ± "
        f"{best['std_cv_cindex']:.4f}"
    )

    # ========================================================
    # Final preprocessing
    # ========================================================

    scaler = StandardScaler()

    X_dev_scaled = (
        scaler.fit_transform(
            X_dev
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    pca = PCA(
        n_components=
            n_components
    )

    X_dev_pca = (
        pca.fit_transform(
            X_dev_scaled
        )
    )

    X_test_pca = (
        pca.transform(
            X_test_scaled
        )
    )

    # ========================================================
    # Final Cox
    # ========================================================

    train_df = make_cox_df(
        X_dev_pca,
        time_dev,
        event_dev
    )

    test_df = make_cox_df(
        X_test_pca,
        time_test,
        event_test
    )

    model = CoxPHFitter(
        penalizer=penalizer
    )

    model.fit(
        train_df,
        duration_col=
            "survival_time",
        event_col=
            "event"
    )

    test_cindex = model.score(
        test_df,
        scoring_method=
            "concordance_index"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL UNI2-h WSI RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTest C-index: "
        f"{test_cindex:.4f}"
    )


if __name__ == "__main__":
    main()