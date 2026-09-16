from pathlib import Path

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

QC_PATH = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "feature_qc.csv"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pathology_features"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "meanpool_cox"
)


# ============================================================
# Configuration
# ============================================================

MIN_PATCHES = 100

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
# Build WSI cohort
# ============================================================

def load_cohort():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    qc = pd.read_csv(
        QC_PATH
    )

    # --------------------------------------------------------
    # Same pathology QC as Attention MIL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Valid survival information
    # --------------------------------------------------------

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
        .drop_duplicates(
            subset=["case_id"]
        )
        .reset_index(drop=True)
    )

    return cohort


# ============================================================
# L2-normalize each patch embedding
# ============================================================

def normalize_patch_embeddings(features):

    norms = np.linalg.norm(
        features,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(
        norms,
        1e-12
    )

    return (
        features
        / norms
    )


# ============================================================
# Build one patient vector
# ============================================================

def mean_pool_patient(case_id):

    feature_path = (
        FEATURE_ROOT
        / case_id
        / "features.npy"
    )

    features = np.load(
        feature_path
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Same idea as MIL preprocessing:
    # normalize every ResNet patch vector
    # --------------------------------------------------------

    features = (
        normalize_patch_embeddings(
            features
        )
    )

    # --------------------------------------------------------
    # Mean pooling
    #
    # [N patches, 2048]
    #       ↓
    # [2048]
    # --------------------------------------------------------

    patient_vector = (
        features.mean(
            axis=0
        )
    )

    return patient_vector


# ============================================================
# Build patient-level WSI matrix
# ============================================================

def build_wsi_matrix(cohort):

    vectors = []

    case_ids = []

    total = len(
        cohort
    )

    print(
        "\nBuilding mean-pooled WSI vectors..."
    )

    for i, row in cohort.iterrows():

        case_id = row[
            "case_id"
        ]

        vector = mean_pool_patient(
            case_id
        )

        vectors.append(
            vector
        )

        case_ids.append(
            case_id
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
        "\nWSI matrix:",
        X.shape
    )

    return X


# ============================================================
# Preprocess one fold
# ============================================================

def preprocess_fold(
    X_train,
    X_val,
    n_components
):

    # --------------------------------------------------------
    # Standardization
    # fitted ONLY on fold training data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PCA
    # fitted ONLY on fold training data
    # --------------------------------------------------------

    effective_components = min(
        n_components,
        X_train_scaled.shape[0] - 1,
        X_train_scaled.shape[1]
    )

    pca = PCA(
        n_components=
            effective_components
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

    return (
        X_train_pca,
        X_val_pca,
        effective_components
    )


# ============================================================
# Cox dataframe
# ============================================================

def make_cox_dataframe(
    X,
    survival_time,
    event
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

    df[
        "survival_time"
    ] = np.asarray(
        survival_time
    )

    df[
        "event"
    ] = np.asarray(
        event
    )

    return df


# ============================================================
# Evaluate one configuration
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

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        splitter.split(
            X,
            events
        ),
        start=1
    ):

        X_train = (
            X.iloc[
                train_idx
            ]
        )

        X_val = (
            X.iloc[
                val_idx
            ]
        )

        time_train = (
            times.iloc[
                train_idx
            ]
        )

        time_val = (
            times.iloc[
                val_idx
            ]
        )

        event_train = (
            events.iloc[
                train_idx
            ]
        )

        event_val = (
            events.iloc[
                val_idx
            ]
        )

        # ----------------------------------------------------
        # Fold-specific preprocessing
        # ----------------------------------------------------

        (
            X_train_pca,
            X_val_pca,
            _
        ) = preprocess_fold(
            X_train,
            X_val,
            n_components
        )

        train_df = (
            make_cox_dataframe(
                X_train_pca,
                time_train,
                event_train
            )
        )

        val_df = (
            make_cox_dataframe(
                X_val_pca,
                time_val,
                event_val
            )
        )

        try:

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

        except Exception as error:

            print(
                "\nFailed:"
            )

            print(
                "PCA =",
                n_components
            )

            print(
                "penalizer =",
                penalizer
            )

            print(
                "fold =",
                fold
            )

            print(
                error
            )

            return None

    return {

        "n_components":
            n_components,

        "penalizer":
            penalizer,

        "mean_cv_cindex":
            np.mean(
                scores
            ),

        "std_cv_cindex":
            np.std(
                scores
            )
    }


# ============================================================
# CV search
# ============================================================

def cross_validation_search(
    X_dev,
    time_dev,
    event_dev
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

            result = (
                evaluate_configuration(
                    X_dev,
                    time_dev,
                    event_dev,
                    n_components,
                    penalizer
                )
            )

            if result is not None:

                results.append(
                    result
                )

                print(
                    "    CV C-index = "
                    f"{result['mean_cv_cindex']:.4f}"
                    " ± "
                    f"{result['std_cv_cindex']:.4f}"
                )

    results = pd.DataFrame(
        results
    )

    results = (
        results
        .sort_values(
            "mean_cv_cindex",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return results


# ============================================================
# Train final model
# ============================================================

def train_final_model(
    X_dev,
    time_dev,
    event_dev,
    X_test,
    n_components,
    penalizer
):

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

    effective_components = min(
        n_components,
        X_dev_scaled.shape[0] - 1,
        X_dev_scaled.shape[1]
    )

    pca = PCA(
        n_components=
            effective_components
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

    dev_df = (
        make_cox_dataframe(
            X_dev_pca,
            time_dev,
            event_dev
        )
    )

    model = CoxPHFitter(
        penalizer=penalizer
    )

    model.fit(
        dev_df,
        duration_col=
            "survival_time",
        event_col=
            "event"
    )

    return (
        model,
        X_test_pca,
        scaler,
        pca
    )


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Cohort
    # ========================================================

    cohort = load_cohort()

    print("=" * 70)
    print("WSI MEAN POOLING + COX")
    print("=" * 70)

    print(
        "\nUsable patients:",
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
                subset[
                    "event"
                ].sum()
            )
        )

    # ========================================================
    # Mean-pooled ResNet vectors
    # ========================================================

    X = build_wsi_matrix(
        cohort
    )

    # ========================================================
    # Development / Test
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

    X_dev = X.loc[
        dev["case_id"]
    ]

    X_test = X.loc[
        test["case_id"]
    ]

    time_dev = (
        dev[
            "survival_time"
        ]
        .reset_index(
            drop=True
        )
    )

    event_dev = (
        dev[
            "event"
        ]
        .reset_index(
            drop=True
        )
    )

    time_test = (
        test[
            "survival_time"
        ]
        .reset_index(
            drop=True
        )
    )

    event_test = (
        test[
            "event"
        ]
        .reset_index(
            drop=True
        )
    )

    X_dev = (
        X_dev
        .reset_index(
            drop=True
        )
    )

    X_test = (
        X_test
        .reset_index(
            drop=True
        )
    )

    print(
        "\nDevelopment patients:",
        len(X_dev)
    )

    print(
        "Development deaths:",
        int(
            event_dev.sum()
        )
    )

    print(
        "\nTest patients:",
        len(X_test)
    )

    print(
        "Test deaths:",
        int(
            event_test.sum()
        )
    )

    # ========================================================
    # CV search
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

    # ========================================================
    # Results
    # ========================================================

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

    best = (
        results.iloc[0]
    )

    best_components = int(
        best[
            "n_components"
        ]
    )

    best_penalizer = float(
        best[
            "penalizer"
        ]
    )

    print(
        "\nBest PCA components:",
        best_components
    )

    print(
        "Best penalizer:",
        best_penalizer
    )

    print(
        "Mean CV C-index:",
        round(
            best[
                "mean_cv_cindex"
            ],
            4
        )
    )

    # ========================================================
    # Final model
    # ========================================================

    (
        model,
        X_test_pca,
        scaler,
        pca
    ) = train_final_model(
        X_dev,
        time_dev,
        event_dev,
        X_test,
        best_components,
        best_penalizer
    )

    test_df = (
        make_cox_dataframe(
            X_test_pca,
            time_test,
            event_test
        )
    )

    test_cindex = (
        model.score(
            test_df,
            scoring_method=
                "concordance_index"
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL MEAN-POOL WSI RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTest C-index: "
        f"{test_cindex:.4f}"
    )

    # ========================================================
    # Test predictions
    # ========================================================

    test_features = (
        test_df.drop(
            columns=[
                "survival_time",
                "event"
            ]
        )
    )

    risk = (
        model.predict_partial_hazard(
            test_features
        )
    )

    predictions = pd.DataFrame({

        "case_id":
            test[
                "case_id"
            ].values,

        "survival_time":
            time_test.values,

        "event":
            event_test.values,

        "risk_score":
            risk.values
    })

    predictions.to_csv(
        RESULTS_DIR
        / "test_predictions.csv",
        index=False
    )


if __name__ == "__main__":
    main()