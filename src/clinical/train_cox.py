from pathlib import Path

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "clinical"
)


# ============================================================
# Configuration
# ============================================================

NUMERIC_FEATURES = [
    "age"
]

CATEGORICAL_FEATURES = [
    "sex",
    "stage",
    "T",
    "N",
    "M"
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
# Load data
# ============================================================

def load_data():

    df = pd.read_csv(
        MANIFEST_PATH
    )

    df = df[
        df["survival_time"].notna()
    ].copy()

    df = df[
        df["survival_time"] > 0
    ].copy()

    df = df[
        df["event"].isin([0, 1])
    ].copy()

    df["event"] = (
        df["event"]
        .astype(int)
    )

    return df


# ============================================================
# Preprocessor
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ============================================================
# Convert transformed data to Cox dataframe
# ============================================================

def make_cox_dataframe(
    X,
    survival_time,
    event
):

    columns = [
        f"x_{i}"
        for i in range(
            X.shape[1]
        )
    ]

    df = pd.DataFrame(
        X,
        columns=columns
    )

    df["survival_time"] = (
        np.asarray(
            survival_time
        )
    )

    df["event"] = (
        np.asarray(
            event
        )
    )

    return df


# ============================================================
# Evaluate one penalizer
# ============================================================

def evaluate_penalizer(
    X,
    times,
    events,
    penalizer
):

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_scores = []

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

        # ----------------------------------------
        # Preprocessing fitted ONLY on fold train
        # ----------------------------------------

        preprocessor = (
            build_preprocessor()
        )

        X_train_processed = (
            preprocessor
            .fit_transform(
                X_train
            )
        )

        X_val_processed = (
            preprocessor
            .transform(
                X_val
            )
        )

        train_df = (
            make_cox_dataframe(
                X_train_processed,
                time_train,
                event_train
            )
        )

        val_df = (
            make_cox_dataframe(
                X_val_processed,
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
                duration_col="survival_time",
                event_col="event"
            )

            score = model.score(
                val_df,
                scoring_method="concordance_index"
            )

            fold_scores.append(
                score
            )

        except Exception as error:

            print(
                f"Failed penalizer="
                f"{penalizer}, fold={fold}"
            )

            print(error)

            return None

    return {
        "penalizer":
            penalizer,

        "mean_cv_cindex":
            np.mean(
                fold_scores
            ),

        "std_cv_cindex":
            np.std(
                fold_scores
            )
    }


# ============================================================
# Hyperparameter search
# ============================================================

def cross_validation_search(
    X_dev,
    time_dev,
    event_dev
):

    results = []

    for i, penalizer in enumerate(
        PENALIZERS,
        start=1
    ):

        print(
            f"[{i}/{len(PENALIZERS)}] "
            f"penalizer={penalizer}"
        )

        result = (
            evaluate_penalizer(
                X_dev,
                time_dev,
                event_dev,
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
    penalizer
):

    preprocessor = (
        build_preprocessor()
    )

    X_dev_processed = (
        preprocessor
        .fit_transform(
            X_dev
        )
    )

    X_test_processed = (
        preprocessor
        .transform(
            X_test
        )
    )

    dev_df = make_cox_dataframe(
        X_dev_processed,
        time_dev,
        event_dev
    )

    model = CoxPHFitter(
        penalizer=penalizer
    )

    model.fit(
        dev_df,
        duration_col="survival_time",
        event_col="event"
    )

    return (
        model,
        preprocessor,
        X_test_processed
    )


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    # ========================================================
    # Development / Test
    # ========================================================

    dev = df[
        df["split"].isin(
            [
                "train",
                "val"
            ]
        )
    ].copy()

    test = df[
        df["split"]
        == "test"
    ].copy()

    features = (
        NUMERIC_FEATURES
        +
        CATEGORICAL_FEATURES
    )

    X_dev = dev[
        features
    ].copy()

    X_test = test[
        features
    ].copy()

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
        "=" * 60
    )

    print(
        "CLINICAL COX CROSS-VALIDATION"
    )

    print(
        "=" * 60
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

    results = (
        cross_validation_search(
            X_dev,
            time_dev,
            event_dev
        )
    )

    results.to_csv(
        RESULTS_DIR
        / "clinical_cox_cv_results.csv",
        index=False
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TOP CONFIGURATIONS"
    )

    print(
        "=" * 60
    )

    print(
        results.to_string(
            index=False
        )
    )

    best = (
        results.iloc[0]
    )

    best_penalizer = float(
        best[
            "penalizer"
        ]
    )

    print(
        "\nBest penalizer:",
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
    # Final training
    # ========================================================

    (
        model,
        preprocessor,
        X_test_processed
    ) = train_final_model(
        X_dev,
        time_dev,
        event_dev,
        X_test,
        best_penalizer
    )

    test_df = make_cox_dataframe(
        X_test_processed,
        time_test,
        event_test
    )

    # ========================================================
    # Final test
    # ========================================================

    test_cindex = (
        model.score(
            test_df,
            scoring_method=
                "concordance_index"
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL CLINICAL COX RESULT"
    )

    print(
        "=" * 60
    )

    print(
        f"Test C-index: "
        f"{test_cindex:.4f}"
    )

    # ========================================================
    # Predictions
    # ========================================================

    features_test = (
        test_df
        .drop(
            columns=[
                "survival_time",
                "event"
            ]
        )
    )

    risk = (
        model
        .predict_partial_hazard(
            features_test
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
        / "clinical_cox_test_predictions.csv",
        index=False
    )


if __name__ == "__main__":
    main()