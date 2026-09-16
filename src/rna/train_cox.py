from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

RNA_DIR = DATA_DIR / "rna_preprocessed"

MANIFEST_PATH = DATA_DIR / "patient_manifest_split.csv"

TRAIN_PATH = RNA_DIR / "X_train_rna.csv"
VAL_PATH = RNA_DIR / "X_val_rna.csv"
TEST_PATH = RNA_DIR / "X_test_rna.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "rna"


# ============================================================
# Load data
# ============================================================

def load_data():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    X_train = pd.read_csv(
        TRAIN_PATH,
        index_col=0
    )

    X_val = pd.read_csv(
        VAL_PATH,
        index_col=0
    )

    X_test = pd.read_csv(
        TEST_PATH,
        index_col=0
    )

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    return (
        manifest,
        X_train,
        X_val,
        X_test
    )


# ============================================================
# Add survival labels
# ============================================================

def attach_survival_labels(
    X,
    manifest
):

    labels = (
        manifest[
            [
                "case_id",
                "survival_time",
                "event"
            ]
        ]
        .drop_duplicates("case_id")
        .set_index("case_id")
    )

    # Keep only patients for which we have RNA
    labels = labels.loc[X.index]

    df = X.copy()

    df["survival_time"] = labels[
        "survival_time"
    ]

    df["event"] = labels[
        "event"
    ]

    # Remove invalid survival entries
    df = df.dropna(
        subset=[
            "survival_time",
            "event"
        ]
    )

    df = df[
        df["survival_time"] > 0
    ]

    return df


# ============================================================
# Train one Cox model
# ============================================================

def train_cox(
    train_df,
    penalizer
):

    model = CoxPHFitter(
        penalizer=penalizer
    )

    model.fit(
        train_df,
        duration_col="survival_time",
        event_col="event"
    )

    return model


# ============================================================
# Hyperparameter selection
# ============================================================

def select_penalizer(
    train_df,
    val_df
):

    penalizers = [
        0.001,
        0.01,
        0.1,
        0.5,
        1.0,
        5.0,
        10.0
    ]

    results = []

    best_model = None
    best_penalizer = None
    best_val_cindex = -1

    print("\nSelecting penalizer...")
    print("=" * 50)

    for penalizer in penalizers:

        try:

            model = train_cox(
                train_df,
                penalizer
            )

            train_cindex = model.score(
                train_df,
                scoring_method="concordance_index"
            )

            val_cindex = model.score(
                val_df,
                scoring_method="concordance_index"
            )

            print(
                f"penalizer={penalizer:<6} "
                f"train={train_cindex:.4f} "
                f"val={val_cindex:.4f}"
            )

            results.append({
                "penalizer": penalizer,
                "train_cindex": train_cindex,
                "val_cindex": val_cindex
            })

            if val_cindex > best_val_cindex:

                best_val_cindex = val_cindex

                best_penalizer = penalizer

                best_model = model

        except Exception as error:

            print(
                f"penalizer={penalizer} failed:"
            )

            print(error)

    return (
        best_model,
        best_penalizer,
        best_val_cindex,
        pd.DataFrame(results)
    )


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        manifest,
        X_train,
        X_val,
        X_test
    ) = load_data()

    # --------------------------------------------------------
    # Attach survival labels
    # --------------------------------------------------------

    train_df = attach_survival_labels(
        X_train,
        manifest
    )

    val_df = attach_survival_labels(
        X_val,
        manifest
    )

    test_df = attach_survival_labels(
        X_test,
        manifest
    )

    print("\nPatients used:")

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    # --------------------------------------------------------
    # Choose regularization
    # --------------------------------------------------------

    (
        best_model,
        best_penalizer,
        best_val_cindex,
        results
    ) = select_penalizer(
        train_df,
        val_df
    )

    if best_model is None:
        raise RuntimeError(
            "No Cox model could be fitted."
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_cindex = best_model.score(
        test_df,
        scoring_method="concordance_index"
    )

    train_cindex = best_model.score(
        train_df,
        scoring_method="concordance_index"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BEST RNA COX MODEL")
    print("=" * 60)

    print(
        "Best penalizer:",
        best_penalizer
    )

    print(
        f"Train C-index:      "
        f"{train_cindex:.4f}"
    )

    print(
        f"Validation C-index: "
        f"{best_val_cindex:.4f}"
    )

    print(
        f"Test C-index:       "
        f"{test_cindex:.4f}"
    )

    # --------------------------------------------------------
    # Save hyperparameter results
    # --------------------------------------------------------

    results.to_csv(
        RESULTS_DIR
        / "cox_penalizer_search.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    X_test_features = test_df.drop(
        columns=[
            "survival_time",
            "event"
        ]
    )

    risk = best_model.predict_partial_hazard(
        X_test_features
    )

    predictions = pd.DataFrame({
        "case_id": X_test_features.index,
        "risk_score": risk.values,
        "survival_time":
            test_df["survival_time"].values,
        "event":
            test_df["event"].values
    })

    predictions.to_csv(
        RESULTS_DIR
        / "cox_test_predictions.csv",
        index=False
    )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()