from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from lifelines import CoxPHFitter


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

RNA_PATH = DATA_DIR / "rna_log2_tpm.csv"
MANIFEST_PATH = DATA_DIR / "patient_manifest_split.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "rna"


# ============================================================
# Configuration
# ============================================================

N_GENES = 2000

PCA_COMPONENTS = [
    5,
    10,
    20,
    30,
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
# Load data
# ============================================================

def load_data():

    rna = pd.read_csv(
        RNA_PATH,
        index_col=0
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    # Keep only patients that exist in RNA matrix
    manifest = manifest[
        manifest["case_id"].isin(rna.index)
    ].copy()

    # Valid survival only
    manifest = manifest[
        manifest["survival_time"].notna()
    ].copy()

    manifest = manifest[
        manifest["survival_time"] > 0
    ].copy()

    return rna, manifest


# ============================================================
# Preprocessing
# ============================================================

def preprocess_fold(
    X_train,
    X_val,
    n_components
):
    """
    EVERYTHING is fitted only on fold training data:
        - variable gene selection
        - scaler
        - PCA
    """

    # --------------------------------------------------------
    # 1. Variable genes
    # --------------------------------------------------------

    variances = X_train.var(axis=0)

    selected_genes = (
        variances
        .sort_values(ascending=False)
        .head(N_GENES)
        .index
    )

    X_train = X_train[selected_genes]
    X_val = X_val[selected_genes]

    # --------------------------------------------------------
    # 2. Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    # --------------------------------------------------------
    # 3. PCA
    # --------------------------------------------------------

    effective_components = min(
        n_components,
        X_train_scaled.shape[0] - 1,
        X_train_scaled.shape[1]
    )

    pca = PCA(
        n_components=effective_components
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_val_pca = pca.transform(
        X_val_scaled
    )

    return (
        X_train_pca,
        X_val_pca,
        effective_components
    )


# ============================================================
# Create Cox dataframe
# ============================================================

def build_cox_dataframe(
    X,
    survival_time,
    event
):

    columns = [
        f"PC{i + 1}"
        for i in range(X.shape[1])
    ]

    df = pd.DataFrame(
        X,
        columns=columns
    )

    df["survival_time"] = (
        np.asarray(survival_time)
    )

    df["event"] = (
        np.asarray(event)
    )

    return df


# ============================================================
# Cross-validation for one configuration
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

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, events),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        time_train = times.iloc[train_idx]
        time_val = times.iloc[val_idx]

        event_train = events.iloc[train_idx]
        event_val = events.iloc[val_idx]

        # ----------------------------------------------------
        # Preprocessing inside fold
        # ----------------------------------------------------

        (
            X_train_pca,
            X_val_pca,
            effective_components
        ) = preprocess_fold(
            X_train,
            X_val,
            n_components
        )

        # ----------------------------------------------------
        # Cox dataframes
        # ----------------------------------------------------

        train_df = build_cox_dataframe(
            X_train_pca,
            time_train,
            event_train
        )

        val_df = build_cox_dataframe(
            X_val_pca,
            time_val,
            event_val
        )

        try:

            model = CoxPHFitter(
                penalizer=penalizer
            )

            model.fit(
                train_df,
                duration_col="survival_time",
                event_col="event",
                show_progress=False
            )

            val_cindex = model.score(
                val_df,
                scoring_method="concordance_index"
            )

            fold_scores.append(
                val_cindex
            )

        except Exception as error:

            print(
                f"\nFailed:"
                f" PCA={n_components},"
                f" penalizer={penalizer},"
                f" fold={fold}"
            )

            print(error)

            return None

    return {
        "n_components": n_components,
        "penalizer": penalizer,

        "mean_cv_cindex":
            np.mean(fold_scores),

        "std_cv_cindex":
            np.std(fold_scores),

        "fold_scores":
            fold_scores
    }


# ============================================================
# Hyperparameter search
# ============================================================

def cross_validation_search(
    X,
    times,
    events
):

    results = []

    total = (
        len(PCA_COMPONENTS)
        * len(PENALIZERS)
    )

    counter = 0

    for n_components in PCA_COMPONENTS:

        for penalizer in PENALIZERS:

            counter += 1

            print(
                f"[{counter}/{total}] "
                f"PCA={n_components}, "
                f"penalizer={penalizer}"
            )

            result = evaluate_configuration(
                X,
                times,
                events,
                n_components,
                penalizer
            )

            if result is not None:

                results.append(result)

                print(
                    f"    CV C-index = "
                    f"{result['mean_cv_cindex']:.4f} "
                    f"± "
                    f"{result['std_cv_cindex']:.4f}"
                )

    results_df = pd.DataFrame([
        {
            "n_components":
                r["n_components"],

            "penalizer":
                r["penalizer"],

            "mean_cv_cindex":
                r["mean_cv_cindex"],

            "std_cv_cindex":
                r["std_cv_cindex"]
        }

        for r in results
    ])

    results_df = results_df.sort_values(
        "mean_cv_cindex",
        ascending=False
    )

    return results_df


# ============================================================
# Train final model on development set
# ============================================================

def train_final_model(
    X_dev,
    time_dev,
    event_dev,
    X_test,
    n_components,
    penalizer
):

    # --------------------------------------------------------
    # Select variable genes using development set ONLY
    # --------------------------------------------------------

    variances = X_dev.var(axis=0)

    selected_genes = (
        variances
        .sort_values(ascending=False)
        .head(N_GENES)
        .index
    )

    X_dev = X_dev[selected_genes]
    X_test = X_test[selected_genes]

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_dev_scaled = scaler.fit_transform(
        X_dev
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    effective_components = min(
        n_components,
        X_dev_scaled.shape[0] - 1,
        X_dev_scaled.shape[1]
    )

    pca = PCA(
        n_components=effective_components
    )

    X_dev_pca = pca.fit_transform(
        X_dev_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    # --------------------------------------------------------
    # Cox
    # --------------------------------------------------------

    dev_df = build_cox_dataframe(
        X_dev_pca,
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
        X_test_pca,
        selected_genes,
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

    rna, manifest = load_data()

    # ========================================================
    # Development set = train + validation
    #
    # Test remains completely untouched.
    # ========================================================

    dev_manifest = manifest[
        manifest["split"].isin(
            ["train", "val"]
        )
    ].copy()

    test_manifest = manifest[
        manifest["split"] == "test"
    ].copy()

    X_dev = rna.loc[
        dev_manifest["case_id"]
    ]

    X_test = rna.loc[
        test_manifest["case_id"]
    ]

    time_dev = dev_manifest[
        "survival_time"
    ].reset_index(drop=True)

    event_dev = dev_manifest[
        "event"
    ].reset_index(drop=True)

    time_test = test_manifest[
        "survival_time"
    ].reset_index(drop=True)

    event_test = test_manifest[
        "event"
    ].reset_index(drop=True)

    X_dev = X_dev.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    print("=" * 60)
    print("RNA COX CROSS-VALIDATION")
    print("=" * 60)

    print(
        "\nDevelopment patients:",
        len(X_dev)
    )

    print(
        "Development deaths:",
        int(event_dev.sum())
    )

    print(
        "\nTest patients:",
        len(X_test)
    )

    print(
        "Test deaths:",
        int(event_test.sum())
    )

    # ========================================================
    # Cross-validation
    # ========================================================

    results = cross_validation_search(
        X_dev,
        time_dev,
        event_dev
    )

    results.to_csv(
        RESULTS_DIR
        / "cox_cv_results.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("TOP CONFIGURATIONS")
    print("=" * 60)

    print(
        results.head(10).to_string(
            index=False
        )
    )

    # ========================================================
    # Best configuration
    # ========================================================

    best = results.iloc[0]

    best_components = int(
        best["n_components"]
    )

    best_penalizer = float(
        best["penalizer"]
    )

    print("\nBest configuration:")

    print(
        "PCA components:",
        best_components
    )

    print(
        "Penalizer:",
        best_penalizer
    )

    print(
        "Mean CV C-index:",
        round(
            best["mean_cv_cindex"],
            4
        )
    )

    # ========================================================
    # Train final model
    # ========================================================

    (
        model,
        X_test_pca,
        genes,
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

    # ========================================================
    # Test data
    # ========================================================

    test_df = build_cox_dataframe(
        X_test_pca,
        time_test,
        event_test
    )

    test_cindex = model.score(
        test_df,
        scoring_method="concordance_index"
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULT")
    print("=" * 60)

    print(
        f"Test C-index: "
        f"{test_cindex:.4f}"
    )

    # ========================================================
    # Predictions
    # ========================================================

    features = test_df.drop(
        columns=[
            "survival_time",
            "event"
        ]
    )

    risk = model.predict_partial_hazard(
        features
    )

    predictions = pd.DataFrame({
        "case_id":
            test_manifest["case_id"].values,

        "survival_time":
            time_test.values,

        "event":
            event_test.values,

        "risk_score":
            risk.values
    })

    predictions.to_csv(
        RESULTS_DIR
        / "cox_cv_test_predictions.csv",
        index=False
    )

    pd.Series(
        genes,
        name="gene_id"
    ).to_csv(
        RESULTS_DIR
        / "cox_selected_genes.csv",
        index=False
    )


if __name__ == "__main__":
    main()