from pathlib import Path
import copy
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

RNA_PATH = DATA_DIR / "rna_log2_tpm.csv"
MANIFEST_PATH = DATA_DIR / "patient_manifest_split.csv"

RESULTS_DIR = PROJECT_ROOT / "results" / "rna" / "deepsurv"

N_GENES = 2000

# We keep 20 PCs because this was selected
# on the development cohort in the Cox experiment.
N_COMPONENTS = 20

N_FOLDS = 5

RANDOM_STATE = 42

LEARNING_RATE = 1e-3

MAX_EPOCHS = 500
PATIENCE = 50


# ============================================================
# Small hyperparameter search
# ============================================================

HIDDEN_OPTIONS = [
    (32,),
    (64, 32),
    (128, 64)
]

DROPOUT_OPTIONS = [
    0.1,
    0.3
]

WEIGHT_DECAY_OPTIONS = [
    1e-4,
    1e-3
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    manifest = manifest[
        manifest["case_id"].isin(rna.index)
    ].copy()

    manifest = manifest[
        manifest["survival_time"].notna()
    ].copy()

    manifest = manifest[
        manifest["event"].isin([0, 1])
    ].copy()

    manifest = manifest[
        manifest["survival_time"] > 0
    ].copy()

    manifest["event"] = (
        manifest["event"]
        .astype(int)
    )

    return rna, manifest


# ============================================================
# RNA preprocessing
# ============================================================

def preprocess_fold(
    X_train,
    X_val
):

    # --------------------------------------------------------
    # Gene variance selection
    # FIT ON TRAIN ONLY
    # --------------------------------------------------------

    variances = X_train.var(axis=0)

    selected_genes = (
        variances
        .sort_values(ascending=False)
        .head(N_GENES)
        .index
    )

    X_train = X_train[
        selected_genes
    ]

    X_val = X_val[
        selected_genes
    ]

    # --------------------------------------------------------
    # Standardization
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
    # --------------------------------------------------------

    pca = PCA(
        n_components=N_COMPONENTS
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
        selected_genes,
        scaler,
        pca
    )


# ============================================================
# DeepSurv network
# ============================================================

class DeepSurv(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout
    ):

        super().__init__()

        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            layers.append(
                nn.Linear(
                    previous_dim,
                    hidden_dim
                )
            )

            layers.append(
                nn.ReLU()
            )

            layers.append(
                nn.Dropout(
                    dropout
                )
            )

            previous_dim = hidden_dim

        # Final scalar = log-risk
        layers.append(
            nn.Linear(
                previous_dim,
                1
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return (
            self.network(x)
            .squeeze(-1)
        )


# ============================================================
# Cox partial likelihood loss
# ============================================================

def cox_loss(
    risk,
    times,
    events
):
    """
    Negative Cox partial log-likelihood.

    Breslow handling of tied event times.
    """

    # --------------------------------------------------------
    # Sort patients:
    # longest survival -> shortest survival
    # --------------------------------------------------------

    order = torch.argsort(
        times,
        descending=True
    )

    risk = risk[order]
    times = times[order]
    events = events[order]

    # --------------------------------------------------------
    # log(sum exp(risk)) over each risk set
    # --------------------------------------------------------

    log_cumulative_risk = (
        torch.logcumsumexp(
            risk,
            dim=0
        )
    )

    unique_times = torch.unique_consecutive(
        times
    )

    log_likelihood = torch.tensor(
        0.0,
        device=risk.device
    )

    number_events = events.sum()

    for time in unique_times:

        mask = (
            times == time
        )

        event_mask = (
            mask
            &
            (events == 1)
        )

        d = event_mask.sum()

        if d == 0:
            continue

        # Numerator
        event_risk = (
            risk[event_mask]
            .sum()
        )

        # Last patient in this tied time group
        indices = torch.where(
            mask
        )[0]

        last_index = indices[-1]

        # Breslow denominator
        denominator = (
            d
            *
            log_cumulative_risk[
                last_index
            ]
        )

        log_likelihood = (
            log_likelihood
            +
            event_risk
            -
            denominator
        )

    return (
        -log_likelihood
        /
        number_events.clamp(min=1)
    )


# ============================================================
# C-index
# ============================================================

def calculate_cindex(
    model,
    X,
    times,
    events,
    device
):

    model.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X,
            dtype=torch.float32,
            device=device
        )

        risk = (
            model(X_tensor)
            .cpu()
            .numpy()
        )

    # lifelines expects larger prediction
    # = longer survival.
    #
    # DeepSurv predicts larger value
    # = higher risk.
    #
    # Therefore use -risk.

    return concordance_index(
        times,
        -risk,
        events
    )


# ============================================================
# Train one fold
# ============================================================

def train_one_fold(
    X_train,
    time_train,
    event_train,
    X_val,
    time_val,
    event_val,
    hidden_dims,
    dropout,
    weight_decay,
    device,
    seed
):

    set_seed(seed)

    model = DeepSurv(
        input_dim=N_COMPONENTS,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )

    # --------------------------------------------------------
    # Tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=device
    )

    time_train_tensor = torch.tensor(
        time_train,
        dtype=torch.float32,
        device=device
    )

    event_train_tensor = torch.tensor(
        event_train,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_cindex = -np.inf

    best_state = None

    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        optimizer.zero_grad()

        risk = model(
            X_train_tensor
        )

        loss = cox_loss(
            risk,
            time_train_tensor,
            event_train_tensor
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_cindex = calculate_cindex(
            model,
            X_val,
            time_val,
            event_val,
            device
        )

        if val_cindex > best_cindex:

            best_cindex = val_cindex

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            break

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_cindex,
        best_epoch
    )


# ============================================================
# Evaluate one configuration with CV
# ============================================================

def evaluate_configuration(
    X,
    times,
    events,
    hidden_dims,
    dropout,
    weight_decay,
    device
):

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_scores = []
    best_epochs = []

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

        # ----------------------------------------------------
        # Raw RNA
        # ----------------------------------------------------

        X_train_raw = (
            X.iloc[train_idx]
        )

        X_val_raw = (
            X.iloc[val_idx]
        )

        # ----------------------------------------------------
        # Preprocessing INSIDE fold
        # ----------------------------------------------------

        (
            X_train,
            X_val,
            _,
            _,
            _
        ) = preprocess_fold(
            X_train_raw,
            X_val_raw
        )

        time_train = (
            times.iloc[train_idx]
            .values
        )

        time_val = (
            times.iloc[val_idx]
            .values
        )

        event_train = (
            events.iloc[train_idx]
            .values
        )

        event_val = (
            events.iloc[val_idx]
            .values
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            model,
            val_cindex,
            best_epoch
        ) = train_one_fold(
            X_train,
            time_train,
            event_train,
            X_val,
            time_val,
            event_val,
            hidden_dims,
            dropout,
            weight_decay,
            device,
            seed=RANDOM_STATE + fold
        )

        fold_scores.append(
            val_cindex
        )

        best_epochs.append(
            best_epoch
        )

    return {

        "hidden_dims":
            str(hidden_dims),

        "dropout":
            dropout,

        "weight_decay":
            weight_decay,

        "mean_cv_cindex":
            np.mean(
                fold_scores
            ),

        "std_cv_cindex":
            np.std(
                fold_scores
            ),

        "median_best_epoch":
            int(
                np.median(
                    best_epochs
                )
            )
    }


# ============================================================
# Hyperparameter search
# ============================================================

def search_hyperparameters(
    X_dev,
    time_dev,
    event_dev,
    device
):

    results = []

    total = (
        len(HIDDEN_OPTIONS)
        *
        len(DROPOUT_OPTIONS)
        *
        len(WEIGHT_DECAY_OPTIONS)
    )

    counter = 0

    for hidden_dims in HIDDEN_OPTIONS:

        for dropout in DROPOUT_OPTIONS:

            for weight_decay in (
                WEIGHT_DECAY_OPTIONS
            ):

                counter += 1

                print(
                    f"\n[{counter}/{total}] "
                    f"hidden={hidden_dims}, "
                    f"dropout={dropout}, "
                    f"weight_decay={weight_decay}"
                )

                result = (
                    evaluate_configuration(
                        X_dev,
                        time_dev,
                        event_dev,
                        hidden_dims,
                        dropout,
                        weight_decay,
                        device
                    )
                )

                print(
                    "CV C-index = "
                    f"{result['mean_cv_cindex']:.4f}"
                    " ± "
                    f"{result['std_cv_cindex']:.4f}"
                )

                print(
                    "Median best epoch =",
                    result[
                        "median_best_epoch"
                    ]
                )

                results.append(
                    result
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
# Parse hidden dimensions
# ============================================================

def parse_hidden_dims(text):

    text = (
        text
        .replace("(", "")
        .replace(")", "")
    )

    values = [
        value.strip()
        for value in text.split(",")
        if value.strip()
    ]

    return tuple(
        int(value)
        for value in values
    )


# ============================================================
# Final preprocessing
# ============================================================

def fit_final_preprocessing(
    X_dev,
    X_test
):

    # --------------------------------------------------------
    # Variable genes
    # --------------------------------------------------------

    variances = (
        X_dev.var(
            axis=0
        )
    )

    selected_genes = (
        variances
        .sort_values(
            ascending=False
        )
        .head(N_GENES)
        .index
    )

    X_dev = X_dev[
        selected_genes
    ]

    X_test = X_test[
        selected_genes
    ]

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=N_COMPONENTS
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

    return (
        X_dev_pca,
        X_test_pca,
        selected_genes,
        scaler,
        pca
    )


# ============================================================
# Train final model
# ============================================================

def train_final_model(
    X,
    times,
    events,
    hidden_dims,
    dropout,
    weight_decay,
    epochs,
    device
):

    set_seed(
        RANDOM_STATE
    )

    model = DeepSurv(
        input_dim=N_COMPONENTS,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    times_tensor = torch.tensor(
        times,
        dtype=torch.float32,
        device=device
    )

    events_tensor = torch.tensor(
        events,
        dtype=torch.float32,
        device=device
    )

    for epoch in range(
        epochs
    ):

        model.train()

        optimizer.zero_grad()

        risk = model(
            X_tensor
        )

        loss = cox_loss(
            risk,
            times_tensor,
            events_tensor
        )

        loss.backward()

        optimizer.step()

    return model


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    rna, manifest = load_data()

    # --------------------------------------------------------
    # DEVELOPMENT = train + val
    # TEST remains untouched
    # --------------------------------------------------------

    dev_manifest = manifest[
        manifest["split"].isin(
            [
                "train",
                "val"
            ]
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

    time_dev = (
        dev_manifest[
            "survival_time"
        ]
        .reset_index(drop=True)
    )

    event_dev = (
        dev_manifest[
            "event"
        ]
        .reset_index(drop=True)
    )

    time_test = (
        test_manifest[
            "survival_time"
        ]
        .reset_index(drop=True)
    )

    event_test = (
        test_manifest[
            "event"
        ]
        .reset_index(drop=True)
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
        + "=" * 60
    )

    print(
        "DEEPSURV CROSS-VALIDATION"
    )

    print(
        "=" * 60
    )

    results = search_hyperparameters(
        X_dev,
        time_dev,
        event_dev,
        device
    )

    results.to_csv(
        RESULTS_DIR
        / "deepsurv_cv_results.csv",
        index=False
    )

    # ========================================================
    # Best configuration
    # ========================================================

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
        results.head(10)
        .to_string(
            index=False
        )
    )

    best = results.iloc[0]

    hidden_dims = parse_hidden_dims(
        best["hidden_dims"]
    )

    dropout = float(
        best["dropout"]
    )

    weight_decay = float(
        best["weight_decay"]
    )

    epochs = int(
        best["median_best_epoch"]
    )

    print(
        "\nBest architecture:",
        hidden_dims
    )

    print(
        "Dropout:",
        dropout
    )

    print(
        "Weight decay:",
        weight_decay
    )

    print(
        "Training epochs:",
        epochs
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
    # Final preprocessing
    # ========================================================

    (
        X_dev_final,
        X_test_final,
        selected_genes,
        scaler,
        pca
    ) = fit_final_preprocessing(
        X_dev,
        X_test
    )

    # ========================================================
    # Final training
    # ========================================================

    model = train_final_model(
        X_dev_final,
        time_dev.values,
        event_dev.values,
        hidden_dims,
        dropout,
        weight_decay,
        epochs,
        device
    )

    # ========================================================
    # Final TEST
    # ========================================================

    test_cindex = calculate_cindex(
        model,
        X_test_final,
        time_test.values,
        event_test.values,
        device
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL DEEPSURV TEST RESULT"
    )

    print(
        "=" * 60
    )

    print(
        f"Test C-index: "
        f"{test_cindex:.4f}"
    )

    # ========================================================
    # Test predictions
    # ========================================================

    model.eval()

    with torch.no_grad():

        X_tensor = torch.tensor(
            X_test_final,
            dtype=torch.float32,
            device=device
        )

        risks = (
            model(
                X_tensor
            )
            .cpu()
            .numpy()
        )

    predictions = pd.DataFrame({

        "case_id":
            test_manifest[
                "case_id"
            ].values,

        "survival_time":
            time_test.values,

        "event":
            event_test.values,

        "risk_score":
            risks
    })

    predictions.to_csv(
        RESULTS_DIR
        / "deepsurv_test_predictions.csv",
        index=False
    )

    # ========================================================
    # Save model and preprocessing
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "hidden_dims":
                hidden_dims,

            "dropout":
                dropout,

            "n_components":
                N_COMPONENTS
        },
        RESULTS_DIR
        / "deepsurv_model.pt"
    )

    pd.Series(
        selected_genes,
        name="gene_id"
    ).to_csv(
        RESULTS_DIR
        / "selected_genes.csv",
        index=False
    )

    joblib.dump(
        scaler,
        RESULTS_DIR
        / "scaler.joblib"
    )

    joblib.dump(
        pca,
        RESULTS_DIR
        / "pca.joblib"
    )

    print(
        f"\nResults saved in:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()