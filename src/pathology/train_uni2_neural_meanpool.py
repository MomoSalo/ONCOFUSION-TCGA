from pathlib import Path
import copy
import random

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from lifelines.utils import concordance_index


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
    / "uni2_neural_meanpool"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

INPUT_DIM = 1536
EMBED_DIM = 128

MAX_TRAIN_PATCHES = 128

# Use all available patches during validation/test
MAX_EVAL_PATCHES = None

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100

VAL_EVERY = 5
PATIENCE = 6


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
# Neural mean-pooling model
# ============================================================

class NeuralMeanPool(nn.Module):

    def __init__(
        self,
        input_dim=1536,
        embed_dim=128
    ):
        super().__init__()

        # Learn a new representation
        # for each UNI2-h patch
        self.project = nn.Sequential(
            nn.Linear(
                input_dim,
                embed_dim
            ),
            nn.ReLU()
        )

        # Patient-level survival risk
        self.risk_head = nn.Linear(
            embed_dim,
            1
        )

    def forward(self, x):

        # x:
        # [N_patches, 1536]

        h = self.project(x)

        # h:
        # [N_patches, 128]

        # ----------------------------------------------------
        # NO ATTENTION
        #
        # Every patch has exactly equal weight.
        # ----------------------------------------------------

        slide_embedding = h.mean(
            dim=0
        )

        # [128] -> scalar
        risk = self.risk_head(
            slide_embedding
        ).squeeze()

        return (
            risk,
            slide_embedding
        )


# ============================================================
# Cohort
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
        cohort["event"].isin(
            [0, 1]
        )
    ].copy()

    cohort["event"] = (
        cohort["event"]
        .astype(int)
    )

    cohort = (
        cohort
        .drop_duplicates(
            "case_id"
        )
        .reset_index(
            drop=True
        )
    )

    return cohort


# ============================================================
# Load UNI2-h bag
# ============================================================

def load_bag(
    h5_path,
    max_patches=None,
    seed=None
):

    with h5py.File(
        h5_path,
        "r"
    ) as f:

        dataset = f[
            "features"
        ]

        # Expected:
        # [1, N, 1536]

        if (
            dataset.ndim != 3
            or
            dataset.shape[0] != 1
            or
            dataset.shape[2] != INPUT_DIM
        ):
            raise ValueError(
                f"Unexpected shape: "
                f"{dataset.shape}"
            )

        n_patches = (
            dataset.shape[1]
        )

        # ----------------------------------------------------
        # Random patch subset during training
        # ----------------------------------------------------

        if (
            max_patches is not None
            and
            n_patches > max_patches
        ):

            rng = np.random.default_rng(
                seed
            )

            indices = rng.choice(
                n_patches,
                size=max_patches,
                replace=False
            )

            indices = np.sort(
                indices
            )

            features = dataset[
                0,
                indices,
                :
            ]

        else:

            features = dataset[
                0,
                :,
                :
            ]

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# Cox loss
# ============================================================

def cox_loss(
    risks,
    times,
    events
):

    # Sort by decreasing survival time
    order = torch.argsort(
        times,
        descending=True
    )

    risks = risks[
        order
    ]

    times = times[
        order
    ]

    events = events[
        order
    ]

    log_cumulative_risk = (
        torch.logcumsumexp(
            risks,
            dim=0
        )
    )

    unique_times = (
        torch.unique_consecutive(
            times
        )
    )

    log_likelihood = torch.tensor(
        0.0,
        device=risks.device
    )

    n_events = events.sum()

    for time in unique_times:

        time_mask = (
            times == time
        )

        event_mask = (
            time_mask
            &
            (events == 1)
        )

        d = event_mask.sum()

        if d == 0:
            continue

        event_risk = (
            risks[
                event_mask
            ]
            .sum()
        )

        indices = torch.where(
            time_mask
        )[0]

        last_index = indices[
            -1
        ]

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
        n_events.clamp(
            min=1
        )
    )


# ============================================================
# One patient
# ============================================================

def patient_risk(
    model,
    h5_path,
    device,
    max_patches=None,
    seed=None
):

    features = load_bag(
        h5_path,
        max_patches=max_patches,
        seed=seed
    )

    x = torch.tensor(
        features,
        dtype=torch.float32,
        device=device
    )

    # Same preprocessing as Attention MIL
    x = F.normalize(
        x,
        p=2,
        dim=1
    )

    return model(x)


# ============================================================
# Train one epoch
# ============================================================

def train_epoch(
    model,
    cohort,
    optimizer,
    device,
    epoch
):

    model.train()

    optimizer.zero_grad()

    risks = []

    for _, row in cohort.iterrows():

        case_id = row[
            "case_id"
        ]

        # Exactly the same patch-sampling
        # strategy as Attention MIL
        seed = (
            RANDOM_SEED
            +
            epoch * 100_000
            +
            sum(
                ord(c)
                for c in case_id
            )
        )

        risk, _ = patient_risk(
            model,
            row[
                "uni2_h5_path"
            ],
            device,
            max_patches=
                MAX_TRAIN_PATCHES,
            seed=seed
        )

        risks.append(
            risk
        )

    risks = torch.stack(
        risks
    )

    times = torch.tensor(
        cohort[
            "survival_time"
        ].values,
        dtype=torch.float32,
        device=device
    )

    events = torch.tensor(
        cohort[
            "event"
        ].values,
        dtype=torch.float32,
        device=device
    )

    loss = cox_loss(
        risks,
        times,
        events
    )

    loss.backward()

    optimizer.step()

    return float(
        loss.detach().cpu()
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    cohort,
    device
):

    model.eval()

    risks = []

    with torch.no_grad():

        for _, row in cohort.iterrows():

            risk, _ = (
                patient_risk(
                    model,
                    row[
                        "uni2_h5_path"
                    ],
                    device,
                    max_patches=
                        MAX_EVAL_PATCHES
                )
            )

            risks.append(
                float(
                    risk.cpu()
                )
            )

    risks = np.asarray(
        risks
    )

    times = cohort[
        "survival_time"
    ].values

    events = cohort[
        "event"
    ].values

    cindex = concordance_index(
        times,
        -risks,
        events
    )

    return (
        cindex,
        risks
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "UNI2-h + NEURAL MEAN POOLING SURVIVAL"
    )
    print("=" * 70)

    print(
        "\nDevice:",
        device
    )

    cohort = load_cohort()

    train = (
        cohort[
            cohort["split"]
            == "train"
        ]
        .reset_index(
            drop=True
        )
    )

    val = (
        cohort[
            cohort["split"]
            == "val"
        ]
        .reset_index(
            drop=True
        )
    )

    test = (
        cohort[
            cohort["split"]
            == "test"
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTotal patients:",
        len(cohort)
    )

    for name, subset in [
        ("Train", train),
        ("Validation", val),
        ("Test", test)
    ]:

        print(
            f"\n{name}: "
            f"{len(subset)} patients"
        )

        print(
            "Deaths:",
            int(
                subset[
                    "event"
                ].sum()
            )
        )

        print(
            "Censored:",
            int(
                (
                    subset["event"]
                    == 0
                ).sum()
            )
        )

    # ========================================================
    # Model
    # ========================================================

    model = NeuralMeanPool(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=
            WEIGHT_DECAY
    )

    # ========================================================
    # Training
    # ========================================================

    best_val_cindex = -np.inf
    best_state = None
    best_epoch = 0

    patience_counter = 0

    history = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 70
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        loss = train_epoch(
            model,
            train,
            optimizer,
            device,
            epoch
        )

        if (
            epoch % VAL_EVERY
            != 0
        ):

            print(
                f"Epoch {epoch:03d}"
                f" | loss={loss:.4f}"
            )

            continue

        val_cindex, _ = evaluate(
            model,
            val,
            device
        )

        print(
            f"Epoch {epoch:03d}"
            f" | loss={loss:.4f}"
            f" | val C-index="
            f"{val_cindex:.4f}"
        )

        history.append(
            {
                "epoch":
                    epoch,

                "loss":
                    loss,

                "val_cindex":
                    val_cindex
            }
        )

        if (
            val_cindex
            >
            best_val_cindex
        ):

            best_val_cindex = (
                val_cindex
            )

            best_epoch = epoch

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No best model obtained."
        )

    # ========================================================
    # Restore best model
    # ========================================================

    model.load_state_dict(
        best_state
    )

    train_cindex, _ = evaluate(
        model,
        train,
        device
    )

    test_cindex, test_risks = (
        evaluate(
            model,
            test,
            device
        )
    )

    # ========================================================
    # Final results
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL UNI2-h NEURAL MEAN POOL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest epoch:",
        best_epoch
    )

    print(
        "Train C-index:",
        f"{train_cindex:.4f}"
    )

    print(
        "Validation C-index:",
        f"{best_val_cindex:.4f}"
    )

    print(
        "Test C-index:",
        f"{test_cindex:.4f}"
    )

    # ========================================================
    # Save model
    # ========================================================

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                INPUT_DIM,

            "embed_dim":
                EMBED_DIM,

            "best_epoch":
                best_epoch,

            "best_val_cindex":
                best_val_cindex,

            "test_cindex":
                test_cindex
        },
        RESULTS_DIR
        / "uni2_neural_meanpool.pt"
    )

    # ========================================================
    # Save training history
    # ========================================================

    pd.DataFrame(
        history
    ).to_csv(
        RESULTS_DIR
        / "training_history.csv",
        index=False
    )

    # ========================================================
    # Save test predictions
    # ========================================================

    predictions = pd.DataFrame(
        {
            "case_id":
                test[
                    "case_id"
                ].values,

            "survival_time":
                test[
                    "survival_time"
                ].values,

            "event":
                test[
                    "event"
                ].values,

            "risk_score":
                test_risks
        }
    )

    predictions.to_csv(
        RESULTS_DIR
        / "test_predictions.csv",
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()