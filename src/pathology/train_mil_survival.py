from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from lifelines.utils import concordance_index

from attention_mil import AttentionMIL


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
    / "mil"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

MIN_PATCHES = 100

# During training:
# randomly select at most 128 patches/patient.
#
# Different patches are sampled at each epoch.
MAX_TRAIN_PATCHES = 128

# Validation/test:
# use the entire bag.
MAX_EVAL_PATCHES = 1000


# MIL architecture
INPUT_DIM = 2048
EMBED_DIM = 128
ATTENTION_DIM = 64

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100

# Validation is expensive, especially on CPU.
VAL_EVERY = 5

# 6 validation checks without improvement
# = approximately 30 epochs.
PATIENCE = 6


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Build usable WSI cohort
# ============================================================

def load_cohort():

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    qc = pd.read_csv(
        QC_PATH
    )

    # --------------------------------------------------------
    # Only WSI patients that:
    #
    # - completed processing
    # - have valid 2048-dim embeddings
    # - have >=100 patches
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

    # Valid survival information
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

    return cohort


# ============================================================
# Load one patient's ResNet feature bag
# ============================================================

def load_bag(
    case_id,
    max_patches=None,
    seed=None
):

    feature_path = (
        FEATURE_ROOT
        / case_id
        / "features.npy"
    )

    # mmap_mode avoids loading everything
    # unnecessarily before selecting patches.
    features = np.load(
        feature_path,
        mmap_mode="r"
    )

    n_patches = (
        features.shape[0]
    )

    # --------------------------------------------------------
    # Random patch sampling
    # --------------------------------------------------------

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

        features = features[
            indices
        ]

    # Force independent float32 array
    features = np.asarray(
        features,
        dtype=np.float32
    )

    return features


# ============================================================
# Cox partial likelihood loss
# ============================================================

def cox_loss(
    risks,
    times,
    events
):
    """
    Negative Cox partial log-likelihood.

    Uses Breslow approximation for tied event times.
    """

    # --------------------------------------------------------
    # Sort longest survival -> shortest survival
    # --------------------------------------------------------

    order = torch.argsort(
        times,
        descending=True
    )

    risks = risks[order]
    times = times[order]
    events = events[order]

    # --------------------------------------------------------
    # Risk-set denominator
    # --------------------------------------------------------

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

        # Events occurring at this time
        event_risk = (
            risks[event_mask]
            .sum()
        )

        indices = torch.where(
            time_mask
        )[0]

        # Risk set = patients whose time >= current time
        last_index = indices[-1]

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
# Forward one patient
# ============================================================

def patient_risk(
    model,
    case_id,
    device,
    max_patches,
    seed=None
):

    features = load_bag(
        case_id,
        max_patches=max_patches,
        seed=seed
    )

    x = torch.tensor(
        features,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Normalize each ResNet embedding
    #
    # shape remains:
    # [N_patches, 2048]
    # --------------------------------------------------------

    x = F.normalize(
        x,
        p=2,
        dim=1
    )

    risk, attention, slide_embedding = (
        model(x)
    )

    return (
        risk,
        attention,
        slide_embedding
    )


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

    # --------------------------------------------------------
    # One risk score per patient
    # --------------------------------------------------------

    for i, row in cohort.iterrows():

        case_id = row[
            "case_id"
        ]

        # Different deterministic patch sample
        # each epoch + patient
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

        risk, _, _ = patient_risk(
            model,
            case_id,
            device,
            max_patches=MAX_TRAIN_PATCHES,
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
# Evaluate C-index
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

            risk, _, _ = patient_risk(
                model,
                row["case_id"],
                device,
                max_patches=MAX_EVAL_PATCHES
            )

            risks.append(
                float(
                    risk.cpu()
                )
            )

    times = cohort[
        "survival_time"
    ].values

    events = cohort[
        "event"
    ].values

    # Deep survival convention:
    # larger model output = higher risk.
    #
    # lifelines concordance_index:
    # larger predicted value = longer survival.
    #
    # Therefore use -risk.
    score = concordance_index(
        times,
        -np.asarray(
            risks
        ),
        events
    )

    return (
        score,
        np.asarray(
            risks
        )
    )


# ============================================================
# Save attention for test patients
# ============================================================

def save_test_attention(
    model,
    test,
    device
):

    attention_dir = (
        RESULTS_DIR
        / "attention"
    )

    attention_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model.eval()

    with torch.no_grad():

        for _, row in test.iterrows():

            case_id = row[
                "case_id"
            ]

            risk, attention, _ = (
                patient_risk(
                    model,
                    case_id,
                    device,
                    max_patches=MAX_EVAL_PATCHES
                )
            )

            attention = (
                attention
                .cpu()
                .numpy()
            )

            # --------------------------------------------
            # Patch coordinates
            # --------------------------------------------

            patch_index_path = (
                FEATURE_ROOT
                / case_id
                / "patch_index.csv"
            )

            patch_index = pd.read_csv(
                patch_index_path
            )

            # Evaluation uses all patches.
            # Rows correspond directly to embeddings.
            patch_index = (
                patch_index
                .iloc[
                    :len(attention)
                ]
                .copy()
            )

            patch_index[
                "attention"
            ] = attention

            patch_index[
                "risk_score"
            ] = float(
                risk.cpu()
            )

            patch_index = (
                patch_index
                .sort_values(
                    "attention",
                    ascending=False
                )
            )

            patch_index.to_csv(
                attention_dir
                / f"{case_id}.csv",
                index=False
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

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("WSI ATTENTION MIL SURVIVAL")
    print("=" * 70)

    print(
        "\nDevice:",
        device
    )

    # ========================================================
    # Cohort
    # ========================================================

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
        "\nUsable WSI patients:",
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

    model = AttentionMIL(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        attention_dim=ATTENTION_DIM
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
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

        # --------------------------------------------
        # Validation every few epochs
        # --------------------------------------------

        if (
            epoch % VAL_EVERY
            != 0
        ):

            print(
                f"Epoch {epoch:03d} "
                f"| loss={loss:.4f}"
            )

            continue

        val_cindex, _ = evaluate(
            model,
            val,
            device
        )

        print(
            f"Epoch {epoch:03d} "
            f"| loss={loss:.4f} "
            f"| val C-index="
            f"{val_cindex:.4f}"
        )

        history.append({

            "epoch":
                epoch,

            "loss":
                loss,

            "val_cindex":
                val_cindex
        })

        # --------------------------------------------
        # Early stopping
        # --------------------------------------------

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

    # ========================================================
    # Restore best model
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "No valid model state obtained."
        )

    model.load_state_dict(
        best_state
    )

    # ========================================================
    # FINAL TEST
    #
    # Test has never been used for training/selection.
    # ========================================================

    test_cindex, test_risks = (
        evaluate(
            model,
            test,
            device
        )
    )

    train_cindex, _ = evaluate(
        model,
        train,
        device
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL WSI MIL RESULT"
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

            "attention_dim":
                ATTENTION_DIM,

            "min_patches":
                MIN_PATCHES,

            "best_epoch":
                best_epoch,

            "best_val_cindex":
                best_val_cindex,

            "test_cindex":
                test_cindex
        },
        RESULTS_DIR
        / "mil_survival.pt"
    )

    # ========================================================
    # Save history
    # ========================================================

    pd.DataFrame(
        history
    ).to_csv(
        RESULTS_DIR
        / "training_history.csv",
        index=False
    )

    # ========================================================
    # Save predictions
    # ========================================================

    predictions = pd.DataFrame({

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
    })

    predictions.to_csv(
        RESULTS_DIR
        / "test_predictions.csv",
        index=False
    )

    # ========================================================
    # Attention maps
    # ========================================================

    print(
        "\nSaving test-patient "
        "attention weights..."
    )

    save_test_attention(
        model,
        test,
        device
    )

    print(
        f"\nResults saved in:\n"
        f"{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()