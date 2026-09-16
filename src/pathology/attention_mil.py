from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pathology_features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pathology"
    / "pilot"
)


# ============================================================
# Configuration
# ============================================================

INPUT_DIM = 2048
EMBED_DIM = 256
ATTENTION_DIM = 128


# ============================================================
# Attention MIL
# ============================================================

class AttentionMIL(nn.Module):

    def __init__(
        self,
        input_dim=2048,
        embed_dim=256,
        attention_dim=128
    ):

        super().__init__()

        # ----------------------------------------------------
        # 1. Patch projection
        #
        # ResNet:
        # 2048
        #
        # MIL internal representation:
        # 256
        # ----------------------------------------------------

        self.patch_encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                embed_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                0.25
            )
        )

        # ----------------------------------------------------
        # 2. Gated attention
        # ----------------------------------------------------

        self.attention_V = nn.Sequential(

            nn.Linear(
                embed_dim,
                attention_dim
            ),

            nn.Tanh()
        )

        self.attention_U = nn.Sequential(

            nn.Linear(
                embed_dim,
                attention_dim
            ),

            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(
            attention_dim,
            1
        )

        # ----------------------------------------------------
        # 3. Survival-risk head
        # ----------------------------------------------------

        self.risk_head = nn.Linear(
            embed_dim,
            1
        )

    def forward(self, x):

        # ====================================================
        # x
        #
        # [N_patches, 2048]
        # ====================================================

        H = self.patch_encoder(
            x
        )

        # H:
        # [N_patches, 256]

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        A_V = self.attention_V(
            H
        )

        A_U = self.attention_U(
            H
        )

        # Gated attention
        A = A_V * A_U

        A = self.attention_w(
            A
        )

        # A:
        # [N_patches, 1]

        A = A.squeeze(
            dim=1
        )

        # ----------------------------------------------------
        # Softmax over patches
        #
        # sum(attention_weights) = 1
        # ----------------------------------------------------

        attention_weights = torch.softmax(
            A,
            dim=0
        )

        # ----------------------------------------------------
        # Weighted slide representation
        # ----------------------------------------------------

        slide_embedding = torch.sum(

            attention_weights.unsqueeze(1)
            * H,

            dim=0
        )

        # slide_embedding:
        # [256]

        # ----------------------------------------------------
        # Patient risk
        # ----------------------------------------------------

        risk = self.risk_head(
            slide_embedding
        )

        risk = risk.squeeze()

        return (
            risk,
            attention_weights,
            slide_embedding
        )


# ============================================================
# Find first feature bag
# ============================================================

def get_first_feature_folder():

    folders = [

        folder

        for folder in FEATURE_ROOT.iterdir()

        if (
            folder.is_dir()
            and
            (folder / "features.npy").exists()
        )
    ]

    if not folders:

        raise RuntimeError(
            f"No pathology features found in:\n"
            f"{FEATURE_ROOT}"
        )

    return sorted(
        folders
    )[0]


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("ATTENTION MIL PILOT")
    print("=" * 60)

    print(
        "\nDevice:",
        device
    )

    # ========================================================
    # Load one WSI bag
    # ========================================================

    feature_dir = (
        get_first_feature_folder()
    )

    slide_name = (
        feature_dir.name
    )

    features = np.load(
        feature_dir
        / "features.npy"
    )

    print(
        "\nSlide:",
        slide_name
    )

    print(
        "Feature matrix:",
        features.shape
    )

    # ========================================================
    # Convert to torch
    # ========================================================

    x = torch.tensor(
        features,
        dtype=torch.float32,
        device=device
    )

    # ========================================================
    # Model
    # ========================================================

    model = AttentionMIL(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        attention_dim=ATTENTION_DIM
    ).to(device)

    model.eval()

    # ========================================================
    # Forward pass
    # ========================================================

    with torch.no_grad():

        (
            risk,
            attention_weights,
            slide_embedding
        ) = model(
            x
        )

    attention = (
        attention_weights
        .cpu()
        .numpy()
    )

    # ========================================================
    # Results
    # ========================================================

    print(
        "\nPatch embeddings:",
        x.shape
    )

    print(
        "Slide embedding:",
        slide_embedding.shape
    )

    print(
        "Attention weights:",
        attention.shape
    )

    print(
        "Attention sum:",
        attention.sum()
    )

    print(
        "Risk score:",
        float(
            risk.cpu()
        )
    )

    # ========================================================
    # Load patch metadata
    # ========================================================

    index_path = (
        feature_dir
        / "feature_index.csv"
    )

    if index_path.exists():

        df = pd.read_csv(
            index_path
        )

    else:

        df = pd.DataFrame({

            "embedding_row":
                np.arange(
                    len(attention)
                )
        })

    # ========================================================
    # Add attention
    # ========================================================

    df["attention"] = attention

    df = df.sort_values(
        "attention",
        ascending=False
    )

    # ========================================================
    # Top patches
    # ========================================================

    print(
        "\nTop 10 patches:"
    )

    columns = [

        column

        for column in [
            "patch_file",
            "x_level0",
            "y_level0",
            "tissue_fraction",
            "attention"
        ]

        if column in df.columns
    ]

    print(
        df[
            columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / "pilot_attention.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nAttention saved to:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()