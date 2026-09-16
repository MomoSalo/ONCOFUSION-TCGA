from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    resnet50,
    ResNet50_Weights
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATCH_ROOT = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "pathology_patches"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pathology_features"
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32

# Keep 0 on Windows for now: simplest and safest.
NUM_WORKERS = 0


# ============================================================
# Patch dataset
# ============================================================

class PatchDataset(Dataset):

    def __init__(
        self,
        patch_dir,
        transform
    ):

        self.patch_dir = patch_dir

        self.patch_paths = sorted(
            patch_dir.glob("patch_*.png")
        )

        if len(self.patch_paths) == 0:
            raise RuntimeError(
                f"No patches found in:\n{patch_dir}"
            )

        self.transform = transform

    def __len__(self):

        return len(
            self.patch_paths
        )

    def __getitem__(
        self,
        index
    ):

        path = self.patch_paths[
            index
        ]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return (
            image,
            path.name
        )


# ============================================================
# Find first extracted slide
# ============================================================

def get_first_patch_folder():

    folders = [
        path
        for path in PATCH_ROOT.iterdir()
        if path.is_dir()
    ]

    if not folders:

        raise RuntimeError(
            f"No slide patch folder found in:\n"
            f"{PATCH_ROOT}"
        )

    return folders[0]


# ============================================================
# Build pretrained ResNet50 encoder
# ============================================================

def build_encoder(device):

    # --------------------------------------------------------
    # Pretrained ImageNet weights
    # --------------------------------------------------------

    weights = (
        ResNet50_Weights.DEFAULT
    )

    model = resnet50(
        weights=weights
    )

    # --------------------------------------------------------
    # Original ResNet:
    #
    # image
    #   ↓
    # convolutions
    #   ↓
    # global average pooling
    #   ↓
    # 2048-dimensional vector
    #   ↓
    # fc classifier (1000 classes)
    #
    # We remove the final classifier.
    # --------------------------------------------------------

    model.fc = nn.Identity()

    model = model.to(
        device
    )

    model.eval()

    # Correct preprocessing for these weights
    transform = weights.transforms()

    return (
        model,
        transform
    )


# ============================================================
# Extract embeddings
# ============================================================

def extract_embeddings(
    model,
    dataloader,
    device
):

    all_features = []
    all_filenames = []

    total_batches = len(
        dataloader
    )

    with torch.no_grad():

        for batch_index, (
            images,
            filenames
        ) in enumerate(
            dataloader,
            start=1
        ):

            images = images.to(
                device
            )

            # ----------------------------------------
            # Shape:
            #
            # images:
            # [B, 3, 224, 224]
            #
            # embeddings:
            # [B, 2048]
            # ----------------------------------------

            embeddings = model(
                images
            )

            embeddings = (
                embeddings
                .cpu()
                .numpy()
            )

            all_features.append(
                embeddings
            )

            all_filenames.extend(
                filenames
            )

            print(
                f"Batch "
                f"{batch_index}/"
                f"{total_batches}"
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    return (
        features,
        all_filenames
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("PATHOLOGY FEATURE EXTRACTION")
    print("=" * 60)

    print(
        "\nDevice:",
        device
    )

    # --------------------------------------------------------
    # Find our pilot slide
    # --------------------------------------------------------

    patch_dir = (
        get_first_patch_folder()
    )

    slide_name = (
        patch_dir.name
    )

    print(
        "\nSlide:",
        slide_name
    )

    # --------------------------------------------------------
    # Build encoder
    # --------------------------------------------------------

    model, transform = (
        build_encoder(
            device
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = PatchDataset(
        patch_dir,
        transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        "\nNumber of patches:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    (
        features,
        filenames
    ) = extract_embeddings(
        model,
        dataloader,
        device
    )

    print(
        "\nEmbedding matrix shape:",
        features.shape
    )

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / slide_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save embedding matrix
    # --------------------------------------------------------

    np.save(
        output_dir
        / "features.npy",
        features
    )

    # --------------------------------------------------------
    # Preserve relationship:
    #
    # row 0 → patch_0000.png
    # row 1 → patch_0001.png
    # ...
    # --------------------------------------------------------

    feature_index = pd.DataFrame({

        "embedding_row":
            np.arange(
                len(filenames)
            ),

        "patch_file":
            filenames
    })

    # --------------------------------------------------------
    # Add coordinates from patch metadata
    # --------------------------------------------------------

    metadata_path = (
        patch_dir
        / "patch_metadata.csv"
    )

    if metadata_path.exists():

        metadata = pd.read_csv(
            metadata_path
        )

        feature_index = (
            feature_index.merge(
                metadata,
                left_on="patch_file",
                right_on="file_name",
                how="left"
            )
        )

    feature_index.to_csv(
        output_dir
        / "feature_index.csv",
        index=False
    )

    # --------------------------------------------------------
    # Quick diagnostics
    # --------------------------------------------------------

    norms = np.linalg.norm(
        features,
        axis=1
    )

    print(
        "\nFeature dimension:",
        features.shape[1]
    )

    print(
        "Mean embedding norm:",
        round(
            float(
                norms.mean()
            ),
            3
        )
    )

    print(
        "Std embedding norm:",
        round(
            float(
                norms.std()
            ),
            3
        )
    )

    print(
        "\nSaved:"
    )

    print(
        output_dir
        / "features.npy"
    )

    print(
        output_dir
        / "feature_index.csv"
    )


if __name__ == "__main__":
    main()