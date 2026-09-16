from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

RNA_PATH = DATA_DIR / "rna_log2_tpm.csv"
MANIFEST_PATH = DATA_DIR / "patient_manifest_split.csv"

OUTPUT_DIR = DATA_DIR / "rna_preprocessed"


# ============================================================
# Configuration
# ============================================================

N_GENES = 2000
N_PCA_COMPONENTS = 100


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

    print("RNA matrix:", rna.shape)
    print("Manifest:", manifest.shape)

    return rna, manifest


# ============================================================
# Match RNA matrix with manifest
# ============================================================

def align_patients(rna, manifest):

    available_patients = set(rna.index)

    manifest = manifest[
        manifest["case_id"].isin(
            available_patients
        )
    ].copy()

    rna = rna.loc[
        manifest["case_id"]
    ]

    print(
        "\nMatched patients:",
        len(manifest)
    )

    return rna, manifest


# ============================================================
# Split
# ============================================================

def split_data(rna, manifest):

    train_ids = manifest[
        manifest["split"] == "train"
    ]["case_id"]

    val_ids = manifest[
        manifest["split"] == "val"
    ]["case_id"]

    test_ids = manifest[
        manifest["split"] == "test"
    ]["case_id"]

    X_train = rna.loc[train_ids]
    X_val = rna.loc[val_ids]
    X_test = rna.loc[test_ids]

    return X_train, X_val, X_test


# ============================================================
# Select variable genes
# ============================================================

def select_variable_genes(
    X_train,
    X_val,
    X_test,
    n_genes=2000
):

    # IMPORTANT:
    # variance computed ONLY on training data

    variances = X_train.var(axis=0)

    genes = (
        variances
        .sort_values(ascending=False)
        .head(n_genes)
        .index
    )

    X_train = X_train[genes]
    X_val = X_val[genes]
    X_test = X_test[genes]

    print(
        f"\nSelected genes: {len(genes)}"
    )

    return (
        X_train,
        X_val,
        X_test,
        genes
    )


# ============================================================
# Standardization
# ============================================================

def standardize(
    X_train,
    X_val,
    X_test
):

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


# ============================================================
# PCA
# ============================================================

def apply_pca(
    X_train,
    X_val,
    X_test,
    n_components=100
):

    # Cannot have more components than
    # number of training patients
    max_components = min(
        n_components,
        X_train.shape[0],
        X_train.shape[1]
    )

    pca = PCA(
        n_components=max_components
    )

    # Fit ONLY on train
    train_pca = pca.fit_transform(
        X_train
    )

    val_pca = pca.transform(
        X_val
    )

    test_pca = pca.transform(
        X_test
    )

    print(
        "\nPCA components:",
        max_components
    )

    print(
        "Explained variance:",
        round(
            pca.explained_variance_ratio_.sum(),
            3
        )
    )

    return (
        train_pca,
        val_pca,
        test_pca,
        pca
    )


# ============================================================
# Convert to DataFrames
# ============================================================

def to_dataframe(
    array,
    patient_ids
):

    columns = [
        f"PC{i+1}"
        for i in range(array.shape[1])
    ]

    return pd.DataFrame(
        array,
        index=patient_ids,
        columns=columns
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------
    # Load
    # ----------------------------------------

    rna, manifest = load_data()

    rna, manifest = align_patients(
        rna,
        manifest
    )

    # ----------------------------------------
    # Split
    # ----------------------------------------

    X_train, X_val, X_test = split_data(
        rna,
        manifest
    )

    print(
        "\nTrain:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    # ----------------------------------------
    # Variable genes
    # ----------------------------------------

    (
        X_train,
        X_val,
        X_test,
        genes
    ) = select_variable_genes(
        X_train,
        X_val,
        X_test,
        N_GENES
    )

    # Save selected genes
    pd.Series(
        genes,
        name="gene_id"
    ).to_csv(
        OUTPUT_DIR / "selected_genes.csv",
        index=False
    )

    # ----------------------------------------
    # Standardization
    # ----------------------------------------

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = standardize(
        X_train,
        X_val,
        X_test
    )

    # ----------------------------------------
    # PCA
    # ----------------------------------------

    (
        train_pca,
        val_pca,
        test_pca,
        pca
    ) = apply_pca(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        N_PCA_COMPONENTS
    )

    # ----------------------------------------
    # DataFrames
    # ----------------------------------------

    train_df = to_dataframe(
        train_pca,
        X_train.index
    )

    val_df = to_dataframe(
        val_pca,
        X_val.index
    )

    test_df = to_dataframe(
        test_pca,
        X_test.index
    )

    # ----------------------------------------
    # Save
    # ----------------------------------------

    train_df.to_csv(
        OUTPUT_DIR / "X_train_rna.csv"
    )

    val_df.to_csv(
        OUTPUT_DIR / "X_val_rna.csv"
    )

    test_df.to_csv(
        OUTPUT_DIR / "X_test_rna.csv"
    )

    # Save PCA statistics
    np.save(
        OUTPUT_DIR
        / "pca_explained_variance.npy",
        pca.explained_variance_ratio_
    )

    print(
        "\nRNA preprocessing complete."
    )

    print(
        "\nFiles saved in:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()