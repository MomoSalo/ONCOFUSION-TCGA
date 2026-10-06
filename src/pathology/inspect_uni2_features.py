from pathlib import Path
import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNI_ROOT = (
    PROJECT_ROOT
    / "UNI2-h_features"
    / "TCGA"
)

h5_files = list(UNI_ROOT.rglob("*.h5"))

print("Number of H5 files:", len(h5_files))

if len(h5_files) == 0:
    raise RuntimeError("No .h5 files found.")

path = h5_files[0]

print("\nExample file:")
print(path.name)

with h5py.File(path, "r") as f:
    print("\nKeys:")
    print(list(f.keys()))

    features = f["features"][:]
    coords = f["coords"][:]

    print("\nFeatures shape:")
    print(features.shape)

    print("\nCoordinates shape:")
    print(coords.shape)

    print("\nFeature dtype:")
    print(features.dtype)

    print("\nFirst coordinate:")
    print(coords[0])