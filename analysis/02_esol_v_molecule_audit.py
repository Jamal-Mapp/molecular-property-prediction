# ============================================================
# ESOL-V Molecule Audit
# Missing Values, Duplicate Molecules, and Target Integrity
# ============================================================

import re

import numpy as np
import pandas as pd
from datasets import load_dataset


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET_NAME = "molvision/ESOL-V-SMILES-0"


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

dataset = load_dataset(DATASET_NAME, split="train")

print("=" * 72)
print("ESOL-V MOLECULE AUDIT")
print("=" * 72)

print(f"Dataset      : {DATASET_NAME}")
print(f"Samples      : {len(dataset)}")
print(f"Columns      : {dataset.column_names}")


# ------------------------------------------------------------
# Parse Regression Target
# ------------------------------------------------------------

def extract_logs(answer: str) -> float:
    """
    Extract the numeric logS target from the MolVision answer format.

    Example
    -------
    '<float>-2.484</float>' -> -2.484
    """
    if answer is None:
        return np.nan

    match = re.search(
        r"<float>\s*([-+]?\d*\.?\d+)\s*</float>",
        answer,
    )

    return float(match.group(1)) if match else np.nan


# ------------------------------------------------------------
# Build Analysis Table
# ------------------------------------------------------------

records = []

for index, sample in enumerate(dataset):
    records.append(
        {
            "index": index,
            "smiles": sample["TargetMolecule"],
            "logS": extract_logs(sample["Answer"]),
            "sample_method": sample["SampleMethod"],
            "sample_num": sample["SampleNum"],
            "sample_rep": sample["SampleRep"],
            "has_image": sample["image"] is not None,
        }
    )

df = pd.DataFrame(records)


# ------------------------------------------------------------
# Dataset Integrity
# ------------------------------------------------------------

total_records = len(df)
unique_smiles = df["smiles"].nunique()
duplicate_rows = df["smiles"].duplicated().sum()

missing_smiles = df["smiles"].isna().sum()
missing_targets = df["logS"].isna().sum()
missing_images = (~df["has_image"]).sum()

print("\n" + "=" * 72)
print("DATASET INTEGRITY")
print("=" * 72)

print(f"Total records             : {total_records}")
print(f"Unique SMILES             : {unique_smiles}")
print(f"Duplicate rows            : {duplicate_rows}")

print(f"\nMissing SMILES            : {missing_smiles}")
print(f"Missing logS targets      : {missing_targets}")
print(f"Missing molecular images  : {missing_images}")


# ------------------------------------------------------------
# Target Statistics
# ------------------------------------------------------------

print("\n" + "=" * 72)
print("TARGET STATISTICS")
print("=" * 72)

print(df["logS"].describe().to_string())


# ------------------------------------------------------------
# Duplicate Molecule Check
# ------------------------------------------------------------

duplicates = (
    df[df["smiles"].duplicated(keep=False)]
    .sort_values(["smiles", "index"])
)

print("\n" + "=" * 72)
print("DUPLICATE MOLECULE CHECK")
print("=" * 72)

if duplicates.empty:
    print("No duplicate SMILES were found.")
else:
    print(
        f"Found {len(duplicates)} rows belonging to duplicated "
        f"SMILES groups."
    )
    print()
    print(
        duplicates[
            ["index", "smiles", "logS"]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# Solubility Extremes
# ------------------------------------------------------------

print("\n" + "=" * 72)
print("5 LEAST SOLUBLE MOLECULES")
print("=" * 72)

print(
    df.nsmallest(5, "logS")[
        ["index", "smiles", "logS"]
    ].to_string(index=False)
)

print("\n" + "=" * 72)
print("5 MOST SOLUBLE MOLECULES")
print("=" * 72)

print(
    df.nlargest(5, "logS")[
        ["index", "smiles", "logS"]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Final Status
# ------------------------------------------------------------

print("\n" + "=" * 72)
print("AUDIT SUMMARY")
print("=" * 72)

if missing_smiles == 0 and missing_targets == 0 and missing_images == 0:
    print("All records contain SMILES, logS targets, and molecular images.")
else:
    print("One or more records contain missing values.")

if duplicate_rows > 0:
    print(
        "Duplicate molecular records are present and should be "
        "considered during train/validation/test splitting."
    )
else:
    print("No duplicate SMILES were detected.")

print("=" * 72)


# ------------------------------------------------------------
# Research Observation
# ------------------------------------------------------------
#
# All 220 records contain valid target values, SMILES representations,
# and molecular images.
#
# However, only 196 exact SMILES strings are unique. Duplicate molecular
# records therefore represent a potential source of train-test leakage
# if rows are randomly split without accounting for molecular identity.
#
# Canonical molecular identity is examined separately in the scaffold
# analysis.
