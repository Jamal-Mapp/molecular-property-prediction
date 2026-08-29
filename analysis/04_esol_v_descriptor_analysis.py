# ============================================================
# ESOL-V Chemical Descriptor Analysis
# RDKit Molecular Descriptors and Correlation with logS
# ============================================================

# !pip -q install datasets rdkit

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET_NAME = "molvision/ESOL-V-SMILES-0"


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

dataset = load_dataset(DATASET_NAME, split="train")

print("=" * 72)
print("ESOL-V CHEMICAL DESCRIPTOR ANALYSIS")
print("=" * 72)

print(f"Dataset      : {DATASET_NAME}")
print(f"Samples      : {len(dataset)}")


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
    match = re.search(
        r"<float>\s*([-+]?\d*\.?\d+)\s*</float>",
        answer,
    )

    return float(match.group(1)) if match else np.nan


# ------------------------------------------------------------
# Calculate Molecular Descriptors
# ------------------------------------------------------------

def calculate_descriptors(smiles: str):
    """
    Calculate a set of interpretable RDKit molecular descriptors.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "AromaticRings": Lipinski.NumAromaticRings(mol),
        "HeavyAtoms": Lipinski.HeavyAtomCount(mol),
    }


# ------------------------------------------------------------
# Build Descriptor Table
# ------------------------------------------------------------

records = []
invalid_smiles = []

for index, sample in enumerate(dataset):
    smiles = sample["TargetMolecule"]
    logs = extract_logs(sample["Answer"])

    descriptors = calculate_descriptors(smiles)

    if descriptors is None:
        invalid_smiles.append((index, smiles))
        continue

    row = {
        "index": index,
        "smiles": smiles,
        "logS": logs,
    }

    row.update(descriptors)
    records.append(row)

descriptor_df = pd.DataFrame(records)


# ------------------------------------------------------------
# Descriptor Extraction Summary
# ------------------------------------------------------------

print("\n" + "=" * 72)
print("DESCRIPTOR EXTRACTION")
print("=" * 72)

print(f"Valid molecules   : {len(descriptor_df)}")
print(f"Invalid SMILES    : {len(invalid_smiles)}")

if invalid_smiles:
    print("\nInvalid molecules:")
    for index, smiles in invalid_smiles:
        print(f"{index}: {smiles}")

print("\nFirst 5 descriptor rows:")
print(
    descriptor_df.head().to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Descriptor Summary Statistics
# ------------------------------------------------------------

descriptor_columns = [
    "MolWt",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "AromaticRings",
    "HeavyAtoms",
]

print("\n" + "=" * 72)
print("DESCRIPTOR SUMMARY STATISTICS")
print("=" * 72)

print(
    descriptor_df[
        descriptor_columns
    ]
    .describe()
    .T
    .to_string()
)


# ------------------------------------------------------------
# Correlation with logS
# ------------------------------------------------------------

correlations = (
    descriptor_df[
        ["logS"] + descriptor_columns
    ]
    .corr(numeric_only=True)["logS"]
    .drop("logS")
    .sort_values()
)

correlation_table = pd.DataFrame(
    {
        "Descriptor": correlations.index,
        "Correlation with logS": correlations.values,
    }
)

print("\n" + "=" * 72)
print("CORRELATION WITH logS")
print("=" * 72)

print(
    correlation_table.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Correlation Bar Chart
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    correlation_table["Descriptor"],
    correlation_table["Correlation with logS"],
)

plt.axvline(
    0,
    linewidth=1,
)

plt.xlabel("Pearson Correlation with logS")
plt.ylabel("Molecular Descriptor")
plt.title("ESOL-V Molecular Descriptor Correlation with Solubility")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# LogP vs logS
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    descriptor_df["LogP"],
    descriptor_df["logS"],
    alpha=0.7,
)

plt.xlabel("LogP")
plt.ylabel("Measured log Solubility (logS)")
plt.title("ESOL-V LogP vs Measured Solubility")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Molecular Weight vs logS
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    descriptor_df["MolWt"],
    descriptor_df["logS"],
    alpha=0.7,
)

plt.xlabel("Molecular Weight")
plt.ylabel("Measured log Solubility (logS)")
plt.title("ESOL-V Molecular Weight vs Measured Solubility")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# TPSA vs logS
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    descriptor_df["TPSA"],
    descriptor_df["logS"],
    alpha=0.7,
)

plt.xlabel("Topological Polar Surface Area (TPSA)")
plt.ylabel("Measured log Solubility (logS)")
plt.title("ESOL-V TPSA vs Measured Solubility")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Research Observation
# ------------------------------------------------------------
#
# Calculated LogP shows the strongest individual linear relationship
# with measured solubility in this ESOL-V subset.
#
# Molecular weight, aromatic ring count, heavy atom count, and total
# ring count also show moderate negative correlations with logS.
#
# In contrast, individual polarity and hydrogen-bond descriptors show
# comparatively weaker correlations when examined independently.
#
# These observations establish a chemically interpretable baseline for
# later prediction experiments using more complex molecular representations.


print("\n" + "=" * 72)
print("CHEMICAL DESCRIPTOR ANALYSIS COMPLETE")
print("=" * 72)
