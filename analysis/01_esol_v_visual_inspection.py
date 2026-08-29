# ============================================================
# ESOL-V Visual Dataset Inspection
# ============================================================

import re

import matplotlib.pyplot as plt
from datasets import load_dataset


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET_NAME = "molvision/ESOL-V-SMILES-0"
NUM_SAMPLES = 6


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

dataset = load_dataset(DATASET_NAME, split="train")

print("=" * 72)
print("ESOL-V DATASET")
print("=" * 72)

print(f"Dataset      : {DATASET_NAME}")
print(f"Samples      : {len(dataset)}")
print(f"Columns      : {dataset.column_names}")


# ------------------------------------------------------------
# Parse Regression Target
# ------------------------------------------------------------

def extract_logs(answer: str):
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

    return float(match.group(1)) if match else None


# ------------------------------------------------------------
# Visualize Example Molecules
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10),
)

axes = axes.flatten()

for index, ax in enumerate(axes[:NUM_SAMPLES]):
    sample = dataset[index]

    image = sample["image"]
    smiles = sample["TargetMolecule"]
    logs = extract_logs(sample["Answer"])

    ax.imshow(image)
    ax.axis("off")

    # Wrap long SMILES strings for readability.
    if len(smiles) > 45:
        smiles_display = smiles[:45] + "\n" + smiles[45:90]
    else:
        smiles_display = smiles

    ax.set_title(
    f"Molecule {index + 1}\n"
    f"logS = {logs:.3f}\n"
    f"{smiles_display}",
    fontsize=10,
    )

plt.suptitle(
    "ESOL-V Molecular Structures and Measured Solubility",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Dataset Summary
# ------------------------------------------------------------

targets = [
    extract_logs(answer)
    for answer in dataset["Answer"]
]

valid_targets = [
    value
    for value in targets
    if value is not None
]

print("\n" + "=" * 72)
print("DATASET SUMMARY")
print("=" * 72)

print(f"Total records         : {len(dataset)}")
print(f"Valid logS targets    : {len(valid_targets)}")
print(f"Image resolution      : {dataset[0]['image'].size}")
print(f"Representation        : {dataset[0]['SampleRep']}")
print(f"Sampling method       : {dataset[0]['SampleMethod']}")
print(f"Minimum logS          : {min(valid_targets):.3f}")
print(f"Maximum logS          : {max(valid_targets):.3f}")
