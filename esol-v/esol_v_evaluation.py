"""
ESOL-V Regression Reproduction
==============================

Reproduction of the MolVision ESOL-V molecular-property regression
experiment using the SMILES 2-shot scaffold condition.

Dataset:
    molvision/ESOL-V-SMILES-2

Model:
    Qwen/Qwen-VL-Chat-Int4

Task:
    Predict aqueous solubility (logS) from the dataset-provided prompt.

Experimental condition:
    - 220 samples
    - SMILES representation
    - 2-shot scaffold prompting
    - Text-only model input
    - Molecular images are not passed to the model
    - Predictions are parsed from newly generated tokens only

Evaluation metrics:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R²
    - Pearson correlation

The script checkpoints after every sample so interrupted cluster runs
can be resumed without repeating completed inference.
"""

from __future__ import annotations

import json
import platform
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

DATASET_NAME = "molvision/ESOL-V-SMILES-2"
MODEL_NAME = "Qwen/Qwen-VL-Chat-Int4"

EXPECTED_SAMPLES = 220
SEED = 1234

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_FILE = RESULTS_DIR / "esol_v_checkpoint.csv"
FINAL_FILE = RESULTS_DIR / "esol_v_predictions.csv"
SUMMARY_FILE = RESULTS_DIR / "esol_v_summary.csv"
UNPARSEABLE_FILE = RESULTS_DIR / "esol_v_unparseable.csv"
METADATA_FILE = RESULTS_DIR / "esol_v_metadata.json"


# =============================================================================
# Reproducibility
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# Regression-output parsing
# =============================================================================

# MolVision ESOL-V responses encode the predicted value inside:
#
#     <float>VALUE</float>
#
# Scientific notation and signed decimal values are supported.

FLOAT_PATTERN = re.compile(
    r"<float>\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*</float>",
    re.IGNORECASE,
)


def extract_float(text: str | None) -> float | None:
    """
    Extract the first numeric value enclosed in <float>...</float>.

    Parameters
    ----------
    text:
        Model output or dataset answer.

    Returns
    -------
    float or None
        Parsed floating-point value if a valid tag is found.
    """
    if text is None:
        return None

    match = FLOAT_PATTERN.search(str(text))

    if match is None:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


# =============================================================================
# Dataset
# =============================================================================

print("=" * 80)
print("ESOL-V REGRESSION REPRODUCTION")
print("=" * 80)

print(f"\nDataset: {DATASET_NAME}")
print(f"Model:   {MODEL_NAME}")

print("\nLoading dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

print(f"Loaded samples: {len(dataset)}")

if len(dataset) != EXPECTED_SAMPLES:
    raise RuntimeError(
        f"Expected {EXPECTED_SAMPLES} samples, "
        f"but dataset contains {len(dataset)}."
    )

print("Dataset size verified.")


# =============================================================================
# Environment
# =============================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this reproduction. "
        "Run the script on a GPU-enabled environment."
    )

device = torch.device("cuda")

print("\nGPU environment")
print("-" * 80)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU:            {torch.cuda.get_device_name(0)}")
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA version:   {torch.version.cuda}")


# =============================================================================
# Output directory
# =============================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Resume support
# =============================================================================

if CHECKPOINT_FILE.exists():
    checkpoint_df = pd.read_csv(CHECKPOINT_FILE)

    if "dataset_index" not in checkpoint_df.columns:
        raise RuntimeError(
            f"Checkpoint exists but does not contain 'dataset_index': "
            f"{CHECKPOINT_FILE}"
        )

    checkpoint_df = (
        checkpoint_df
        .drop_duplicates(subset=["dataset_index"], keep="last")
        .sort_values("dataset_index")
        .reset_index(drop=True)
    )

    records = checkpoint_df.to_dict("records")
    completed_indices = set(checkpoint_df["dataset_index"].astype(int))

    print(
        f"\nResuming from checkpoint: "
        f"{len(completed_indices)}/{EXPECTED_SAMPLES} samples complete."
    )

else:
    records = []
    completed_indices = set()

    print("\nNo checkpoint found. Starting a new run.")


# =============================================================================
# Model
# =============================================================================

print("\nLoading model...")

model_load_start = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
).eval()

model_load_seconds = time.time() - model_load_start

print(f"Model loaded in {model_load_seconds / 60:.2f} minutes.")


# =============================================================================
# Inference
# =============================================================================

session_start = time.time()

for dataset_index in range(EXPECTED_SAMPLES):

    if dataset_index in completed_indices:
        continue

    row = dataset[dataset_index]

    question = str(row["Question"])
    ground_truth_raw = str(row["Answer"])
    ground_truth = extract_float(ground_truth_raw)

    if ground_truth is None:
        raise RuntimeError(
            f"Could not parse ground-truth value for sample "
            f"{dataset_index}: {ground_truth_raw!r}"
        )

    print(
        f"\n[{dataset_index + 1:03d}/{EXPECTED_SAMPLES}] "
        f"Running inference..."
    )

    inference_start = time.time()

    prediction = None
    generated_continuation = ""
    error = ""

    try:
        # The dataset-provided Question field is passed directly to the model.
        # Molecular structure images are intentionally not included in this
        # SMILES-only experimental condition.
        inputs = tokenizer(
            question,
            return_tensors="pt",
        ).to(device)

        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
            )

        # Decode only tokens generated after the input prompt.
        generated_ids = output_ids[0][input_length:]

        generated_continuation = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        prediction = extract_float(generated_continuation)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    inference_seconds = time.time() - inference_start

    absolute_error = (
        abs(prediction - ground_truth)
        if prediction is not None
        else None
    )

    record = {
        "dataset_index": dataset_index,
        "ground_truth_raw": ground_truth_raw,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "absolute_error": absolute_error,
        "question": question,
        "generated_continuation": generated_continuation,
        "inference_seconds": inference_seconds,
        "error": error,
        "image_passed_to_model": False,
    }

    records.append(record)
    completed_indices.add(dataset_index)

    # Save after every sample so long-running jobs can safely resume.
    pd.DataFrame(records).to_csv(
        CHECKPOINT_FILE,
        index=False,
    )

    if prediction is None:
        print(
            f"  Ground truth: {ground_truth:.6f}\n"
            f"  Prediction:   UNPARSEABLE\n"
            f"  Time:         {inference_seconds:.2f} s"
        )
    else:
        print(
            f"  Ground truth: {ground_truth:.6f}\n"
            f"  Prediction:   {prediction:.6f}\n"
            f"  Abs. error:   {absolute_error:.6f}\n"
            f"  Time:         {inference_seconds:.2f} s"
        )


session_seconds = time.time() - session_start


# =============================================================================
# Final predictions
# =============================================================================

results_df = pd.DataFrame(records)

results_df = (
    results_df
    .drop_duplicates(subset=["dataset_index"], keep="last")
    .sort_values("dataset_index")
    .reset_index(drop=True)
)

results_df.to_csv(
    FINAL_FILE,
    index=False,
)


# =============================================================================
# Evaluation
# =============================================================================

valid_df = results_df[
    results_df["prediction"].notna()
    & results_df["ground_truth"].notna()
].copy()

unparseable_df = results_df[
    results_df["prediction"].isna()
].copy()

unparseable_df.to_csv(
    UNPARSEABLE_FILE,
    index=False,
)

parsed_count = len(valid_df)
unparseable_count = len(unparseable_df)

parse_rate = (
    parsed_count / EXPECTED_SAMPLES
    if EXPECTED_SAMPLES
    else float("nan")
)

if parsed_count == 0:
    raise RuntimeError(
        "No model predictions could be parsed. "
        "Regression metrics cannot be calculated."
    )

y_true = valid_df["ground_truth"].astype(float).to_numpy()
y_pred = valid_df["prediction"].astype(float).to_numpy()

mae = mean_absolute_error(
    y_true,
    y_pred,
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred,
    )
)

r2 = r2_score(
    y_true,
    y_pred,
)

if parsed_count >= 2:
    pearson_r, pearson_p = pearsonr(
        y_true,
        y_pred,
    )
else:
    pearson_r = float("nan")
    pearson_p = float("nan")


# =============================================================================
# Timing
# =============================================================================

mean_inference_seconds = (
    results_df["inference_seconds"]
    .dropna()
    .astype(float)
    .mean()
)

total_recorded_inference_seconds = (
    results_df["inference_seconds"]
    .dropna()
    .astype(float)
    .sum()
)


# =============================================================================
# Summary
# =============================================================================

summary = {
    "dataset": DATASET_NAME,
    "model": MODEL_NAME,
    "expected_samples": EXPECTED_SAMPLES,
    "completed_samples": len(results_df),
    "parsed_samples": parsed_count,
    "parse_rate": parse_rate,
    "unparseable_samples": unparseable_count,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "pearson_r": pearson_r,
    "pearson_p": pearson_p,
    "model_load_seconds": model_load_seconds,
    "session_seconds": session_seconds,
    "mean_inference_seconds": mean_inference_seconds,
    "total_recorded_inference_seconds": total_recorded_inference_seconds,
    "image_passed_to_model": False,
}

pd.DataFrame([summary]).to_csv(
    SUMMARY_FILE,
    index=False,
)


# =============================================================================
# Reproducibility metadata
# =============================================================================

metadata = {
    "experiment": "ESOL-V regression reproduction",
    "dataset": DATASET_NAME,
    "model": MODEL_NAME,
    "task": "aqueous solubility regression",
    "target": "logS",
    "samples": EXPECTED_SAMPLES,
    "seed": SEED,
    "prompt_source": "dataset Question field",
    "generation": {
        "method": "model.generate",
        "do_sample": False,
    },
    "prediction_parsing": {
        "source": "generated continuation only",
        "format": "<float>VALUE</float>",
    },
    "image_passed_to_model": False,
    "metrics": [
        "MAE",
        "RMSE",
        "R2",
        "Pearson correlation",
    ],
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    },
}

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )


# =============================================================================
# Final report
# =============================================================================

print("\n" + "=" * 80)
print("ESOL-V FULL 220 RESULTS")
print("=" * 80)

print(
    f"\nCompleted:   {len(results_df)}/{EXPECTED_SAMPLES}"
)

print(
    f"Parsed:      {parsed_count}/{EXPECTED_SAMPLES} "
    f"({parse_rate:.2%})"
)

print(
    f"Unparseable: {unparseable_count}"
)

print("\n" + "-" * 80)
print("REGRESSION METRICS")
print("-" * 80)

print(f"MAE:       {mae:.6f}")
print(f"RMSE:      {rmse:.6f}")
print(f"R²:        {r2:.6f}")
print(f"Pearson r: {pearson_r:.6f}")
print(f"Pearson p: {pearson_p:.6g}")

print("\n" + "-" * 80)
print("TIMING")
print("-" * 80)

print(
    f"Model load:     "
    f"{model_load_seconds / 60:.2f} minutes"
)

print(
    f"Current session:"
    f" {session_seconds / 60:.2f} minutes"
)

print(
    f"Mean inference: "
    f"{mean_inference_seconds:.2f} seconds/sample"
)

print("\n" + "-" * 80)
print("OUTPUT FILES")
print("-" * 80)

print(f"Predictions:  {FINAL_FILE}")
print(f"Summary:      {SUMMARY_FILE}")
print(f"Unparseable:  {UNPARSEABLE_FILE}")
print(f"Metadata:     {METADATA_FILE}")
print(f"Checkpoint:   {CHECKPOINT_FILE}")

print("\nReproduction complete.")