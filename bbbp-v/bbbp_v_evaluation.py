# ================================================================
# MolVision BBBP-V — Qwen-VL Full Newton Reproduction
# ================================================================
#
# Dataset: molvision/BBBP-V-SMILES-2
# Model:   Qwen/Qwen-VL-Chat-Int4
# Samples: 410
#
# PATH A:
#   Released-style full-sequence parsing
#
# PATH B:
#   Generated-continuation-only parsing
#
# Checkpoints after EVERY sample.
# Safe to resume by rerunning this script.
# ================================================================

import os
import re
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ================================================================
# CONFIG
# ================================================================

DATASET_NAME = "molvision/BBBP-V-SMILES-2"
MODEL_NAME = "Qwen/Qwen-VL-Chat-Int4"

BASE = Path.home() / "molvision-esol-reproduction"
RESULTS = BASE / "results" / "bbbp"

RESULTS.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = RESULTS / "Qwen_BBBP_V_SMILES_2_full410_checkpoint.csv"
FINAL_FILE = RESULTS / "Qwen_BBBP_V_SMILES_2_full410_newton.csv"
SUMMARY_FILE = RESULTS / "Qwen_BBBP_V_SMILES_2_full410_newton_summary.csv"
UNPARSEABLE_FILE = RESULTS / "Qwen_BBBP_V_SMILES_2_full410_unparseable.csv"
METADATA_FILE = RESULTS / "Qwen_BBBP_V_SMILES_2_full410_metadata.json"

EXPECTED_SAMPLES = 410


# ================================================================
# REPRODUCIBILITY
# ================================================================

torch.manual_seed(1234)
np.random.seed(1234)


# ================================================================
# ENVIRONMENT
# ================================================================

print("\n" + "=" * 90)
print("MOLVISION BBBP-V — NEWTON FULL REPRODUCTION")
print("=" * 90)

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Aborting production run.")

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB"
)


# ================================================================
# PARSERS
# ================================================================

BOOLEAN_PATTERN = re.compile(
    r"<boolean>\s*(Yes|No)\s*</boolean>",
    re.IGNORECASE
)


def extract_boolean(text):
    if text is None:
        return None

    try:
        if pd.isna(text):
            return None
    except Exception:
        pass

    match = BOOLEAN_PATTERN.search(str(text))

    if not match:
        return None

    value = match.group(1).lower()

    return "Yes" if value == "yes" else "No"


def extract_demo_labels(question):
    matches = BOOLEAN_PATTERN.findall(str(question))

    return [
        "Yes" if value.lower() == "yes" else "No"
        for value in matches
    ]


# ================================================================
# DATASET
# ================================================================

print("\n" + "=" * 90)
print("LOADING BBBP-V")
print("=" * 90)

dataset = load_dataset(DATASET_NAME)
df = dataset["train"].to_pandas()

print("Dataset:", DATASET_NAME)
print("Samples:", len(df))

if len(df) != EXPECTED_SAMPLES:
    raise RuntimeError(
        f"Expected {EXPECTED_SAMPLES} samples but received {len(df)}."
    )

print("✓ Dataset verified: 410 samples")


# ================================================================
# RESUME CHECKPOINT
# ================================================================

results = []
completed_indices = set()

if CHECKPOINT_FILE.exists():

    checkpoint_df = pd.read_csv(CHECKPOINT_FILE)

    if len(checkpoint_df):

        checkpoint_df = (
            checkpoint_df
            .sort_values("dataset_index")
            .drop_duplicates(
                subset=["dataset_index"],
                keep="last"
            )
        )

        results = checkpoint_df.to_dict("records")

        completed_indices = set(
            checkpoint_df["dataset_index"]
            .astype(int)
            .tolist()
        )

    print("\n" + "=" * 90)
    print("CHECKPOINT FOUND")
    print("=" * 90)

    print("Completed:", len(completed_indices))
    print("Remaining:", len(df) - len(completed_indices))

else:

    print("\nNo checkpoint found.")
    print("Starting new 410-sample production run.")


# ================================================================
# MODEL
# ================================================================

model_load_seconds = 0.0

if len(completed_indices) < len(df):

    print("\n" + "=" * 90)
    print("LOADING QWEN-VL-CHAT-INT4")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    print("✓ Tokenizer loaded")

    model_start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    model_load_seconds = time.time() - model_start

    print(
        f"✓ Model loaded in {model_load_seconds:.2f}s "
        f"({model_load_seconds / 60:.2f} min)"
    )


# ================================================================
# FULL 410 INFERENCE
# ================================================================

session_start = time.time()
completed_at_start = len(completed_indices)

print("\n" + "=" * 90)
print("STARTING FULL BBBP-V INFERENCE")
print("=" * 90)

print("Total:", len(df))
print("Already completed:", completed_at_start)
print("Remaining:", len(df) - completed_at_start)


for idx in range(len(df)):

    if idx in completed_indices:
        continue

    row = df.iloc[idx]

    question = str(row["Question"])
    ground_truth_raw = str(row["Answer"])

    ground_truth = extract_boolean(ground_truth_raw)

    demos = extract_demo_labels(question)

    demo1 = demos[0] if len(demos) >= 1 else None
    demo2 = demos[1] if len(demos) >= 2 else None

    print("\n" + "-" * 90)
    print(f"SAMPLE {idx + 1}/{len(df)} | DATASET INDEX {idx}")
    print("-" * 90)

    print(
        "Truth:", ground_truth,
        "| Demo1:", demo1,
        "| Demo2:", demo2
    )

    inputs = tokenizer(
        question,
        return_tensors="pt"
    ).to(device)

    input_length = inputs["input_ids"].shape[1]

    full_decoded = None
    generated_text = None

    author_prediction = None
    generated_prediction = None

    error_message = None

    inference_start = time.time()

    try:

        with torch.no_grad():

            # ====================================================
            # RELEASED MOLVISION GENERATION CALL
            # ====================================================
            output_ids = model.generate(**inputs)

        inference_seconds = time.time() - inference_start

        # ========================================================
        # PATH A — FULL SEQUENCE
        # ========================================================

        full_decoded = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        author_prediction = extract_boolean(full_decoded)

        # ========================================================
        # PATH B — GENERATED CONTINUATION ONLY
        # ========================================================

        generated_ids = output_ids[0][input_length:]

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        generated_prediction = extract_boolean(generated_text)

    except Exception as exc:

        inference_seconds = time.time() - inference_start
        error_message = repr(exc)

        print("INFERENCE ERROR:", error_message)

    author_correct = (
        author_prediction == ground_truth
        if author_prediction is not None
        else False
    )

    generated_correct = (
        generated_prediction == ground_truth
        if generated_prediction is not None
        else False
    )

    print(f"Inference: {inference_seconds:.2f}s")

    print(
        "Full-sequence:",
        author_prediction,
        "| Correct:",
        author_correct
    )

    print(
        "Generated-only:",
        generated_prediction,
        "| Correct:",
        generated_correct
    )

    record = {

        "dataset_index": idx,

        "ground_truth_raw": ground_truth_raw,
        "ground_truth": ground_truth,

        "demo1": demo1,
        "demo2": demo2,
        "num_demo_labels": len(demos),

        "question": question,

        "author_parser_prediction": author_prediction,
        "generated_only_prediction": generated_prediction,

        "author_correct": author_correct,
        "generated_correct": generated_correct,

        "full_decoded_response": full_decoded,
        "generated_continuation": generated_text,

        "inference_seconds": inference_seconds,
        "error": error_message,

        "image_passed_to_model": False
    }

    results.append(record)

    # ============================================================
    # CHECKPOINT AFTER EVERY SAMPLE
    # ============================================================

    checkpoint_df = (
        pd.DataFrame(results)
        .sort_values("dataset_index")
        .drop_duplicates(
            subset=["dataset_index"],
            keep="last"
        )
    )

    checkpoint_df.to_csv(
        CHECKPOINT_FILE,
        index=False
    )

    completed_total = len(checkpoint_df)

    processed_this_session = (
        completed_total - completed_at_start
    )

    elapsed = time.time() - session_start

    if processed_this_session > 0:

        average = elapsed / processed_this_session
        remaining = len(df) - completed_total
        eta_seconds = average * remaining

        print(
            f"Progress: {completed_total}/{len(df)} "
            f"({100 * completed_total / len(df):.1f}%)"
        )

        print(
            f"Session average: {average:.2f}s/sample"
        )

        print(
            f"Estimated remaining: {eta_seconds / 60:.1f} min"
        )

        print("✓ checkpoint saved")


# ================================================================
# FINALIZE
# ================================================================

session_seconds = time.time() - session_start

final_df = (
    pd.DataFrame(results)
    .sort_values("dataset_index")
    .drop_duplicates(
        subset=["dataset_index"],
        keep="last"
    )
    .reset_index(drop=True)
)

final_df.to_csv(
    FINAL_FILE,
    index=False
)


# ================================================================
# METRICS
# ================================================================

def evaluate_path(dataframe, prediction_column):

    valid = dataframe[
        dataframe[prediction_column].isin(["Yes", "No"])
        &
        dataframe["ground_truth"].isin(["Yes", "No"])
    ].copy()

    parsed = len(valid)

    parse_rate = (
        parsed / len(dataframe)
        if len(dataframe)
        else 0.0
    )

    if parsed == 0:

        return {
            "parsed": 0,
            "parse_rate": 0.0,
            "accuracy": np.nan,
            "f1": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan
        }

    y_true = valid["ground_truth"]
    y_pred = valid[prediction_column]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="Yes"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["No", "Yes"]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "parsed": parsed,
        "parse_rate": parse_rate,
        "accuracy": accuracy,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


author_metrics = evaluate_path(
    final_df,
    "author_parser_prediction"
)

generated_metrics = evaluate_path(
    final_df,
    "generated_only_prediction"
)


# ================================================================
# UNPARSEABLE GENERATED OUTPUTS
# ================================================================

unparseable = final_df[
    ~final_df[
        "generated_only_prediction"
    ].isin(["Yes", "No"])
].copy()

unparseable.to_csv(
    UNPARSEABLE_FILE,
    index=False
)


# ================================================================
# SUMMARY
# ================================================================

summary = pd.DataFrame([{

    "dataset": DATASET_NAME,
    "model": MODEL_NAME,

    "total_samples": len(final_df),

    "author_parsed": author_metrics["parsed"],
    "author_parse_rate": author_metrics["parse_rate"],
    "author_accuracy": author_metrics["accuracy"],
    "author_f1": author_metrics["f1"],

    "author_tn": author_metrics["tn"],
    "author_fp": author_metrics["fp"],
    "author_fn": author_metrics["fn"],
    "author_tp": author_metrics["tp"],

    "generated_parsed": generated_metrics["parsed"],
    "generated_parse_rate": generated_metrics["parse_rate"],
    "generated_accuracy": generated_metrics["accuracy"],
    "generated_f1": generated_metrics["f1"],

    "generated_tn": generated_metrics["tn"],
    "generated_fp": generated_metrics["fp"],
    "generated_fn": generated_metrics["fn"],
    "generated_tp": generated_metrics["tp"],

    "generated_unparseable": len(unparseable),

    "model_load_seconds": model_load_seconds,
    "session_seconds": session_seconds,

    "mean_inference_seconds":
        final_df["inference_seconds"].mean(),

    "median_inference_seconds":
        final_df["inference_seconds"].median(),

    "image_passed_to_model": False
}])

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ================================================================
# METADATA
# ================================================================

metadata = {

    "dataset": DATASET_NAME,
    "model": MODEL_NAME,

    "samples": len(final_df),

    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "torch": torch.__version__,

    "gpu": torch.cuda.get_device_name(0),

    "seed": 1234,

    "generation_call":
        "model.generate(**inputs)",

    "released_path":
        (
            "Decode complete output sequence "
            "(prompt + generated continuation) "
            "and extract first <boolean> tag."
        ),

    "generated_only_path":
        (
            "Remove input token sequence, "
            "decode generated continuation only, "
            "then extract first <boolean> tag."
        ),

    "image_passed_to_model": False,

    "checkpoint_file": str(CHECKPOINT_FILE),
    "final_file": str(FINAL_FILE),
    "summary_file": str(SUMMARY_FILE),
    "unparseable_file": str(UNPARSEABLE_FILE)
}

with open(
    METADATA_FILE,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ================================================================
# FINAL REPORT
# ================================================================

print("\n\n" + "=" * 90)
print("FULL BBBP-V RESULTS")
print("=" * 90)

print("\nSamples:", len(final_df))


print("\n" + "-" * 90)
print("RELEASED-CODE / FULL-SEQUENCE PATH")
print("-" * 90)

print(
    f"Parsed: "
    f"{author_metrics['parsed']}/{len(final_df)} "
    f"({author_metrics['parse_rate'] * 100:.2f}%)"
)

print(
    f"Accuracy: "
    f"{author_metrics['accuracy']:.4f} "
    f"({author_metrics['accuracy'] * 100:.2f}%)"
)

print(
    f"F1: {author_metrics['f1']:.4f}"
)

print(
    "Confusion [TN FP FN TP]:",
    [
        author_metrics["tn"],
        author_metrics["fp"],
        author_metrics["fn"],
        author_metrics["tp"]
    ]
)


print("\n" + "-" * 90)
print("GENERATED-CONTINUATION-ONLY PATH")
print("-" * 90)

print(
    f"Parsed: "
    f"{generated_metrics['parsed']}/{len(final_df)} "
    f"({generated_metrics['parse_rate'] * 100:.2f}%)"
)

print(
    f"Accuracy: "
    f"{generated_metrics['accuracy']:.4f} "
    f"({generated_metrics['accuracy'] * 100:.2f}%)"
)

print(
    f"F1: {generated_metrics['f1']:.4f}"
)

print(
    "Confusion [TN FP FN TP]:",
    [
        generated_metrics["tn"],
        generated_metrics["fp"],
        generated_metrics["fn"],
        generated_metrics["tp"]
    ]
)

print(
    "Generated-only unparseable:",
    len(unparseable)
)


print("\n" + "-" * 90)
print("TIMING")
print("-" * 90)

print(
    f"Model load: "
    f"{model_load_seconds / 60:.2f} min"
)

print(
    f"Inference session: "
    f"{session_seconds / 60:.2f} min"
)

print(
    f"Mean/sample: "
    f"{final_df['inference_seconds'].mean():.2f}s"
)

print(
    f"Median/sample: "
    f"{final_df['inference_seconds'].median():.2f}s"
)


print("\n" + "-" * 90)
print("FILES")
print("-" * 90)

print("Checkpoint:", CHECKPOINT_FILE)
print("Full results:", FINAL_FILE)
print("Summary:", SUMMARY_FILE)
print("Unparseable:", UNPARSEABLE_FILE)
print("Metadata:", METADATA_FILE)


print("\n" + "=" * 90)
print("✓ FULL 410-SAMPLE BBBP-V RUN COMPLETE")
print("=" * 90)