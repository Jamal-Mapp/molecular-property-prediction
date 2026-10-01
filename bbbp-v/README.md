# BBBP-V Reproduction

This directory contains a reproduction of the **MolVision BBBP-V molecular property classification experiment** using the SMILES 2-shot scaffold condition.

The goal of this reproduction is to verify the evaluation pipeline, establish a working baseline, and examine how output-parsing strategy affects reported classification performance.

## Experimental Setup

| Component | Configuration |
|---|---|
| Dataset | `molvision/BBBP-V-SMILES-2` |
| Model | `Qwen/Qwen-VL-Chat-Int4` |
| Task | Binary molecular property classification |
| Samples | 410 |
| Representation | SMILES |
| In-context examples | 2 |
| Sampling strategy | Scaffold |
| Generation | `model.generate(**inputs)` |
| Molecular image passed to model | No |

The dataset-provided `Question` field is used directly as the model prompt.

Although the dataset contains molecular structure images, this experiment evaluates the **SMILES-only condition**, so the images are not provided to the model.

---

## Evaluation Paths

Two output-parsing approaches are evaluated.

### 1. Full-Sequence Parsing

The complete decoded model sequence is evaluated:

`prompt + generated continuation`

The first valid `<boolean>Yes</boolean>` or `<boolean>No</boolean>` tag is extracted from the decoded sequence.

This path reproduces the behavior used during the initial MolVision reproduction.

### 2. Generated-Continuation-Only Parsing

The input-token portion of the sequence is removed before decoding.

Only newly generated tokens are evaluated:

`generated continuation only`

The same `<boolean>` parser is then applied.

This second path was added to determine whether predictions extracted from the complete decoded sequence differ from predictions contained in the model's newly generated response.

---

## Results

The experiment was completed across all **410 BBBP-V samples**.

| Evaluation Path | Parsed | Parse Rate | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Full sequence | 353 / 410 | 86.10% | 84.42% | 0.8980 |
| Generated continuation only | 349 / 410 | 85.12% | 75.64% | 0.8604 |

### Confusion Matrices

Confusion matrices use the ordering:

`[TN, FP, FN, TP]`

**Full-sequence parsing**

`[56, 32, 23, 242]`

**Generated-continuation-only parsing**

`[2, 85, 0, 262]`

The generated-continuation-only path contained **61 unparseable outputs**.

---

## Interpretation

The two parsing strategies produce noticeably different classification results despite being derived from the same model generations.

The full-sequence path achieves higher measured accuracy and produces a substantially different confusion matrix from the generated-continuation-only path.

This difference motivates closer examination of how evaluation code extracts predictions from autoregressive model outputs.

These results should therefore be interpreted as both:

1. a reproduction of the BBBP-V classification experiment, and
2. an evaluation of how output parsing influences the reported metrics.

No causal conclusion is made here about why the two paths differ.

---

## Reproduction Script

The full experiment is implemented in:

`bbbp_v_evaluation.py`

The script includes:

- dataset verification
- CUDA environment checks
- deterministic random seeds
- checkpoint/resume support
- per-sample inference timing
- full-sequence parsing
- generated-continuation-only parsing
- accuracy and F1 computation
- confusion-matrix computation
- unparseable-output collection
- experiment metadata export

Checkpointing occurs after every sample so long cluster runs can be resumed without restarting the full experiment.

---

## Output Files

Running the evaluation creates a local `results/` directory containing:

```text
results/
├── bbbp_v_checkpoint.csv
├── bbbp_v_predictions.csv
├── bbbp_v_summary.csv
├── bbbp_v_unparseable.csv
└── bbbp_v_metadata.json
