# ESOL-V Reproduction

This directory contains a reproduction of the **MolVision ESOL-V molecular property regression experiment** using the SMILES 2-shot scaffold condition.

The goal of this reproduction is to verify the regression evaluation pipeline, establish a working MolVision baseline, and evaluate Qwen-VL on aqueous solubility prediction across the complete ESOL-V experiment.

## Experimental Setup

| Component | Configuration |
|---|---|
| Dataset | `molvision/ESOL-V-SMILES-2` |
| Model | `Qwen/Qwen-VL-Chat-Int4` |
| Task | Molecular property regression |
| Target property | Aqueous solubility (`logS`) |
| Samples | 220 |
| Representation | SMILES |
| In-context examples | 2 |
| Sampling strategy | Scaffold |
| Generation | `model.generate(**inputs, do_sample=False)` |
| Molecular image passed to model | No |

The dataset-provided `Question` field is used directly as the model prompt.

Although ESOL-V contains molecular structure images, this experiment evaluates the **SMILES-only condition**, so molecular images are not provided to the model.

---

## Prediction Parsing

ESOL-V is a regression task, so the model is expected to produce a continuous numerical prediction using the format:

`<float>VALUE</float>`

For evaluation, only tokens generated after the original input prompt are decoded:

`generated continuation only`

The first valid numerical value enclosed by `<float>` tags is extracted as the predicted aqueous solubility.

This avoids extracting numerical values that may already appear inside the input prompt.

---

## Evaluation Metrics

Predictions are evaluated using four regression metrics:

- **Mean Absolute Error (MAE)** — average absolute difference between predicted and measured solubility.
- **Root Mean Squared Error (RMSE)** — measures prediction error while placing greater weight on larger errors.
- **R²** — measures how much of the observed variation in solubility is explained by the predictions.
- **Pearson correlation** — measures the linear relationship between predicted and measured solubility values.

Only successfully parsed model predictions are included when calculating the regression metrics.

Unparseable generations are recorded separately for inspection.

---

## Reproduction Script

The complete experiment is implemented in:

`esol_v_evaluation.py`

The script includes:

- ESOL-V dataset loading and verification
- CUDA environment validation
- deterministic random seeds
- Qwen-VL model loading
- checkpoint/resume support
- per-sample inference timing
- generated-continuation-only decoding
- `<float>` prediction parsing
- MAE computation
- RMSE computation
- R² computation
- Pearson correlation analysis
- unparseable-output collection
- experiment metadata export

Checkpointing occurs after every sample so long cluster runs can be resumed without restarting the complete 220-sample experiment.

---

## Output Files

Running the evaluation creates a local `results/` directory containing:

```text
results/
├── esol_v_checkpoint.csv
├── esol_v_predictions.csv
├── esol_v_summary.csv
├── esol_v_unparseable.csv
└── esol_v_metadata.json
