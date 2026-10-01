# Molecular Property Prediction

A living research repository for exploring molecular representations, chemical datasets, and machine-learning approaches to molecular property prediction.

## About This Repository

Molecular property prediction asks a fundamental question:

**What can we learn about a molecule's properties from the way we represent it?**

Molecules can be represented in many forms, including physicochemical descriptors, molecular fingerprints, SMILES, SELFIES, molecular graphs, 2D structure images, learned embeddings, and combinations of multiple representations.

Each representation preserves different information and introduces different assumptions into a predictive model.

This repository documents an ongoing investigation of those representations and their usefulness for molecular property prediction.

The goal is not simply to collect model scores. The work here is intended to build an understandable research record:

- what was investigated
- why it was investigated
- what was observed
- what questions those observations raise next

---

## Research Approach

The project begins with the **data rather than the model**.

Before evaluating increasingly complex prediction methods, each dataset is examined for characteristics that may influence an experiment, including:

- target-property distributions
- duplicate and equivalent molecular records
- physicochemical characteristics
- structural and scaffold diversity
- relationships between simple molecular descriptors and prediction targets
- potential sources of redundancy or bias

These observations provide context for later model comparisons and help establish what information is already available from relatively simple molecular representations.

The project then moves from dataset characterization into **controlled reproduction experiments**.

Two MolVision tasks have currently been reproduced:

- **ESOL-V regression** for aqueous-solubility prediction
- **BBBP-V binary classification** for blood-brain barrier permeability prediction

These reproduction experiments provide working baselines before introducing new representations, prompting strategies, evaluation procedures, or multimodal extensions.

---

# ESOL-V Dataset Analysis

The first dataset examined in detail is **ESOL-V**, using an aqueous-solubility subset represented with SMILES.

Exploratory work includes:

- dataset structure inspection
- duplicate-molecule analysis
- SMILES canonicalization
- solubility target-distribution analysis
- RDKit molecular descriptor extraction
- descriptor-to-solubility correlation analysis
- Bemis-Murcko scaffold analysis

## Preliminary Dataset Audit

| Observation | Result |
| --- | ---: |
| Dataset records | 220 |
| Exact unique SMILES | 196 |
| Unique canonical molecules | 195 |
| Unique Bemis-Murcko scaffolds | 75 |
| Singleton scaffolds | 56 |
| Scaffold diversity ratio | 0.385 |
| Acyclic molecules | 42 |

Canonicalization revealed that the number of chemically unique molecules is slightly smaller than the number of unique SMILES strings.

This illustrates why molecular identity should be checked before treating every textual representation as an independent sample.

The scaffold analysis also suggests a mixed structural distribution: many scaffold types appear only once, while a smaller number of structural categories account for a substantial portion of the molecules.

---

## Preliminary Descriptor Observations

Several common molecular descriptors were calculated with RDKit and compared with measured aqueous solubility (`logS`).

Among the descriptors examined so far, calculated **LogP** shows the strongest linear relationship with solubility.

| Descriptor | Pearson correlation with logS |
| --- | ---: |
| LogP | -0.798 |
| Molecular weight | -0.631 |
| Aromatic rings | -0.582 |
| Heavy atoms | -0.576 |
| Ring count | -0.572 |
| H-bond donors | 0.203 |
| TPSA | 0.150 |
| H-bond acceptors | 0.097 |

These are **exploratory associations**, not model-performance results or causal claims.

They provide a useful baseline question for subsequent experiments:

> **How much predictive information can more sophisticated molecular representations contribute beyond simple physicochemical descriptors?**

---

# MolVision Reproduction Experiments

The repository currently contains two full molecular-property prediction reproductions using the MolVision benchmark and Qwen-VL.

## ESOL-V Regression

The ESOL-V reproduction evaluates continuous aqueous-solubility prediction.

| Component | Configuration |
| --- | --- |
| Dataset | `molvision/ESOL-V-SMILES-2` |
| Model | `Qwen/Qwen-VL-Chat-Int4` |
| Task | Regression |
| Target | Aqueous solubility (`logS`) |
| Samples | 220 |
| Representation | SMILES |
| In-context examples | 2 |
| Sampling strategy | Scaffold |
| Image input | No |

Predictions are parsed from the model's **generated continuation only** and evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Pearson correlation

Detailed methodology and code are available in [`esol-v/`](./esol-v/).

---

## BBBP-V Classification

The BBBP-V reproduction evaluates binary blood-brain barrier permeability prediction.

| Component | Configuration |
| --- | --- |
| Dataset | `molvision/BBBP-V-SMILES-2` |
| Model | `Qwen/Qwen-VL-Chat-Int4` |
| Task | Binary classification |
| Samples | 410 |
| Representation | SMILES |
| In-context examples | 2 |
| Sampling strategy | Scaffold |
| Image input | No |

Two prediction-parsing paths were compared:

1. full decoded sequence
2. generated continuation only

### BBBP-V Results

| Evaluation Path | Parsed | Parse Rate | Accuracy | F1 |
| --- | ---: | ---: | ---: | ---: |
| Full sequence | 353 / 410 | 86.10% | 84.42% | 0.8980 |
| Generated continuation only | 349 / 410 | 85.12% | 75.64% | 0.8604 |

The substantial difference between these two evaluation paths motivates closer inspection of how autoregressive model outputs are parsed when computing molecular-property prediction metrics.

Detailed methodology and code are available in [`bbbp-v/`](./bbbp-v/).

---

# Research Questions

The questions in this repository are expected to evolve as the investigation develops.

Current directions include:

1. What information about molecular properties is captured by simple physicochemical descriptors?

2. How much chemical and structural diversity exists within the datasets being evaluated?

3. How does predictive performance change across different molecular representations?

4. Do representations such as SMILES, SELFIES, fingerprints, molecular images, or learned features provide complementary information?

5. When does combining representations produce meaningful gains rather than additional computational complexity?

6. How sensitive are reported molecular-property prediction results to prompting and output-parsing methodology?

7. Can reproduced MolVision baselines be extended into controlled experiments that isolate the contribution of molecular representation?

---

# Repository Organization

The repository is organized by research stage.

```text
molecular-property-prediction/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── analysis/
│   ├── 01_esol_v_visual_inspection.py
│   ├── 02_esol_v_molecule_audit.py
│   ├── 03_esol_v_target_distribution.py
│   ├── 04_esol_v_descriptor_analysis.py
│   └── 05_esol_v_scaffold_analysis.py
│
├── esol-v/
│   ├── README.md
│   └── esol_v_evaluation.py
│
└── bbbp-v/
    ├── README.md
    └── bbbp_v_evaluation.py
