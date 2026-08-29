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

---

# Current Study: ESOL-V

The first dataset under investigation is **ESOL-V**, using an aqueous-solubility subset represented with SMILES.

Current exploratory work includes:

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

The scaffold analysis also suggests a mixed structural distribution: many scaffold types appear only once, while a small number of structural categories account for a substantial portion of the molecules.

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

# Research Questions

The questions in this repository are expected to evolve as the investigation develops.

Current directions include:

1. What information about molecular properties is captured by simple physicochemical descriptors?

2. How much chemical and structural diversity exists within the datasets being evaluated?

3. How does predictive performance change across different molecular representations?

4. Do representations such as SMILES, SELFIES, fingerprints, molecular images, or learned features provide complementary information?

5. When does combining representations produce meaningful gains rather than additional computational complexity?

---

# Repository Organization

The repository will be organized incrementally as the research develops.

Dataset-specific exploration will be kept under `analysis/`, while later experiments, models, and reusable code will be added when needed.

```text
molecular-property-prediction/
│
├── README.md
│
└── analysis/
    └── esol-v/
```

The root README serves as the **front page of the research project**.

More detailed methodology, code, figures, results, and observations will live with the corresponding analysis or experiment.

---

# Research Status

### Current

**ESOL-V dataset characterization and chemical-diversity analysis**

### Next

Prepare a controlled modeling dataset and establish simple predictive baselines before evaluating more complex molecular representations.

---

# Reproducibility and Interpretation

This is an **active research repository**.

Results may be refined as analyses are expanded, errors are identified, or experimental assumptions change.

Exploratory findings are labeled as such, and observations derived in this repository should not be interpreted as claims made by the original dataset authors unless explicitly cited.

---

*This README will evolve with the research. The repository is intended to preserve both the progression of the investigation and the evidence supporting its conclusions.*
