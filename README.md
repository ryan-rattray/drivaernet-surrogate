# DrivAerNet Surrogate
An experiment in surrogate CAE modeling, with a touch of tabular foundation model predictions.

Predicting drag coefficient (Cd) from DrivAerNet++ design parameters.
Four models are compared, using the authors' original split:  
- gradient boosting (LightGBM)
- a hand-built PyTorch MLP
- a tabular foundation model (TabPFN)
- mesh GNN using 3D geometry as input

Dataset:  DrivAerNet++ (Elrefaie et al., NeurIPS 2024), CC BY-NC 4.0

## Stages and Status

My work is organized in Flights: themed stages of the experiment,
each with its own done criteria. This allows me to do cool things like play with surrogate modeling while not interfering with my family life.  

- **Flight 1:  The tabular surrogate (complete 11-8-2026).** Predict Cd from the design parameters with traditional ML and a tabular foundation model.  R² of predicted values vs ground truth is the chosen metric.

- **Flight 2: The geometric surrogate (in process.)**
A graph neural network built from scratch on the car meshes, predicting Cd from geometry alone on the same split, compared against a pretrained 3D backbone adapted to the task.

- **Flight 3: Serve it (planned).** One of the surrogates behind a small API, containerized, so the repo can be cloned and run with one command.

- **Flight 4: Package the portfolio (planned).** Create the visualizations:
    - predicted-vs-actual figures
    - mesh renders
    - a demo page with sliders for the 23 parameters driving live Cd predictions 

## Running Results
**Flight 1:  Tabular Models**

Gradient boosting and MLP were run with default settings.  Tuning will occur in a later flight.

TabPFN has test results already, as there is no training.   
| Contender | What it is | Train R² | Val R² | Test R² |
|---|---|---|---|---|
| Gradient boosting | the conventional tool, trained | 0.9123343713333352 | 0.6356559013459231 | deferred |
| PyTorch MLP | hand-built network, trained | 0.811007961730985 | 0.3193305738393898 | deferred |
| TabPFN | foundation model, context only, no training | — (no training) | 0.8123367964963065 | 0.8185785487036595 |

- TabPFN beat both of the default-value trained traditional models on validation results.
- Both of the trained traditional models overfit significantly.
- The hand-built PyTorch MLP with typical default settings was not at all competitive on this data.

## The Motivation
As a CAE engineer, I have developed a fascination for incorporating machine learning surrogate models for FEA simulations.  This project is largely to develop the skills required to do so.  I have also recently begun to use tabular foundation models in my predictive work, so it seemed like a natural progression to merge them into surrogate modeling.

## Considered Model Families
Surrogate modeling for CAE can take on two forms:  traditional predictions from tabularized design parameters, or specialized deep learning surrogates trained on the CAD geometry itself.  This project aims to compare the traditional with the deep.  The model families considered:

1.  Gradient Boosted Decision Trees
    - LightGBM
2.  Feed-Forward Neural Network
    - Hand-built Multilayer Perceptron (PyTorch)
3.  Tabular Foundation Models
    - TabPFN
4.  Graph Neural Network
    - architecture to be designed in Flight 2

## The Data

DrivAerNet++ (Elrefaie et al., NeurIPS 2024), released under CC BY-NC 4.0.
The tabular lane uses 4,165 parametric designs: 23 design parameters per
design, with the categorical choices carried in the design ID string rather
than in extra columns. Drag coefficient (Cd) ranges from 0.200 to 0.320.
This repository distributes no dataset files; the data must be obtained
from the original sources.

Original project and dataset: [DrivAerNet++ official repository](https://github.com/Mohamedelrefaie/DrivAerNet)


### Getting the data

1. Download the drag-values file, `DrivAerNetPlusPlus_Cd_8k_Updated.csv`
   (linked as "Drag Values" in the Performance Data section of the
   DrivAerNet++ README), and save it into `data/raw/` under that name.
   Its columns are `ID` and `Drag_Value`.
2. Download the design-parameters file, `DrivAerNet_ParametricData.csv`,
   from the Hugging Face dataset MoElrefaie/DrivAerNet (you will need to
   accept the dataset conditions there), and save it into `data/raw/`
   under that name. The same file is also available individually from the
   dataset's Harvard Dataverse page. Its design-ID column is `Experiment`.
3. Run `src/check_data.py` from the repo root. It joins the two files on
   the design ID (`Experiment` in the parameters file, `ID` in the drag
   file), verifies the merge design-for-design, and writes
   `data/designs.csv` plus `results/data_check.txt`. Then run
   `src/make_split.py` to build `data/split.csv`.

`data/raw/` and `data/designs.csv` are gitignored: the authors' data
stays local, and the committed `data/split.csv` is the split contract.

## Setup

Python from python.org, with a per-project `.venv` and pip (no conda).
Create and activate the venv, then install the pinned environment:

```
pip install -r requirements.txt
```

The direct installs are LightGBM, PyTorch, scikit-learn, pandas, and
NumPy, plus pytest for the test suite. The remaining lines in
`requirements.txt` are their dependencies, pinned so the environment
reproduces exactly.

TabPFN is deliberately not in `requirements.txt`: `src/run_tabpfn.py`
runs on Kaggle, so a Pylance warning that the `tabpfn` import cannot be
resolved locally is expected — see How to Run.


## The train/val/test Split

The authors' published split is used for each model:
- 3,051 train
- 555 validation
- 559 test designs
- 4,165 total

The split is saved as design IDs in `data/split.csv`. The same split is inherited unchanged by the later flights.

## Repo Layout

```
drivaernet-surrogate/
├── README.md
├── requirements.txt       (the pinned environment; see Setup)
├── data/
│   ├── designs.csv          (not distributed; see The Data)
│   └── split.csv            (the authors' published train/val/test split)
├── results/
│   ├── comparison.md        (the comparison table, built by compare_results.py)
│   ├── data_check.txt
│   ├── gbdt_results.txt
│   ├── mlp_results.txt
│   └── tabpfn_results.txt
├── src/
│   ├── common.py                  (shared loading: the 23 features, the split, the leak guard)
│   ├── check_data.py              (verifies the joined table; writes results/data_check.txt)
│   ├── make_split.py              (builds data/split.csv)
│   ├── train_gbdt_defaults.py     (LightGBM at default settings; validation only)
│   ├── train_mlp_defaults.py      (hand-built PyTorch MLP at defaults; validation only)
│   ├── run_tabpfn.py              (TabPFN — runs on Kaggle, not in the local venv)
│   └── compare_results.py         (builds results/comparison.md from the results files)
└── tests/
    └── test_split_integrity.py    (split counts, disjointness, and the results-file rules)
```


## How to Run

From the repo root, with the `.venv` active, one script per step, in
stair order:

- `python src/check_data.py` — verify the joined table; writes `results/data_check.txt`
- `python src/make_split.py` — build `data/split.csv` from the authors' published split
- `python src/train_gbdt_defaults.py` — LightGBM at default settings; validation only
- `python src/train_mlp_defaults.py` — hand-built PyTorch MLP at defaults; validation only
- `src/run_tabpfn.py` — TabPFN; runs on Kaggle, not in the local venv (the Pylance "tabpfn could not be resolved" warning locally is expected)
- `python src/compare_results.py` — build `results/comparison.md` from the results files
- `python -m pytest` — split integrity and results-file rules (3 tests)


## Ground Rules
- Validation is the selection metric. Test never feeds a decision for a trained model; the trained contenders' test reads are deferred to the later decision comparison.
- TabPFN is exempt from deferral: it does not train or tune, so its test read has no feedback loop.
- When tuning happens later, the champion refits on train plus validation with its configuration frozen before any test read.

## Tuning

No tuning in Flight 1, by design. Flight 1 compares the three tabular models at their default settings. Tuning belongs to the later decision comparison, after the GNN is built: the trained contenders get tuned versions in separate files (tune_gbdt.py, tune_mlp.py), and the best of them faces the GNN and TabPFN.

## Credit

The dataset is **DrivAerNet++** (Elrefaie et al., NeurIPS 2024), a large-scale
dataset of parametric car designs with CFD-computed drag coefficients,
released under the CC BY-NC 4.0 license. This repository distributes no
dataset files; the data must be obtained from the original source.

- Paper: Elrefaie, M., et al. "DrivAerNet++: A Large-Scale Multimodal Car
  Dataset with Computational Fluid Dynamics Simulations and Deep Learning
  Benchmarks." NeurIPS 2024, Datasets and Benchmarks Track.
  arXiv:2406.09624 — https://arxiv.org/abs/2406.09624
- Official repository: https://github.com/Mohamedelrefaie/DrivAerNet

TabPFN is the tabular foundation model developed by Prior Labs; in this
project it was run on Kaggle. The trained models were built with LightGBM
and PyTorch, with scikit-learn for scoring.


