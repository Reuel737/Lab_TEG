# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scientific Context

This IC (Iniciação Científica) is part of a larger research effort led by Prof. Murilo Brunazzo Medeiros (UFSC-CTJ) whose dissertation (*"Análise de parâmetros de conforto térmico de cabine por meio de simulações CFD e redes neurais profundas"*, 2025) establishes the methodological framework. The IC implements the **surrogate model** branch of that work: training fully-connected MLPs to reproduce CFD field variables on demand, bypassing the need to re-run expensive simulations for each new set of boundary conditions.

**Motivation**: CFD simulations of a vehicle cabin can take hours to days. A trained neural network that maps spatial coordinates + boundary conditions → field variables runs in milliseconds, enabling real-time thermal comfort analysis and parametric studies.

## Physical Setup

### Vehicle Cabin CFD Model
- Geometry reconstructed from Mao, Ji Wang & Junming Li (2018) — a generic passenger vehicle cabin
- ANSYS Fluent solver, steady-state RANS k–ω SST turbulence model
- Radiation modeled with Discrete Ordinates Method (DOM)
- 45 CFD simulation cases generated via DOE (Design of Experiments) spanning the boundary condition space

### Boundary Conditions (DOE factors)
| Variable | Description | Unit |
|----------|-------------|------|
| `Vel`    | Air velocity at inlet ducts (dashboard nozzles) | m/s |
| `Tinsu`  | Inlet air temperature | °C or K |
| `Qinsu`  | Inlet air volume flow rate | m³/s |

The 45 cases cover combinations of these three factors via Latin Hypercube Sampling (LHS) to span the operational envelope of the HVAC system.

### Output Variables (9 CFD fields)
| Variable | Physical meaning | Unit |
|----------|-----------------|------|
| `pressure` | Gauge static pressure | Pa |
| `x-velocity` | Velocity component in X | m/s |
| `y-velocity` | Velocity component in Y | m/s |
| `z-velocity` | Velocity component in Z | m/s |
| `temperature` | Air temperature | K |
| `incident-radiation` | Incident thermal radiation flux | W/m² |
| `radiation-temperature` | Radiation temperature | K |
| `rad-heat-flux` | Net radiative heat flux | W/m² |
| `vr` | Resultant velocity magnitude | m/s |

Temperature and radiation fields are especially relevant for computing the **PMV/PPD** thermal comfort indices (ISO 7730 / ASHRAE 55) and the **Mean Radiant Temperature (MRT)**, which depend on the radiation field seen by each occupant body segment.

### Body Regions
The cabin volume is split into 8 body regions aligned with occupant thermal comfort analysis zones:

| Directory | Description |
|-----------|-------------|
| `fp_head` | Front passenger — head zone |
| `lrp_head` | Left rear passenger — head zone |
| `rrp_head` | Right rear passenger — head zone |
| `mt_head` | Driver (motorista) — head zone |
| `mt_core` | Driver — torso/core zone |
| `mt_l_foot` | Driver — left foot zone |
| `mt_r_foot` | Driver — right foot zone |
| `outlet` | Air outlet region |

Each region trains an independent network because the flow physics and variable ranges differ significantly between zones (e.g., foot region has very different velocity/temperature profiles than head region).

## Neural Network Architecture

Each region trains a **Fully Connected Neural Network (FCNN / MLP)** — same family as described in the dissertation's surrogate model approach.

- **Input** (6 features): `x-coordinate`, `y-coordinate`, `z-coordinate`, `Vel`, `Tinsu`, `Qinsu`
- **Output** (9 targets): the 9 CFD field variables listed above
- **Architecture**: Keras `Sequential` — `Normalization` layer → N×`Dense(relu)` hidden layers → `Dense(linear)` output
- **Training**: Adam (lr=0.001, β₁=0.9, β₂=0.999) + `ReduceLROnPlateau` (factor=0.9, patience=50, cooldown=20) + `EarlyStopping` (patience=200), MSE loss
- **K-Fold**: 30-fold CV with `random_state=42`; each fold saves `.keras` model, `_dataset.npz`, `_erros.npz`, `_history.npz`

### Current Best Architecture
`512x4` (4 hidden layers of 512 neurons each) consistently outperforms smaller architectures:
- `128x5` (exp08): val_loss ~0.96
- `256x5` (exp12): val_loss ~0.45
- `512x4` (exp15/512x4): val_loss ~0.30 ← **current best**

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

**K-Fold training (primary workflow):**
```bash
# Edit runs/run_kfold.sh to set DATAFILE, REGIAO, EXP, LAYERS, then:
bash runs/run_kfold.sh

# Or call Tf_kfold.py directly:
python3 Tf_kfold.py \
    --Layers "512 512 512 512" \
    --DataFile "dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas" \
    --FileOutPut "regioes/fp_head/results/512x4_fp_head/512x4_fp_head" \
    --HistCSV "regioes/fp_head/historico/hist_kfold_512x4_fp_head.csv" \
    --KFolds 30 --MaxIter 5000 --BatchSize 500
```

**Single split training (Tf.py):**
```bash
python3 Tf.py --Layers "512 512 512 512" --DataFile <file> --FileOutPut <prefix> \
    --MaxIter 5000 --BatchSize 500 [--Fase4] [--L2 0.0001]
```

**TensorBoard:**
```bash
tensorboard --logdir regioes/<REGIAO>/logs
```

## Data Flow

1. Raw CFD CSVs (45 simulation cases) in `dados_n_filtrados/` → `utils/dataframe.py` concatenates per region
2. `utils/converter_pandas.py` converts CSV → pandas pickle (`.pandas` format via `pd.to_pickle`)
3. Filtered pickles in `dados_filtrados/pandas_regioes/` — filename pattern: `transform-<region>_cellcenter.pandas`
4. `Tf_kfold.py` loads pickle, splits with `KFold`, trains, writes outputs to `regioes/<REGIAO>/results/<EXP>/`

> `dados_filtrados/`, `dados_n_filtrados/`, `results/`, and `logs/` are gitignored.

## Directory Structure

### Region Directories (`regioes/`)
Each body region has its own subdirectory with:
- `results/<EXP>/` — per-fold `.keras`, `_dataset.npz`, `_erros.npz`, `_history.npz` files
- `historico/hist_kfold_<EXP>.csv` — per-run CSV log (one row per fold, appended across runs)
- `logs/<EXP>/fold<NN>/` — TensorBoard event files
- `graficos/` — saved plots (generated by `gerar_graficos.py` and `analise/` scripts)

### Global Files
- `regioes/melhores_modelos.csv` — auto-appended by `Tf_kfold.py`; one row per training run with best fold info
- `regioes/graficos/` — cross-region comparison plots
- `regioes/analise_regioes_metricas.csv` — full metrics table (generated by `analise/analise_regioes.py`)

### Naming Conventions
- Experiment names follow pattern: `<ARCH>_<REGIAO>` (e.g., `512x4_fp_head`)
- Architecture shorthand: `NxM` = M layers of N neurons (e.g., `512x4`)
- `--Layers` argument is space-separated neuron counts (e.g., `"512 512 512 512"`)

## Analysis & Visualization Scripts

### Per-region plots (run after training completes):
```bash
python3 gerar_graficos.py                    # all regions in melhores_modelos.csv
python3 gerar_graficos.py fp_head mt_head    # specific regions
```
Generates: CDF histograms, pred-vs-real scatter, 2-variable coupling, 3D heatmap, 2D contour.

### Cross-region analysis:
```bash
python3 analise/analise_regioes.py           # heatmap R² / MAE por região×variável + bar val_loss
python3 analise/plot_estabilidade_folds.py   # boxplot + line plot of 30-fold stability per region
```

### Per-variable metrics for a single region:
```bash
python3 analise/analise_por_variavel.py                  # all regions
python3 analise/analise_por_variavel.py fp_head mt_head  # specific regions
```
Outputs MSE, MAE, R² per target variable for the best fold of each region.

### Experiment comparison (manual path editing required):
```bash
python3 analise/analise_stat.py   # boxplot val_loss across multiple experiments (fp_head only)
```
