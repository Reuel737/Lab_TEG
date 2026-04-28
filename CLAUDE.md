# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IC (Iniciação Científica) project that trains neural networks to surrogate CFD (Computational Fluid Dynamics) simulations for thermal comfort prediction inside vehicle cabins. Each neural network learns to predict 9 physical quantities (pressure, velocities, temperature, radiation fields) from spatial coordinates and inlet boundary conditions.

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
    --FileOutPut "regioes/fp_head/results/512x4/512x4" \
    --HistCSV "regioes/fp_head/historico/hist_kfold_512x4.csv" \
    --KFolds 30 --MaxIter 5000 --BatchSize 250 --L2 0.0001
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

## Architecture

### Model (Tf_kfold.py / Tf.py)
- **Input** (6 features): `x-coordinate`, `y-coordinate`, `z-coordinate`, `Vel`, `Tinsu`, `Qinsu`
- **Output** (9 targets): `pressure`, `x-velocity`, `y-velocity`, `z-velocity`, `temperature`, `incident-radiation`, `radiation-temperature`, `rad-heat-flux`, `vr`
- **Architecture**: Keras `Sequential` — `Normalization` layer → N×`Dense(relu)` hidden layers → `Dense(linear)` output
- **Training**: Adam (lr=0.001) + `ReduceLROnPlateau` (factor=0.9, patience=50) + `EarlyStopping` (patience=200), MSE loss
- **K-Fold**: 30-fold CV with `random_state=42`; each fold saves `.keras` model, `_dataset.npz`, `_erros.npz`, `_history.npz`

### Data Flow
1. Raw CFD CSVs (cases 1–45) in `dados_n_filtrados/` → `utils/dataframe.py` concatenates per region
2. `utils/converter_pandas.py` converts CSV → pandas pickle (`.pandas` format via `pd.to_pickle`)
3. Filtered pickles in `dados_filtrados/pandas_regioes/` — filename pattern: `transform-<region>_cellcenter.pandas`
4. `Tf_kfold.py` loads pickle, splits with `KFold`, trains, writes outputs to `regioes/<REGIAO>/results/<EXP>/`

> `dados_filtrados/`, `dados_n_filtrados/`, `results/`, and `logs/` are gitignored.

### Region Directories (`regioes/`)
Each body region has its own subdirectory (`fp_head`, `lrp_head`, `mt_head`, `mt_core`, `mt_l_foot`, `mt_r_foot`, `outlet`, `rrp_head`) with:
- `results/<EXP>/` — per-fold `.keras`, `_dataset.npz`, `_erros.npz`, `_history.npz` files
- `historico/hist_kfold_<EXP>.csv` — per-run CSV log (appended across runs)
- `logs/<EXP>/fold<NN>/` — TensorBoard event files
- `graficos/` — saved plots

### Global Registry
`regioes/melhores_modelos.csv` — auto-appended by `Tf_kfold.py` with the best fold per experiment run.

### Naming Conventions
- Experiment names: `expXX` or `expXX_L2_XXXXX` (e.g., `exp15_L2_00001`)
- Architecture shorthand: `NxM` = M layers of N neurons (e.g., `512x4`)
- `--Layers` argument is space-separated neuron counts (e.g., `"512 512 512 512"`)

## Analysis & Visualization Scripts

These scripts require manual path editing before running:

| Script | Purpose |
|--------|---------|
| `analise/analise_por_variavel.py` | Per-target MSE, MAE, R² from a saved model + dataset |
| `analise/analise_stat.py` | Boxplot/scatter statistical comparison across experiments |
| `plots/plot_heatmap.py` | 3D scatter: CFD real vs NN prediction vs absolute error |
| `plots/plot_hist.py` | Cumulative CDF of absolute errors per target variable |
| `plots/plot_contour.py` | 2D contour plot via griddata interpolation on a cross-section |
