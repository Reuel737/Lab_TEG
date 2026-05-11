# AI CFD Surrogate Models

Deep learning surrogate models for accelerating CFD-based thermal comfort analysis inside vehicle cabins.

## Overview

This project explores the use of Fully Connected Neural Networks (FCNNs / MLPs) as surrogate models for Computational Fluid Dynamics (CFD) simulations of vehicle cabin environments.

The objective is to reproduce CFD field variables from spatial coordinates and HVAC boundary conditions without re-running expensive CFD simulations, enabling near real-time predictions for thermal comfort and engineering analysis workflows.

This research is part of an undergraduate scientific research project (Iniciação Científica) at UFSC Joinville and follows the methodological framework proposed by Prof. Murilo Brunazzo Medeiros in his dissertation on CFD and deep learning for thermal comfort analysis.

---

## Motivation

High-fidelity CFD simulations may require hours or even days to compute for each new boundary condition configuration.

By training neural networks to learn the mapping:

(x, y, z, Vel, Tinsu, Qinsu) → CFD Field Variables

the model can infer fluid and thermal fields in milliseconds, enabling:

- Fast thermal comfort evaluation
- Parametric HVAC studies
- Real-time engineering analysis
- AI-assisted simulation workflows
- Reduced computational costs

---

# Scientific Context

This IC (Iniciação Científica) is part of a broader research effort focused on applying deep learning techniques to accelerate thermal comfort analysis in automotive cabin CFD simulations.

The project implements surrogate neural networks capable of approximating CFD field variables based on:
- spatial coordinates
- airflow conditions
- thermal boundary conditions

instead of running expensive CFD simulations for every new operating condition.

The methodology follows the framework established in the dissertation:

“Análise de parâmetros de conforto térmico de cabine por meio de simulações CFD e redes neurais profundas”  
Murilo Brunazzo Medeiros — UFSC (2025)

---

# Physical Simulation Context

## Vehicle Cabin CFD Model

- Vehicle cabin geometry reconstructed from Mao, Ji Wang & Junming Li (2018)
- ANSYS Fluent solver
- Steady-state RANS k–ω SST turbulence model
- Discrete Ordinates Method (DOM) for radiation
- 45 CFD simulation cases generated through Latin Hypercube Sampling (LHS)

---

## Boundary Conditions

| Variable | Description | Unit |
|----------|-------------|------|
| Vel | Air velocity at inlet ducts | m/s |
| Tinsu | Inlet air temperature | K |
| Qinsu | Inlet air volume flow rate | m³/s |

---

# Predicted CFD Variables

The neural networks predict 9 CFD field variables simultaneously:

| Variable | Description |
|----------|-------------|
| pressure | Gauge static pressure |
| x-velocity | Velocity component in X |
| y-velocity | Velocity component in Y |
| z-velocity | Velocity component in Z |
| temperature | Air temperature |
| incident-radiation | Incident thermal radiation |
| radiation-temperature | Radiation temperature |
| rad-heat-flux | Net radiative heat flux |
| vr | Resultant velocity magnitude |

These variables are relevant for:
- PMV/PPD thermal comfort indices
- Mean Radiant Temperature (MRT)
- HVAC analysis
- Cabin thermal optimization

---

# Neural Network Architecture

Each body region is modeled independently using Fully Connected Neural Networks (MLPs).

## Inputs

- X coordinate
- Y coordinate
- Z coordinate
- Vel
- Tinsu
- Qinsu

## Outputs

9 CFD field variables

---

## Architecture

- TensorFlow / Keras
- Sequential FCNN architecture
- Normalization layer
- Dense(ReLU) hidden layers
- Linear output layer
- Adam optimizer
- EarlyStopping
- ReduceLROnPlateau

---

## Current Best Architecture

512x4

Four hidden layers with 512 neurons each consistently achieved the best validation performance across regions.

### Performance Evolution

| Architecture | Validation Loss |
|--------------|----------------|
| 128x5 | ~0.96 |
| 256x5 | ~0.45 |
| 512x4 | ~0.30 |

---

# Body Regions

The cabin is divided into independent thermal regions:

| Region | Description |
|--------|-------------|
| fp_head | Front passenger head |
| lrp_head | Left rear passenger head |
| rrp_head | Right rear passenger head |
| mt_head | Driver head |
| mt_core | Driver torso/core |
| mt_l_foot | Driver left foot |
| mt_r_foot | Driver right foot |
| outlet | Air outlet region |

Independent models are trained because each region presents significantly different flow and thermal characteristics.

---

# Data Pipeline

## Workflow

1. Raw CFD CSV files generated from ANSYS Fluent
2. Concatenation and preprocessing
3. Conversion to pandas pickle format
4. Region-based filtering
5. K-Fold training
6. Metric generation and visualization

---

## Data Flow

Raw CFD CSVs
↓
utils/dataframe.py
↓
utils/converter_pandas.py
↓
Filtered region datasets (.pandas)
↓
Tf_kfold.py
↓
Training + Metrics + Saved Models

---

# Repository Structure

regioes/
├── fp_head/
├── mt_head/
├── mt_core/
├── mt_l_foot/
├── mt_r_foot/
├── lrp_head/
├── rrp_head/
└── outlet/

analise/
utils/
runs/

---

# Setup

## Create virtual environment

python3 -m venv venv
source venv/bin/activate

## Install dependencies

pip install -r requirements.txt

---

# Running Experiments

## K-Fold Training

bash runs/run_kfold.sh

Or directly:

python3 Tf_kfold.py \
    --Layers "512 512 512 512" \
    --DataFile "dados_filtrados/pandas_regioes/transform-fp-head_cellcenter.pandas" \
    --FileOutPut "regioes/fp_head/results/512x4_fp_head/512x4_fp_head" \
    --HistCSV "regioes/fp_head/historico/hist_kfold_512x4_fp_head.csv" \
    --KFolds 30 \
    --MaxIter 5000 \
    --BatchSize 500

---

# TensorBoard

tensorboard --logdir regioes/<REGIAO>/logs

---

# Analysis Scripts

## Per-region plots

python3 gerar_graficos.py

Generates:
- CDF histograms
- Prediction vs real scatter plots
- 2-variable coupling plots
- 3D heatmaps
- 2D contour maps

---

## Cross-region analysis

python3 analise/analise_regioes.py

Outputs:
- R² heatmaps
- MAE heatmaps
- Validation loss comparisons

---

## Per-variable metrics

python3 analise/analise_por_variavel.py

Outputs:
- MSE
- MAE
- R² metrics per target variable

---

# Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- ANSYS Fluent
- CFD
- Deep Learning

---

# Research Applications

This project contributes to:
- AI-assisted CFD acceleration
- Thermal comfort prediction
- HVAC optimization
- Engineering-oriented AI systems
- Reduced-order modeling
- Surrogate modeling
- Real-time engineering analysis

---

# Future Work

- Physics-Informed Neural Networks (PINNs)
- Transformer-based architectures
- Unified multi-region models
- Real-time inference systems
- Thermal comfort prediction pipelines
- AI-assisted engineering optimization

---

# Author

## Reuel Fernandes

Aerospace Engineering — UFSC Joinville

Performance Engineering Team — Nisus Aerodesign

Focused on:
- AI-assisted engineering workflows
- CFD acceleration
- automation systems
- backend systems
- intelligent operational tools
