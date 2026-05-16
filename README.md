# Ecohydro Regulation Diagnosis

This repository contains the core implementation of a dynamic ecohydrological regulation framework for targeted water management. The model is designed to diagnose where and when ecohydrological water regulation is most needed by integrating natural hydroclimatic drivers, vegetation dynamics, human water-use processes, spatial priors, temporal state information, and causal prior masks.

The code supports the study:

**A dynamic ecohydrological regulation framework integrating causal learning and human water-use processes for targeted water management**

## Purpose

This project is not intended as a generic deep-learning benchmark. Its main goal is to support ecohydrological regulation diagnosis in semi-arid basins by:

- simulating daily soil moisture dynamics;
- incorporating human water-use sectors as explicit dynamic inputs;
- using spatial, temporal, and causal priors to guide dynamic encoding;
- partitioning soil moisture changes into natural, vegetation-related, and anthropogenic components;
- identifying regulation-relevant parameters, priority zones, and intervention timing.

## Core modules

### `config.py`

Centralized experiment configuration, including data dimensions, model hyperparameters, ODE constraints, human water-use interpolation options, training settings, and output paths.

### `module1_data.py`

Data preprocessing and tensor construction module. It aligns soil moisture targets, static terrain and soil attributes, dynamic natural drivers, causal prior masks, and human water-use inputs on consistent grid and time axes. It also supports monthly-to-daily interpolation of human water-use data and stratified spatial splitting by hydro-geomorphological clusters.

### `module23_model.py`

Core model components for spatial and dynamic encoding. It includes:

- `GeoHypernet` for static spatial context encoding and FiLM modulation;
- simplified KAN/MLP alternatives for static feature representation;
- causal-prior-gated dynamic encoding;
- Mamba-style sequence modeling with fallback support when `mamba_ssm` is unavailable.

### `module45_train.py`

Training, evaluation, and differentiable ODE decoding module. It includes:

- FiLM-based spatiotemporal coupling;
- a white-box ODE decoder;
- closed-loop soil moisture integration;
- regulation-related parameter outputs;
- mini-batch grid-parallel training;
- model evaluation and result export.

### `main.py`

Main execution script that connects all modules into a runnable pipeline. It loads gridded dynamic data, K-means spatial clusters, HMM temporal states, J-PCMCI+ causal edge statistics, and monthly human water-use data, then trains and evaluates the Causal-Gated Differentiable Mamba-ODE model.

## Model inputs

The model uses:

- static variables: terrain, soil properties, spatial location, and cluster labels;
- natural dynamic drivers: precipitation, LST, PET, and LAI;
- temporal encodings: day-of-year and normalized time features;
- causal prior masks from J-PCMCI+;
- human water-use inputs: irrigation, manufacturing, domestic, and thermal power water withdrawal;
- target variable: daily surface soil moisture.

## Outputs

The pipeline can export:

- soil moisture predictions;
- grid-level evaluation metrics;
- dynamic regulation-related parameters;
- natural, vegetation, and human water-use flux components;
- parameter fields for subsequent regulation-priority diagnosis.

## Notes

The code is structured for research reproducibility and paper-related analysis. File paths and local data directories should be modified according to the user's working environment before running.
