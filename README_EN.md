# Multi-Mechanism Pre-Training of Physics-Informed Neural Networks for Fouling Resistance Reconstruction from Sparse Measurements

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)

Repository for the paper: **"Multi-Mechanism Pre-Training of Physics-Informed Neural Networks for Fouling Resistance Reconstruction from Sparse Measurements"** *(manuscript in preparation)*.

## Overview

Heat exchanger fouling costs the process industry billions annually. This work proposes a multi-mechanism pre-training + sparse fine-tuning framework for Physics-Informed Neural Networks (PINNs) applied to fouling resistance curve reconstruction $R_f(t)$ from sparse measurements ($N = 5$--$30$ points).

**Key finding**: Diversity in pre-training physics prevents catastrophic over-specialization. Single-mechanism (Kern-Seaton only, 200 trajectories) pre-training achieves $R^2 = -1329$ on its own target curve type; multi-mechanism pre-training reaches $R^2 = 0.91$ on the same curve. At equal volume (2,000 trajectories each), single-mechanism reaches $R^2 = 0.62$, while multi-mechanism reaches $R^2 = 0.91$ ($\Delta R^2 = +0.29$ from mechanism diversity).

**Honest assessment**: For pure interpolation, classical methods (smoothing spline, GP-Matérn) achieve median $R^2 \approx 0.99$ at $N=15$ --- competitive with or exceeding the PINN. The PINN's value lies in physics-grounded representations, downstream cleaning decision integration, and relative extrapolation robustness, not universal interpolation superiority. Absolute extrapolation remains unsolved for all methods ($R^2 < 0$).

## Repository Structure

```
├── paper.tex / paper.pdf               # LaTeX manuscript
├── supplementary_materials.tex / .pdf   # Supplementary Tables S1, S2
├── figures/                             # Publication figures
├── code/
│   ├── pinn_model.py                    # Multi-ODE PINN (pre-training + fine-tuning)
│   ├── fouling_models.py                # 5 fouling ODE models (unified interface)
│   ├── multi_ode_data_generator.py      # Synthetic data generation
│   ├── pretrain_finetune_experiments.py # Full experiment pipeline
│   ├── extrapolation_experiment.py      # Extrapolation test
│   ├── multiseed_experiment.py          # Multi-seed validation
│   ├── baseline_ode_rf_fair.py          # ODE-fit baseline + RF comparison
│   ├── baseline_spline_gp.py            # Spline & GP baselines (Supplementary)
│   ├── auto_ode_selection_ablation.py   # N-point-only ODE auto-selection
│   ├── fix_economic.py                  # Standalone economic analysis
│   ├── generate_paper_figures.py        # Paper figure generation
│   └── output/                          # Results, figures, pre-trained models
├── data/real/                           # 18 digitized literature fouling curves
├── requirements.txt                     # Python dependencies
└── chinese/                             # Chinese translation (reference only)
```

## Quick Start

```bash
# Requirements
pip install -r requirements.txt

# Generate pre-training data
cd code && python multi_ode_data_generator.py

# Run full experiment pipeline (~35 min on RTX 4060)
python pretrain_finetune_experiments.py

# Run supplementary experiments
python auto_ode_selection_ablation.py
python baseline_spline_gp.py
python fix_economic.py
```

## Real Fouling Curve Dataset

18 curves digitized from 3 peer-reviewed publications:
- Jradi et al. (2018) — phosphoric acid crystallization (1 curve)
- Benyahia et al. (2014) — crude oil refinery E101 (2 curves)
- Riihimäki et al. (2011) — CaCO₃ crystallization with coatings/surface treatments (15 curves)

Predominantly laboratory-scale studies. Crude oil curves are the closest to industrial conditions.

## Results Summary

| Metric | Value |
|--------|-------|
| Curves with PINN mean $R^2 > 0.90$ ($N=15$, 5 seeds) | 12/18 |
| ODE-fit baseline failure rate (sparse data) | 15/18 curves |
| Multi-mechanism diversity gain over equal-volume single-mechanism | $\Delta R^2 = +0.29$ |
| Extrapolation: PINN beats RF ($N=5$, median $\Delta R^2$) | +1.8 (6/8 curves; absolute $R^2$ negative for all) |
| Smoothing spline median $R^2$ at $N=15$ | 0.99 |
| GP-Matérn median $R^2$ at $N=15$ | 0.99 |
| Auto ODE selection accuracy ($N$-point-only, geometric heuristic) | 87% (47/54 trials) |
| Illustrative cleaning cost reduction (curves with $R^2>0.80$) | 20--67% vs. fixed-interval |

## Contact

Jiancheng Liu — 3240101908@zju.edu.cn  
College of Chemical and Biological Engineering, Zhejiang University, Hangzhou 310058, China
