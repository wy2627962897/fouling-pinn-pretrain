# Multi-Mechanism Pre-training for Robust Physics-Informed Fouling Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

Repository for the paper: **"Multi-Mechanism Pre-training Enables Robust Physics-Informed Fouling Prediction under Sparse Industrial Data"** (Submitted to *Applied Thermal Engineering*, 2026).

## Overview

Heat exchanger fouling costs the process industry billions annually. This work proposes a multi-mechanism pre-training + sparse fine-tuning framework for Physics-Informed Neural Networks (PINNs) applied to fouling resistance prediction $R_f(t)$.

**Key finding**: Diversity in pre-training physics prevents catastrophic over-specialization. Single-mechanism (Kern-Seaton only) pre-training achieves $R^2 = -1329$ on its own target curve type; multi-mechanism pre-training reaches $R^2 = 0.94$ on the same curve.

## Repository Structure

```
├── paper.tex / paper.pdf          # LaTeX manuscript
├── paper_draft.md                 # Markdown draft
├── figures/                       # Publication figures (6 PNGs)
├── code/
│   ├── pinn_model.py              # Multi-ODE PINN (pre-training + fine-tuning)
│   ├── fouling_models.py          # 5 fouling ODE models (unified interface)
│   ├── fouling_simulator.py       # Kern-Seaton simulator
│   ├── multi_ode_data_generator.py # Synthetic data generation
│   ├── pretrain_finetune_experiments.py  # Full experiment pipeline
│   ├── extrapolation_experiment.py       # Extrapolation test
│   ├── multiseed_experiment.py           # Multi-seed validation
│   ├── baseline_ode_rf_fair.py           # ODE-fit baseline + RF comparison
│   └── output/                    # Results, figures, pre-trained models
├── data/real/                     # 18 digitized real fouling curves
└── data_sources.md                # Data source documentation
```

## Quick Start

```bash
# Requirements
pip install torch numpy scipy pandas scikit-learn matplotlib xgboost

# Generate pre-training data
cd code && python multi_ode_data_generator.py

# Run full experiment pipeline (~35 min on RTX 4060)
python pretrain_finetune_experiments.py
```

## Real Fouling Curve Dataset

18 curves digitized from 3 publications: Jradi et al. (2018, phosphoric acid), Benyahia et al. (2014, crude oil refinery), and Riihimäki et al. (2011, CaCO₃ crystallization). Curves in `data/real/curves/cleaned/`.

## Results Summary

| Metric | Value |
|--------|-------|
| Curves with PINN mean $R^2 > 0.90$ (N=15, 5 seeds) | 12/18 |
| Extrapolation advantage over RF (N=5) | $\Delta R^2$ +1.3 to +18.7 |
| Cleaning cost savings vs fixed-interval | 20–97% |

## Citation

```bibtex
@article{liu2026multi,
  title={Multi-Mechanism Pre-training Enables Robust Physics-Informed
         Fouling Prediction under Sparse Industrial Data},
  author={Liu, Jiancheng},
  journal={Applied Thermal Engineering},
  year={2026},
  note={Under review}
}
```

## Contact

Jiancheng Liu — 3240101908@zju.edu.cn  
College of Chemical and Biological Engineering, Zhejiang University
