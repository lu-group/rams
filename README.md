Keep the following line while the repository is under construction. Remove it after approval by Lu.

> Under Construction

# RAMS

The data and code for the paper [RAMS: Residual‐Based Adversarial‐Gradient Moving Sample Method for Scientific Machine Learning in Solving Partial Differential Equations](http://doi.org/10.1002/aidi.202500214).

## Data

Datasets are provided in the corresponding subfolders or can be generated using the scripts therein.

## Code

This repository implements **RAMS** (Residual-based Adaptive Sampling) and related methods for physics-informed neural networks (PINNs) and operator learning. It supports multiple sampling strategies (e.g., RAR, RAD, R3, Random, LHS, Halton, Sobol) and compares their performance with or without trainable sampling.

### Repository structure

- **`src/`** — Core library: training loop, networks (FCNN, DeepONet, ResFCNN, CNN), samplers, visualization, and I/O.
- **`pinn/`** — Physics-Informed Neural Networks: standard PINN benchmarks (e.g., Burgers’ equation) and experiments on high-dimensional PDEs.
- **`piol/`** — Physics-Informed Operator Learning: physics-informed operator learning for the Poisson equation and a tricky ODE problem from [arXiv:2402.11283](https://arxiv.org/abs/2402.11283), where conventional sampling methods perform poorly.
- **`ddol/`** — Data-Driven Operator Learning: RAMS applied to a 1D wave equation and a 2D Burgers equation.

Detailed experimental setups are given in our paper: <http://doi.org/10.1002/aidi.202500214>.

### How to run

1. Install dependencies (from project root):

   ```bash
   pip install -r requirements.txt
   ```

2. Run an experiment by executing the corresponding `run.py` from the project root, for example:

   ```bash
   python pinn/burgers/run.py
   python piol/ol_ode/run.py
   python ddol/wave_eq/run.py
   ```

   Each subfolder contains its own `run.py`, loss modules, and evaluation scripts.

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{author2000title,
  author  = {Weihang Ouyang, Min Zhu, Wei Xiong, Si-Wei Liu, and Lu Lu},
  title   = {RAMS: Residual‐Based Adversarial‐Gradient Moving Sample Method for Scientific Machine Learning in Solving Partial Differential Equations},
  journal = {Advanced Intelligent Discovery},
  year    = {2026},
  doi     = {http://doi.org/10.1002/aidi.202500214}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
