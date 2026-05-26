# The Most Probable Path (MPP) Algorithm Package

MPP is a package for reducing the number of states in a Markov state trajectory using the
most probable path (MPP) algorithm. Based on a microstate trajectory, a Markov state model
is estimated and transition probabilities (and optional geometric descriptors) are used to
iteratively merge states until the resulting macrostates satisfy minimum population and
metastability thresholds.

## Installation

Install from the [Python Package Index](https://pypi.org/project/mpp-lumping/) via pip:

```bash
pip install mpp-lumping
```

Python 3.10 or later is required. The package is tested with Python 3.12, 3.13, and 3.14.

## Features

- Perform the MPP algorithm on microstate trajectories
- Multiple similarity modes:
    - Transition probability (`T`)
    - Kullback-Leibler divergence of transition probabilities (`KL`)
    - Jensen-Shannon divergence of feature distributions (`JS`, combined with `T` or `KL`)
    - Feature-only mode (`none`/`JS`)
- Stochastic lumping (top-N or probability-mass-based)
- G-PCCA as an alternative coarse-graining method
- Variety of analysis and visualization plots (dendrogram, timescales, Sankey, contact maps, CK test, state network, and more)
- Multi-trajectory support
- Three levels of user interface:
    - Python API (`MPP.Lumping`, `MPP.run.Data`)
    - Command-line interface (`python -m MPP.run`)
    - Snakemake workflow

## About

MPP is developed by the [Molecular Dynamics Group](https://www.moldyn.uni-freiburg.de/)
at the University of Freiburg.

## Usage Guides

- [CLI Usage](usage_cli.md) — command-line interface, YAML config, kernel options, plot types
- [Python API Usage](usage_api.md) — `MPP.Lumping`, kernels, result attributes, plots
- [Snakemake Workflow](usage_snakemake.md) — workflow structure, rule overview, expected outputs
