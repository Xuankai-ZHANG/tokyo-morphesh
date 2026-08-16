# tokyo-morphesh

**Morphology-enabled local Energy Sharing** — code and data to reproduce the analysis in
*"Rooftop solar expansion places surplus beside surplus in cities"*: rooftop-PV energy sharing
across regional meshes in Japan's Kanto region, its spatial-equity effects, and urban-form
counterfactuals.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python code/simulation/run_simulation.py --mode smoke     # 2-day engine check
python code/simulation/run_simulation.py --mode primary   # full 7-scenario simulation
python code/simulation/run_metrics.py                     # system + distribution metrics
python code/scenario/scenarios.py                         # urban-form counterfactuals
python code/scenario/sensitivity.py                       # demand response + deployment
python code/scenario/decomposition.py                     # band + additive decomposition
python code/figures/plot_figure2.py                       # manuscript figures 2-4
python code/figures/plot_figure3.py
python code/figures/plot_figure4.py
```

## Data

The derived input, result, and figure-staging data (~3.4 GB) are **not
included in this repository** because of their size and upstream licensing
status. The simulation engine and scripts are self-contained; contact the
author for access to the data.

## License

Code: MIT (`LICENSE`).
