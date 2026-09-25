# Workspace for active learning components

This workspace contains relevant libraries constructed around active learning and vibrational spectroscopy.

The `active_learning` package contains the ALIR paper's active-learning workflow settings. Its current
reference engine is GFNFF rather than the paper's ADF XLYP/DZP level of theory, so quantitative agreement with
the paper should be evaluated separately.

## Citation

If you use this repository, please cite the associated ALIR paper. Bibliographic details will be added here:

```bibtex
@article{ALIR_PAPER,
  author  = {??to/be/filled??},
  title   = {??to/be/filled??},
  journal = {??to/be/filled??},
  year    = {??to/be/filled??},
  doi     = {??to/be/filled??},
}
```

## Repos

- [active_learning](/active_learning/): Automates active learning workflows for atomistic systems, combining data generation, model training, task scheduling, and unified inspection of simulation results.
- [moliterate](/moliterate/): Provides unified access to molecular datasets, combining atomic structures, metadata, and computed properties with modular filtering and collection inspection tools.

## Installation

Fork this repository on GitHub, then clone your fork to install or develop `active_learning`:
```bash
git clone <your-fork-url>
```

Installing a github subdirectory. For example if you want to install only moliterate:
```bash
uv add git+https://github.com/SCM-NV/active_learning_workspace.git#subdirectory=moliterate
```

## Environment setup

From the workspace root, authenticate uv with the SCM package index, then create the environment with the AMS and MACE extras:

```bash
"$AMSBIN/uv" auth login "https://downloads.scm.com/Downloads/packages/uv/channels/2026.1/simple/"
uv lock --upgrade
uv sync --extra ams --extra mace
```

Run the commands below from the workspace root. `uv run` uses the workspace environment, so activation is not required.
