# MACE Trainer Design

This package adapts MACE training to the active-learning `MLIPTrainer` interface.
The public entry point is `MACETrainer`; the other modules keep the pieces of the
workflow small enough to read and test in isolation.

## Main Flow

```text
DataAndSplittingStrategy
  |
  v
MACEDataSettings
  - resolves run paths
  - exports train/validation extxyz files
  - returns MACEPreparedTrainingContext
  |
  v
MACEConfigBuilder
  - resolves REF_* property keys
  - validates dipole-related model/loss settings
  - translates trainer settings into a MACE YAML config
  |
  v
runner.train_mace
  - calls mace.cli.run_train with the generated config
  - temporarily changes cwd and sys.argv for the MACE CLI
  - optionally patches MACE foundation-model loading during fine-tuning
  |
  v
MACETrainer
  - resolves the trained model artifact
  - builds an ASEEngine using AMSMACECalculator
  - persists optional trainer/results JSON
  |
  v
MLIPTrainerResults
```

## Module Responsibilities

`trainer.py`

Owns the active-learning-facing interface: `train`, `finetune`,
`to_data_format`, and result assembly. It should read like orchestration. If a
helper grows into domain logic, move it behind a focused module and keep a thin
compatibility wrapper only when existing callers need it.

`settings.py`

Defines typed Pydantic settings and data-preparation context. This is where
serialized trainer configuration belongs.

`config.py`

Builds MACE runtime config from trainer settings plus the prepared dataset
context. This module is the main seam for testing config translation without
running MACE.

`runner.py`

Contains the MACE CLI adapter and the temporary global-state changes needed by
`mace.cli.run_train`. Keep MACE monkey-patching here so it is easy to audit and
restore.

`metrics.py`

Parses MACE log output into active-learning metric rows and provides plotting
helpers for `*_train.txt` files.

`calculator.py`

Wraps MACE's ASE calculator for AMS usage and converts dipole-like outputs from
the trained model units into ASE `e*Ang`.

## Extension Notes

- Add new serialized configuration fields to Pydantic models in `settings.py`.
- Add new MACE YAML keys in `MACEConfigBuilder`, not directly in `MACETrainer`.
- Add new log formats in `MACETrainingMetrics`, with tests that use small sample
  log strings.
- Keep optional MACE imports lazy unless the module is itself only imported by
  runtime calculator execution.
