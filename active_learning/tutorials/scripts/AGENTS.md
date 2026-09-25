# Tutorial Script Rules

Use this file for work in `active_learning/tutorials/scripts/`.
Also follow the parent project rules in `active_learning/AGENTS.md`.

## Purpose

Scripts in this folder are tutorials first.
Prefer readable, copyable examples over generic frameworks or CLI-heavy utilities.

## Style For Tutorial Scripts

- Write tutorial scripts as plain Python modules.
- Prefer a simple `main(...)` function and small helper functions over command-line parsing.
- Do not add `argparse` unless the user explicitly asks for a CLI.
- Do not use module-level configuration globals for tutorial choices such as dataset path, trainer selection, dry-run mode, or output folder.
- Put tutorial configuration in function arguments, typically on `main(...)`.
- Keep the script runnable with `if __name__ == "__main__": main()`.

## Pydantic / Trainer Initialization

- When initializing trainers or other Pydantic models, prefer direct construction plus attribute assignment.
- Preferred style:
  ```python
  trainer = MACETrainer()
  trainer.train_settings.max_num_epochs = 50
  trainer.train_settings.batch_size = 5
  ```
- Avoid large nested constructor dictionaries for tutorial setup unless explicitly requested.
- If assignment-time validators matter, order attribute assignments so intermediate invalid states are avoided.

## Organizing Trainer Examples

- If a tutorial presents multiple trainer presets/examples, group those constructors in a class with `@staticmethod`s.
- Preferred pattern:
  ```python
  class TrainerExamples:
      @staticmethod
      def get_mace_trainer(output_root: Path) -> MACETrainer:
          ...
  ```
- Keep `get_trainer(...)` as a small dispatcher.

## Typing

- Tutorial choice names such as `example_name` should use `Literal[...]` aliases instead of plain `str`.
- Preferred pattern:
  ```python
  ExampleName = Literal["maceEFD", "m3gnetParAMS", "maceEFD_c"]
  ```

## Data Loading / Tutorial Behavior

- Prefer loading real tutorial data directly when the tutorial is about using existing AL output.
- If one trainer only supports a subset of properties, create a restricted dataset view in a dedicated helper instead of mutating the original split in place.
- Prefer `dry_run=True` by default for tutorial scripts that would otherwise launch expensive training.

## Output / Readability

- Print short dataset and trainer summaries so users can inspect what will run.
- Keep helper names descriptive and tutorial-oriented.
- Avoid unnecessary abstractions, registries, and indirection unless they clearly improve readability.

## Validation

- Run Ruff on changed tutorial scripts when practical.
- Prefer lightweight validation such as import checks or dry runs over full expensive training unless the user asks for the training run.
