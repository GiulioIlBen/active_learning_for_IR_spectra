# MLIP Package Rules

This directory defines the canonical MLIP trainer package. Prefer editing the
modules under `mlip/`; the old `mlip_trainer/` package has been retired.

## Layout

- `core.py` and `splitting_strategy.py` re-export the shared trainer base types.
- Each implementation lives in its own folder:
  - `params_trainer/`
  - `mace_trainer/`
  - `schnetpack_trainer/`
- Calculators should live beside the implementation they belong to, not in
  `engines/`.
- Every implementation folder should include a private `_optional_dependencies.py`
  module when it imports optional backends, helper packages, or calculator base
  classes.

## Registration and Compatibility

- `mlip/__init__.py` is the canonical registry module.
- Keep `ConcreteMLIPTrainer` as the validation seam used by the loop and by
  `TypeAdapter(...)`.
- Update the returned calculator module strings to the new canonical locations
  in the implementation modules.
- Keep package `__init__.py` files import-safe:
  - do not import calculators from package `__init__.py`;
  - do not import optional backends at module import time;
  - keep re-exports limited to trainer/data model symbols that can be imported
    safely in the base environment.

## Editing Rules

- When moving trainer code, keep the implementation and its calculator in the
  same implementation folder.
- If an implementation has large helper clusters, split them into local helper
  modules inside that implementation folder instead of growing the trainer
  module further.
- Keep optional dependency loading local to the trainer/calculator module that
  actually needs it. Import optional packages through a named loader in
  `_optional_dependencies.py` at the specific call site, not at package import
  time.
- Prefer a small set of named loaders per implementation plus one shared error
  formatter in `_optional_dependencies.py`. Use short, context-aware error
  messages that tell users which extra to install.
- If a trainer or calculator needs a shared optional dependency pattern across
  sibling modules, centralize it in that implementation's
  `_optional_dependencies.py` rather than duplicating `importlib` logic.
- When extracting a loader helper, keep compatibility wrappers only if tests or
  downstream code already reach that method name directly.
- Add `pytest.importorskip(...)` guards in tests that import optional calculators
  or backend-specific modules at collection time.
- Keep base package imports working without optional extras installed; if a
  change breaks `import scm.active_learning.mlip`, it is too eager.
- Do not add new code under `mlip_trainer/`; move implementation code here
  instead.
