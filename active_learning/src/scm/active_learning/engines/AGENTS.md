# Engine Extension Rules

This package defines concrete implementations of `Engine` from `core.py`.
New engines must satisfy the `Engine` protocol, be Pydantic-serializable, and be explicitly registered in the discriminated union.

## Required Contract (`core.py`)

Every new engine must subclass:
- `Engine[T]` where `T` is a `ParallelStrategy` subtype accepted by `run_single_point`.

Required fields and behavior:
- `engine_id: str` is mandatory and must not contain `"-"` (enforced by `Engine._validate_engine_id`).
- `type: Literal["YourEngineType"]` must be declared on the concrete model for union discrimination.
- `_validate_properties(self, requested_properties: List[PropertyInfo]) -> List[str]` must reject unsupported properties via `ValueError`.
- `run_single_point(self, properties, dataset, parallel_settings, **kwargs)` must return an iterable of dict results aligned with input order.
- On failed evaluations, return `{"FAILURE": "<message>"}` for that entry (consumed by `SPLabeller`).

Optional hooks:
- Override `is_finetunable()` when the engine can be used in finetune flows.
- Implement `training_dataset` / `validation_dataset` when the engine carries reusable dataset provenance for splitting/import.
- Override `settings` when serialized model state is not directly usable as PLAMS `Settings`.

## Existing Patterns To Follow

- `AMSEngine` stores backend config in `source_settings: Dict[str, JsonValue]` and exposes PLAMS `Settings`.
- `ParAMSEngine` wraps MLIP production settings and delegates single-point logic to `ams_engine`.
- Optional dependencies are guarded (`try/except ImportError`) with actionable runtime errors in classmethods that require them.
- Keep runtime methods pure with respect to model state; use computed properties for derived objects (`settings`, `ams_engine`, datasets).

## Registration Checklist

When adding `YourEngine`:
1. Add the class in a new module under `engines/`.
2. Import it in `engines/__init__.py`.
3. Extend `ConcreteEngines = Annotated[Union[...], Field(discriminator="type")]`.
4. Export it in `__all__`.

Without step 3, Pydantic model loading of loop/trainer/result models will fail for the new engine type.

## AMS Compatibility Constraint

Large parts of the workflow currently assume AMS-compatible engines:
- `to_ams_engine` only accepts `AMSEngine | ParAMSEngine`.
- `AMSTask`, `AMSConformersTask`, and `AMSParallelStrategy.run_tasks` use `to_ams_engine(...)`.

If the new engine is not AMS-compatible, also update those call sites and type unions, or keep it restricted to components that only need generic `Engine.run_single_point` (for example `SPLabeller`).

## Validation Scope

For changes in this folder, run targeted checks:
- tests covering engine serialization/discrimination and single-point execution paths;
- related task-parallelization tests when modifying AMS compatibility logic.
