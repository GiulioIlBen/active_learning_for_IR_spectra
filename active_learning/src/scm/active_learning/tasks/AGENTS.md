# Task Extension Rules

This package defines concrete implementations of `Task` from `core.py`.
New tasks must be Pydantic-serializable, return a concrete task-result model, and be explicitly
registered in the discriminated unions used throughout the loop.

## Required Contract (`core.py`)

Every new task must subclass:
- `Task[T]` where `T` is the engine type the task expects.

Required fields and behavior:
- Declare `type: Literal["YourTaskType"]` on the concrete model for union discrimination.
- Keep `task_id: str` on the model.
- Implement `system_id` as a property that returns the logical system identifier for this task.
- Implement `run(self, engine: T | Engine, *args, **kwargs) -> ConcreteTaskResult`.
- Validate backend compatibility at runtime and raise `TypeError` on engine mismatch.
- Return a concrete result model from `src/scm/active_learning/results/task/`; do not return raw
  PLAMS, ASE, torch-sim, or dataset objects directly.

## Patterns To Follow

- Keep serialized configuration on model fields and construct runtime objects through properties or
  private helpers.
  Examples: `AMSTask.settings`, `AMSMDTask.ams_task`, `TorchSim*Task` helper builders.
- Prefer composition when the reusable execution logic is not itself the serialized task.
  Examples: `SPLabellerData` wraps `SPLabeller`; `AMSMDTask` delegates to `AMSTask`.
- For complex task schemas, use `ConfigDict(extra="forbid")` and Pydantic validators to reject
  invalid combinations early.
- For new mutable defaults, prefer `Field(default_factory=...)` over bare `{}` or `[]`.
- Guard optional dependencies inside helper functions and raise actionable `ImportError`s.
  Keep the base package importable without optional backends installed.
- Reject unexpected runtime kwargs after consuming the supported overrides.
- Keep `run()` short; move backend-specific setup into small helpers when the flow gets longer.

## Result Model Expectations

Every new task should have a matching result model under `results/task/`.

Result rules:
- Add a unique `type: Literal["YourTaskResultType"]` discriminator.
- Store both `task` and `engine`.
- Implement `from_task` and `from_engine` properties.
- Include the minimal metadata that downstream checker/getter logic will need.
- Make backend-object serialization explicit with `field_serializer` / `field_validator` or by
  storing plain paths/metadata.

If the task writes runtime artifacts:
- Return the artifact paths on the result model.
- Register new serialized path-bearing fields in
  `src/scm/active_learning/logging/state_paths.py::PATH_POLICY_REGISTRY` so state-file relocation and
  runtime path collection keep working.

## Registration Checklist

When adding `YourTask`:
1. Create the task class in a module under `tasks/`.
2. Import it in `tasks/__init__.py`.
3. Extend `ConcreteTask = Annotated[Union[...], Field(discriminator="type")]`.
4. Export it in `tasks/__init__.py::__all__`.
5. Create the matching result model in `results/task/`.
6. Import that result in `results/task/__init__.py`.
7. Extend `ConcreteTaskResult = Annotated[Union[...], Field(discriminator="type")]`.
8. If the task or its result introduces any serialized filesystem path field, add a matching
   pattern to `logging/state_paths.py::PATH_POLICY_REGISTRY`.

Without steps 3 and 7, serialized loop state and persisted task results will not load back through
Pydantic. Without step 8, path rewriting and relocation logic will silently miss the new field.

## Engine Compatibility Notes

- Use a concrete engine subtype when the backend is specific (`AMSEngine`, `ASEEngine`,
  `TorchEngine`).
- Use generic `Engine` only when the task depends on the common engine protocol.
  Example: `SPLabellerData` only needs `run_single_point(...)`.
- AMS-backed tasks commonly encode `engine_id-system_id-task_id` into job names and loaders split
  that string back into IDs. If a new AMS-style task follows that convention, keep those IDs
  hyphen-free or update the matching loader/parser logic as well.

## Serialization-Friendly Nested Models

If a task carries configurable nested actions, stops, reporters, or attachments:
- model them as Pydantic classes instead of ad hoc dicts;
- give each concrete nested variant its own `type` discriminator when it must round-trip through
  JSON;
- keep the nested union close to the task module.

Example: `ASEMolecularDynamics.attachments`.

## Validation Scope

For changes in this folder, add focused tests for:
- successful `run()` behavior on the new task;
- engine-mismatch failure paths;
- validator failures for incompatible settings;
- round-trip loading through `ConcreteTask` and `ConcreteTaskResult`;
- path rewriting tests when new result fields store filesystem paths.

Prefer targeted task tests over broad AMS-heavy suites unless the change truly alters shared loop
behavior.
