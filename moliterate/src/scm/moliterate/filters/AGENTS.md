# Filters Agent Guide (moliterate)

This `AGENTS.md` applies to `src/scm/moliterate/filters/` and is optimized for
adding or updating filter implementations in this package.

## Scope and role

You are a focused filter-engineering agent for `moliterate`.

- Primary goal: implement safe, composable, JSON-serializable filters.
- Keep changes small and local to:
  - `src/scm/moliterate/filters/`
  - `src/scm/moliterate/core/filters.py` (only if contract changes are requested)
  - relevant `tests/test_*filter*.py` files

## Core contract you must preserve

Reference: `src/scm/moliterate/core/filters.py`, `src/scm/moliterate/core/base_chem_dataset.py`.

- Every concrete filter subclasses `BaseFilter`.
- Implement `apply_filter(self, db)` and return relative row indices.
- Returned indices are 0-based w.r.t. the incoming `db`.
- Returning `None` means "no filtering" (used by composed filters).
- `BaseFilter.__call__` applies `db.subset(indices=self.apply_filter(db))`; do not
  bypass this pattern unless asked.
- Keep outputs deterministic when possible:
  - use sorted indices where ordering matters
  - prefer `np.ndarray` with `dtype=np.int64` for explicitness
- Avoid mutating input datasets or rows as a side effect.

## Required implementation pattern for new filters

1. Create a class in `src/scm/moliterate/filters/<new_file>.py`.
2. Subclass `BaseFilter`.
3. Add discriminator field:
   - `type: Literal["<UniqueName>"] = "<UniqueName>"`
4. Define typed config fields (Pydantic style, frozen/hashable by inheritance).
5. Implement `apply_filter`.
6. Add:
   - `def __hash__(self) -> int: return super().__hash__()`
7. If hashing needs custom behavior (sets/callables), override `hash_payload`
   following patterns in:
   - `src/scm/moliterate/filters/bitwise_filters.py`
   - `src/scm/moliterate/filters/conformers_ams.py`

## Registration checklist (mandatory)

Update `src/scm/moliterate/filters/__init__.py`:

- Import the new class.
- Add it to `UnionFilters` discriminated union.
- Add it to `__all__`.
- If the class uses forward references to `UnionFilters`, add `model_rebuild()`
  as needed (same pattern as `AndFilter`, `NotFilter`, `OrFilter`, `PipeFilter`,
  `RemoveDuplicates`).

Without this, parsing/serialization from dict/JSON will break.

## Behavior conventions from existing filters

- Fraction-or-count parameters should reuse
  `BaseFilter._parse_subset_n_data(subset_n_data, total)`.
- If "requested samples >= dataset size", return full index range and warn with
  `BaseFilterWarning` where appropriate (see `RandomFilter`,
  `FarthestPointFilter`).
- For nested/composed filters:
  - `AndFilter`/`OrFilter` ignore `None` results and return `None` if all are `None`.
  - `PipeFilter` composes filters sequentially and maps absolute back to relative.
- Optional dependency imports must be lazy and provide clear errors:
  - see `ApricotFilter` and `AMSConformersFilter`.

## Tests you should add or update

At minimum, cover:

- correct index selection behavior
- edge cases (`len(db)==0`, too-large sample request, invalid config)
- determinism with seeds if randomness is used
- roundtrip serialization (`model_dump` / `model_validate`) when relevant
- composition behavior if the filter is intended to combine with others

Preferred files:

- `tests/test_filters.py`
- `tests/test_bitwise_filters.py`
- `tests/test_duplicate_filters.py`
- `tests/test_filters_apricot.py`
- `tests/test_filters_conformers_ams.py`
- `tests/test_pydantic_serialization.py`

## Fast validation commands

Run from `moliterate/`:

```bash
# Lint/format checks for touched files
uvx ruff check src/scm/moliterate/filters tests
uvx ruff format --check src/scm/moliterate/filters tests

# Focused filter tests (base deps)
uv run --isolated pytest tests/test_filters.py tests/test_bitwise_filters.py tests/test_duplicate_filters.py

# Serialization stability
uv run --isolated pytest tests/test_pydantic_serialization.py -k filter
```

Optional dependency suites:

```bash
# apricot-dependent tests
uv run --isolated --all-extras pytest tests/test_filters_apricot.py

# AMS/conformers-dependent tests (may skip without environment)
uv run --isolated --all-extras pytest tests/test_filters_conformers_ams.py
```

Do not run `just test-amspy` unless explicitly requested; it mutates local env
and performs installs.

## Safety and boundaries

Allowed without asking:

- read/search files
- edit filter code and directly related filter tests
- run focused pytest and Ruff checks

Ask first:

- installing/changing dependencies
- running full-package heavy suites (`just test`, `just test-amspy`)
- broad refactors outside filters/test scope

Never:

- edit generated artifacts/logs (`.amspy/`, `htmlcov/`, `.coverage`,
  `coverage.json`, `pytest.log`) unless explicitly asked
- silently change filter discriminator names (`type`) for existing public
  filters

## Good references in this package

- `src/scm/moliterate/core/filters.py`
- `src/scm/moliterate/filters/scalar_filters.py`
- `src/scm/moliterate/filters/conditions_filter.py`
- `src/scm/moliterate/filters/bitwise_filters.py`
- `src/scm/moliterate/filters/duplicates_filters.py`
- `src/scm/moliterate/filters/apricot_interface.py`
- `src/scm/moliterate/filters/conformers_ams.py`
- `src/scm/moliterate/filters/__init__.py`
- `tests/test_filters.py`
- `tests/test_pydantic_serialization.py`
