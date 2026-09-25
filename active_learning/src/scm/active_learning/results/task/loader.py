from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

from pydantic import TypeAdapter

if TYPE_CHECKING:
    from .core import TaskResult


def _resolve_result_type(result_type: Optional[Any]) -> Any:
    if result_type is not None:
        return result_type
    from . import ConcreteTaskResult

    return ConcreteTaskResult


def _from_results_inputs(value: Any) -> Iterable[Any]:
    if isinstance(value, Path):
        yield str(value)
    else:
        yield value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_dir():
            yield str(path / f"{path.name}.dill")


def _try_from_results(value: Any, ids: Optional[Tuple[str, str, str]], errors: List[str]) -> Optional["TaskResult"]:
    from .ams_conformers import AMSConformersResults
    from .ams_task import AMSTaskResult

    for cls in (AMSTaskResult, AMSConformersResults):
        for candidate in _from_results_inputs(value):
            try:
                return cls.from_results(candidate, ids=ids)
            except Exception as exc:
                errors.append(f"{cls.__name__}.from_results({candidate!r}) failed: {type(exc).__name__}: {exc}")
    return None


def _summary_error(value: Any, errors: List[str], final_exc: Exception) -> ValueError:
    attempts = "\n".join(f"{i}. {err}" for i, err in enumerate(errors, start=1))
    message = (
        f"Could not load task result from {value!r}. Tried {len(errors)} method(s):\n"
        f"{attempts}\n"
        f"Final validation error: {type(final_exc).__name__}: {final_exc}"
    )
    return ValueError(message)


def load_task_result(
    value: Any,
    result_type: Optional[Any] = None,
    ids: Optional[Tuple[str, str, str]] = None,
) -> "TaskResult":
    adapter = TypeAdapter(_resolve_result_type(result_type))
    from .core import TaskResult

    errors: List[str] = []

    if isinstance(value, TaskResult):
        try:
            return adapter.validate_python(value)
        except Exception as exc:
            errors.append(f"TypeAdapter.validate_python(TaskResult) failed: {type(exc).__name__}: {exc}")
            raise _summary_error(value=value, errors=errors, final_exc=exc) from exc

    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() == ".json" and path.is_file():
            try:
                return adapter.validate_json(path.read_bytes())
            except Exception as exc:
                errors.append(f"TypeAdapter.validate_json({str(path)!r}) failed: {type(exc).__name__}: {exc}")

    if isinstance(value, (str, Path)) or hasattr(value, "job"):
        loaded = _try_from_results(value, ids=ids, errors=errors)
        if loaded is not None:
            if result_type is None:
                return loaded
            try:
                return adapter.validate_python(loaded)
            except Exception as exc:
                errors.append(f"TypeAdapter.validate_python(from_results_output) failed: {type(exc).__name__}: {exc}")

    try:
        return adapter.validate_python(value)
    except Exception as exc:
        errors.append(f"TypeAdapter.validate_python(raw_input) failed: {type(exc).__name__}: {exc}")
        raise _summary_error(value=value, errors=errors, final_exc=exc) from exc
