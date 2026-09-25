from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional


def _missing_dependency_message(*, dependency: str, install_hint: str, context: str | None = None) -> str:
    prefix = f"{context}: " if context is not None else ""
    return f"{prefix}{dependency} is required but is not installed. Install with: {install_hint}"


@lru_cache(maxsize=1)
def _load_params_deps() -> tuple[Any, Any, Any]:
    try:
        from scm.params import ParAMSJob, ParAMSResults
        from scm.params.common.dataset_evaluator import GroupedResult
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="scm.params",
                install_hint='pip install -e ".[ams]"',
                context="ParAMSTrainer",
            )
        ) from exc
    return ParAMSJob, ParAMSResults, GroupedResult


@lru_cache(maxsize=1)
def _load_input_classes_drivers() -> Any:
    try:
        from scm.input_classes import drivers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="scm.input_classes",
                install_hint='pip install -e ".[ams]"',
                context="ParAMSTrainer builder",
            )
        ) from exc
    return drivers


@lru_cache(maxsize=None)
def _get_m3gnet_backend_error(model: str) -> Optional[str]:
    if model != "UniversalPotential":
        return None
    try:
        from scm.params.machine_learning.m3gnet.job import M3GNetJob
    except ModuleNotFoundError:
        return _missing_dependency_message(
            dependency="scm.params M3GNet support",
            install_hint='pip install -e ".[ams]"',
            context="ParAMSTrainer",
        )
    try:
        M3GNetJob.get_model_path()
    except FileNotFoundError as exc:
        return (
            "The ParAMS M3GNet preset uses `Model=UniversalPotential`, but the required "
            "UniversalPotential parameters are not installed in this AMS environment. "
            "Install the M3GNet UniversalPotential bundle (reported by AMS as `M3GNet-UP-2022`) "
            'or switch to `Model="Custom"`/`Model="ModelDir"` with an explicit model path. '
            f"Original error: {exc}"
        )
    return None
