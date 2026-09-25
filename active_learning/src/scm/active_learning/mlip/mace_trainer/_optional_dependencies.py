from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any


def _missing_dependency_message(*, dependency: str, install_hint: str, context: str | None = None) -> str:
    prefix = f"{context}: " if context is not None else ""
    return f"{prefix}{dependency} is required but is not installed. Install with: {install_hint}"


@lru_cache(maxsize=1)
def _load_mace_train_module() -> Any:
    try:
        return importlib.import_module("mace.cli.run_train")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="mace-torch",
                install_hint='pip install -e ".[mace]"',
                context="MACETrainer",
            )
        ) from exc


@lru_cache(maxsize=1)
def _load_mace_train_main() -> Any:
    module = _load_mace_train_module()
    run_main = getattr(module, "main", None)
    if not callable(run_main):
        raise AttributeError("mace.cli.run_train.main was not found or is not callable.")
    return run_main


@lru_cache(maxsize=1)
def _load_mace_model_script_utils_module() -> Any:
    try:
        return importlib.import_module("mace.tools.model_script_utils")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="mace-torch",
                install_hint='pip install -e ".[mace]"',
                context="MACETrainer fine-tuning",
            )
        ) from exc


@lru_cache(maxsize=1)
def _load_yaml_module() -> Any:
    try:
        return importlib.import_module("yaml")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="PyYAML",
                install_hint="pip install pyyaml",
                context="MACETrainer",
            )
        ) from exc


@lru_cache(maxsize=1)
def _load_ase_dependencies() -> Any:
    try:
        from ase import io as ase_io
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="ASE",
                install_hint="pip install ase",
                context="MACETrainer dataset export",
            )
        ) from exc
    return ase_io


@lru_cache(maxsize=1)
def _load_torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="torch",
                install_hint='pip install -e ".[mace]"',
                context="MACETrainer fine-tuning",
            )
        ) from exc


@lru_cache(maxsize=1)
def _load_mace_finetuning_utils_module() -> Any:
    try:
        return importlib.import_module("mace.tools.finetuning_utils")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="mace-torch",
                install_hint='pip install -e ".[mace]"',
                context="MACETrainer fine-tuning",
            )
        ) from exc


@lru_cache(maxsize=1)
def _load_mace_calculator_class() -> Any:
    try:
        module = importlib.import_module("mace.calculators.mace")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="mace-torch",
                install_hint='pip install -e ".[mace]"',
                context="AMSMACECalculator",
            )
        ) from exc
    return module.MACECalculator
