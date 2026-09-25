from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from scm.active_learning.mlip._torch_compat import ensure_torch_dataloader_t_co_alias


def _missing_dependency_message(*, dependency: str, install_hint: str, context: str | None = None) -> str:
    prefix = f"{context}: " if context is not None else ""
    return f"{prefix}{dependency} is required but is not installed. Install with: {install_hint}"


@lru_cache(maxsize=1)
def _load_schnetpack_training_deps() -> tuple[Any, Any, Any]:
    try:
        ensure_torch_dataloader_t_co_alias()
        from hydra import compose, initialize_config_dir
        from schnetpack.cli import train

        importlib.import_module("schnetpack")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            _missing_dependency_message(
                dependency="SchNetPack training stack",
                install_hint='pip install -e ".[spk]"',
                context="SchNetPackTrainer",
            )
        ) from exc
    return compose, initialize_config_dir, train


@lru_cache(maxsize=1)
def _check_installation() -> str:
    try:
        _load_schnetpack_training_deps()
    except Exception as exc:
        return str(exc)
    return ""
