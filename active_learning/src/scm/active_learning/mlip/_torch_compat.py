"""Compatibility helpers for PyTorch-backed optional trainer dependencies."""

from __future__ import annotations


def ensure_torch_dataloader_t_co_alias() -> None:
    """Restore ``torch.utils.data.dataloader.T_co`` for older dependencies."""
    try:
        import torch.utils.data.dataloader as torch_dataloader
    except ModuleNotFoundError:
        return

    if hasattr(torch_dataloader, "T_co") or not hasattr(torch_dataloader, "_T_co"):
        return

    torch_dataloader.T_co = torch_dataloader._T_co


def ensure_torch_safe_globals_for_e3nn() -> None:
    """Allow older e3nn packaged constants to load under PyTorch 2.6+ defaults."""
    try:
        import torch.serialization
    except ModuleNotFoundError:
        return

    torch.serialization.add_safe_globals([slice])
