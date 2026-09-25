from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from scm.active_learning.engines._shared import to_ase_atoms


def require_torchsim():
    try:
        import torch_sim as ts
    except ImportError as exc:
        raise ImportError(
            "TorchSim tasks require the optional dependency `torch-sim-atomistic` (`pip install -e .[torchsim]`)."
        ) from exc
    return ts


def resolve_registry_member(registry: Any, name: str, label: str) -> Any:
    if hasattr(registry, name):
        return getattr(registry, name)

    lowered = name.lower()
    for candidate in dir(registry):
        if candidate.lower() == lowered:
            return getattr(registry, candidate)
    raise ValueError(f"Unknown TorchSim {label}: {name!r}")


def load_single_atoms(atomistic_system: Any):
    systems = atomistic_system.collect_molecules()
    system = systems
    if isinstance(systems, dict):
        if len(systems) != 1:
            raise ValueError("TorchSim tasks require exactly one structure in atomistic_system.")
        system = next(iter(systems.values()))
    try:
        return to_ase_atoms(system)
    except TypeError as exc:
        molecules_xyz = getattr(atomistic_system, "molecules_xyz", None)
        if isinstance(molecules_xyz, dict):
            if len(molecules_xyz) != 1:
                raise ValueError("TorchSim tasks require exactly one structure in atomistic_system.") from exc
            return to_ase_atoms(next(iter(molecules_xyz.values())))
        raise


def build_trajectory_path(save_folder: Path, results_subdir: str, trajectory_filename: str) -> Path:
    trajectory_path = save_folder / results_subdir / trajectory_filename
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    return trajectory_path


def build_reporter_kwargs(trajectory_path: Path, extra: Dict[str, Any]) -> Dict[str, Any]:
    reporter_kwargs = dict(extra)
    reporter_kwargs["filenames"] = [str(trajectory_path)]
    return reporter_kwargs


def state_final_energy(final_state: Any) -> float:
    for attr in ("energy", "potential_energy"):
        value = getattr(final_state, attr, None)
        if value is None:
            continue
        scalar = _extract_scalar(value)
        if scalar is not None:
            return scalar
    raise ValueError("Could not determine final energy from TorchSim state.")


def state_step_count(final_state: Any, default: int) -> int:
    for attr in ("n_steps", "nsteps", "step"):
        value = getattr(final_state, attr, None)
        if value is None:
            continue
        scalar = _extract_scalar(value)
        if scalar is not None:
            return int(scalar)
    return default


def model_supports_stress(model: Any) -> bool:
    implemented = _implemented_properties(model)
    if "stress" in implemented:
        return True
    compute_stress = getattr(model, "compute_stress", None)
    return bool(compute_stress)


def ensure_force_support(model: Any) -> None:
    implemented = _implemented_properties(model)
    supports_forces = "forces" in implemented or (
        not implemented and getattr(model, "compute_forces", None) is not False
    )
    if not supports_forces:
        raise ValueError("TorchSim optimization requires a model that implements forces.")
    if getattr(model, "compute_forces", None) is False:
        model.compute_forces = True


def _implemented_properties(model: Any) -> set[str]:
    implemented = getattr(model, "implemented_properties", ())
    if implemented is None:
        return set()
    return {str(prop) for prop in implemented}


def _extract_scalar(value: Any) -> float | None:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if hasattr(value, "reshape"):
        flat = value.reshape(-1)
        if len(flat) == 0:
            return None
        value = flat[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
