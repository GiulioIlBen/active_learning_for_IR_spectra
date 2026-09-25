from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, List, Literal, Sequence

import numpy as np
from pydantic import JsonValue
from scm.moliterate import ConcreteInterfaces, PropertyInfo

from scm.active_learning.task_parallelization import ParallelStrategy

from ._shared import extract_system, resolve_dotted_object, to_ase_atoms
from .core import Engine


def _require_torchsim():
    try:
        import torch_sim as ts
    except ImportError as exc:
        raise ImportError(
            "TorchEngine requires the optional dependency `torch-sim-atomistic` (`pip install -e .[torchsim]`)."
        ) from exc
    return ts


class TorchEngine(Engine[ParallelStrategy]):
    type: Literal["TorchEngine"] = "TorchEngine"
    model_factory: str
    model_kwargs: Dict[str, JsonValue] = {}

    _PROPERTY_ALIASES: ClassVar[Dict[str, str]] = {
        "energy": "energy",
        "potential_energy": "energy",
        "free_energy": "energy",
        "forces": "forces",
    }
    _PROPERTY_NATIVE_UNITS: ClassVar[Dict[str, str]] = {
        "energy": "eV",
        "forces": "eV/Ang",
    }

    def _validate_properties(self, requested_properties: List[PropertyInfo]) -> List[str]:
        names = [prop.name for prop in requested_properties]
        unsupported = {name for name in names if name.lower() not in self._PROPERTY_ALIASES}
        if unsupported:
            raise ValueError(
                f"Not available properties: {unsupported} | Available are: {sorted(self._PROPERTY_ALIASES.keys())}"
            )
        return [self._PROPERTY_ALIASES[name.lower()] for name in names]

    def build_model(self):
        factory = resolve_dotted_object(self.model_factory, kind="model factory")
        return factory(**self.model_kwargs)

    def run_single_point(
        self,
        properties: List[PropertyInfo],
        dataset: ConcreteInterfaces,
        parallel_settings: ParallelStrategy,
        **kwargs,
    ) -> Iterable[Dict[str | Literal["FAILURE"], Any | str]]:
        del parallel_settings, kwargs

        canonical_properties = self._validate_properties(properties)
        rows = list(dataset)
        results: List[Dict[str | Literal["FAILURE"], Any | str]] = [{} for _ in rows]
        if not rows:
            return results

        model = self.build_model()
        self._prepare_model(model=model, requested_properties=canonical_properties)

        valid_indices: List[int] = []
        valid_atoms: List[Any] = []
        atom_counts: List[int] = []

        for index, row in enumerate(rows):
            try:
                atoms = to_ase_atoms(extract_system(row))
            except Exception as exc:  # noqa: BLE001
                results[index] = {"FAILURE": str(exc) or exc.__class__.__name__}
                continue
            valid_indices.append(index)
            valid_atoms.append(atoms)
            atom_counts.append(len(atoms))

        if not valid_atoms:
            return results

        ts = _require_torchsim()
        try:
            state = ts.io.atoms_to_state(
                valid_atoms,
                device=self._get_model_device(model),
                dtype=self._get_model_dtype(model),
            )
            predictions = model.forward(state)
            batched = self._split_predictions(
                predictions=predictions,
                atom_counts=atom_counts,
                n_systems=len(valid_atoms),
            )
        except Exception as exc:  # noqa: BLE001
            failure = str(exc) or exc.__class__.__name__
            for index in valid_indices:
                results[index] = {"FAILURE": failure}
            return results

        for batch_index, result_index in enumerate(valid_indices):
            row_result: Dict[str | Literal["FAILURE"], Any | str] = {}
            for prop, canonical in zip(properties, canonical_properties, strict=False):
                value = batched[canonical][batch_index]
                row_result[prop.name] = self._convert_property_value(value, prop, canonical)
            results[result_index] = row_result

        return results

    @staticmethod
    def _get_model_device(model: Any) -> Any:
        return getattr(model, "device", getattr(model, "_device", None))

    @staticmethod
    def _get_model_dtype(model: Any) -> Any:
        return getattr(model, "dtype", getattr(model, "_dtype", None))

    @staticmethod
    def _implemented_properties(model: Any) -> Sequence[str]:
        implemented = getattr(model, "implemented_properties", ())
        if implemented is None:
            return ()
        return tuple(str(prop) for prop in implemented)

    def _prepare_model(self, model: Any, requested_properties: Sequence[str]) -> None:
        implemented = set(self._implemented_properties(model))
        supports_energy = not implemented or "energy" in implemented or "free_energy" in implemented
        if not supports_energy:
            raise ValueError("TorchEngine model must implement energy or free_energy predictions.")
        if "forces" in requested_properties:
            supports_forces = "forces" in implemented or (
                not implemented and getattr(model, "compute_forces", None) is not False
            )
            if not supports_forces:
                raise ValueError("TorchEngine model does not implement forces.")
            compute_forces = getattr(model, "compute_forces", None)
            if compute_forces is False:
                model.compute_forces = True

    @classmethod
    def _split_predictions(
        cls,
        predictions: Dict[str, Any],
        atom_counts: Sequence[int],
        n_systems: int,
    ) -> Dict[str, List[Any]]:
        energy_values = cls._as_numpy(predictions.get("energy", predictions.get("free_energy")))
        if energy_values is None:
            raise ValueError("TorchEngine model output is missing energy.")
        energy_values = np.asarray(energy_values)
        if energy_values.ndim == 0:
            energy_values = energy_values.reshape(1)
        if len(energy_values) != n_systems:
            raise ValueError(f"Expected {n_systems} energy values, received shape {energy_values.shape}.")

        batched: Dict[str, List[Any]] = {"energy": [float(value) for value in energy_values]}

        if "forces" in predictions:
            forces = cls._as_numpy(predictions["forces"])
            if forces is None:
                raise ValueError("TorchEngine model output forces could not be converted.")
            force_batches = cls._split_force_batches(forces=forces, atom_counts=atom_counts, n_systems=n_systems)
            batched["forces"] = force_batches

        return batched

    @classmethod
    def _split_force_batches(cls, forces: Any, atom_counts: Sequence[int], n_systems: int) -> List[Any]:
        force_array = np.asarray(forces)
        if force_array.ndim == 3 and force_array.shape[0] == n_systems:
            return [np.asarray(force_array[index]) for index in range(n_systems)]

        if force_array.ndim != 2:
            raise ValueError(f"Unexpected forces shape: {force_array.shape}")

        batches: List[Any] = []
        start = 0
        for atom_count in atom_counts:
            stop = start + atom_count
            batches.append(np.asarray(force_array[start:stop]))
            start = stop
        if start != len(force_array):
            raise ValueError(f"Force tensor length mismatch: consumed {start} rows from shape {force_array.shape}.")
        return batches

    @staticmethod
    def _as_numpy(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return value

    def _convert_property_value(self, value: Any, prop: PropertyInfo, canonical: str) -> Any:
        native_unit = self._PROPERTY_NATIVE_UNITS.get(canonical)
        if native_unit is None:
            return value
        return value * prop.unit_conversion_from(native_unit)
