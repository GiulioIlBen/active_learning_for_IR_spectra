from __future__ import annotations

import warnings
from importlib import import_module
from io import StringIO
from typing import Any


def resolve_dotted_object(path: str, kind: str = "object") -> Any:
    if "." not in path:
        raise ValueError(f"{kind} must be a fully qualified path like 'package.module.Symbol'")
    module_name, object_name = path.rsplit(".", 1)
    module = import_module(module_name)
    resolved = getattr(module, object_name, None)
    if resolved is None:
        raise ValueError(f"Cannot find {kind} '{object_name}' in module '{module_name}'")
    return resolved


def apply_ams_external_capabilities(calculator: Any, implemented_properties: Any) -> None:
    try:
        from scm.amspipe import AMSExternalCapabilities
    except ImportError:
        warnings.warn(
            "AMSExternalCapabilities are unavailable because optional dependency `scm.amspipe` could not be "
            "imported. Continuing without AMS external capabilities; the calculator remains usable as an ASE "
            "calculator.",
            stacklevel=2,
        )
        calculator.ams_capabilities = None
        return

    calculator.ams_capabilities = AMSExternalCapabilities()
    calculator.ams_capabilities.apply_implemented_properties(implemented_properties)


def extract_system(row: Any) -> Any:
    """Prefer stored ASE atoms so ASE engines avoid optional ChemicalSystem conversion while processing dataset rows."""
    atoms = getattr(row, "atoms", None)
    if atoms is not None:
        return atoms
    if hasattr(row, "chemical_system"):
        return row.chemical_system
    return row


def to_ase_atoms(system: Any):
    try:
        from ase import Atoms
        from ase.io import read as ase_read
    except ImportError as exc:
        raise ImportError("ASE-backed engines require the optional dependency 'ase' (pip install ase).") from exc

    if isinstance(system, Atoms):
        return system.copy()

    if isinstance(system, str):
        try:
            return ase_read(StringIO(system), format="xyz")
        except Exception:
            pass

    for attr in ("to_ase", "as_ase"):
        fn = getattr(system, attr, None)
        if callable(fn):
            converted = fn()
            if isinstance(converted, Atoms):
                return converted

    atoms_attr = getattr(system, "ase_atoms", None)
    if isinstance(atoms_attr, Atoms):
        return atoms_attr.copy()

    try:
        atom_collection = getattr(system, "atoms", system)
        symbols = [atom.symbol for atom in atom_collection]
        positions = [tuple(atom.coords) for atom in atom_collection]
        if symbols:
            return Atoms(symbols=symbols, positions=positions)
    except Exception:
        pass

    raise TypeError(f"Unsupported atom container for ASE conversion: {type(system)}")
