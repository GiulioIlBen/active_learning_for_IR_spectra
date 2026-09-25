from __future__ import annotations

import io

from ase import Atoms
from ase.io import write


def _import_py3dmol():
    try:
        import py3Dmol
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "py3Dmol is not installed. Install the visualization extras with `pip install 'moliterate[viz]'`."
        ) from exc
    return py3Dmol


def atoms_to_xyz(atoms: Atoms) -> str:
    buffer = io.StringIO()
    write(buffer, atoms, format="xyz")
    return buffer.getvalue()


def render_structure_html(atoms: Atoms, width: int = 400, height: int = 480) -> str:
    py3Dmol = _import_py3dmol()
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(atoms_to_xyz(atoms), "xyz")
    viewer.setStyle({"stick": {"radius": 0.18}, "sphere": {"scale": 0.32}})
    if atoms.pbc.any():
        viewer.addUnitCell()
    viewer.zoomTo()
    viewer.setBackgroundColor("white")
    return viewer.write_html()
