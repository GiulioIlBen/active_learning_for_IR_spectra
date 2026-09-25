from __future__ import annotations

import hashlib
import re
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from scm.moliterate.interfaces.rkf_files import RKFDataWarning


def _extract_python_blocks(text: str) -> str:
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks, "No python code blocks found in README.md"
    return "\n\n".join(blocks)


def test_readme_python_block_hash_and_exec(tuples_interface_path) -> None:
    pytest.importorskip("scm.plams")
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    assert readme_path.is_file()
    code = _extract_python_blocks(readme_path.read_text(encoding="utf-8"))

    expected_hash = "fde8510adac8d2bde0fd685afaf2f81bc61a7e19e225fc13daaf94f85d97b174"
    out_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    print("\n", code)
    print(out_hash)
    assert out_hash == expected_hash
    exec_globals = {
        "__name__": "__main__",
        "path": tuples_interface_path[2][1],
        "path2": tuples_interface_path[3][1],
        "ase": types.SimpleNamespace(Atoms=object),
        "Dict": Dict,
        "Any": Any,
        "JsonValue": object,
    }
    with pytest.warns(RKFDataWarning):
        exec(code, exec_globals)
