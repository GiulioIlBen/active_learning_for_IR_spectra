import builtins

import pytest

from scm.moliterate.utils.unit_conversion import conversion_ratio


def test_conversion_ratio_raises_when_plams_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scm.plams" or name.startswith("scm.plams"):
            raise ModuleNotFoundError("No module named 'scm.plams'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match=r"conversion_ratio: plams is required but not installed"):
        conversion_ratio("a", "b")
