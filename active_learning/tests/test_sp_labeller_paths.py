from __future__ import annotations

from pathlib import Path

import pytest
from scm.moliterate import ChemDataSetFormat, PropertyInfo

from scm.active_learning.tasks.sp_labeller import SPLabeller


def _labeller(tmp_path, name, if_exists="rename"):
    return SPLabeller(
        properties=[PropertyInfo(name="energy")],
        out_fmt=ChemDataSetFormat.IN_MEMORY,
        out_path=str(tmp_path / name),
        if_out_path_exists=if_exists,
    )


def test_sp_labeller_out_fmt_mapping_and_serialization():
    labeller = SPLabeller(properties=[PropertyInfo(name="energy")], out_fmt="ASE")

    assert labeller.out_fmt == ChemDataSetFormat.ASE
    assert labeller.model_dump()["out_fmt"] == "ASE"


def test_sp_labeller_get_valid_path_when_missing(tmp_path):
    labeller = _labeller(tmp_path, "missing.db")

    assert labeller._get_valid_path() == str(tmp_path / "missing.db")


def test_sp_labeller_get_valid_path_raise_when_exists(tmp_path):
    path = tmp_path / "exists.db"
    path.write_text("data")

    labeller = _labeller(tmp_path, "exists.db", if_exists="raise")

    with pytest.raises(FileExistsError):
        labeller._get_valid_path()


def test_sp_labeller_get_valid_path_overwrite(tmp_path):
    path = tmp_path / "overwrite.db"
    path.write_text("data")

    labeller = _labeller(tmp_path, "overwrite.db", if_exists="overwrite")

    assert labeller._get_valid_path() == str(path)
    assert not path.exists()


def test_sp_labeller_get_valid_path_rename(tmp_path):
    path = tmp_path / "labelling_000.db"
    path.write_text("data")

    labeller = _labeller(tmp_path, "labelling_000.db", if_exists="rename")

    assert labeller._get_valid_path().endswith("labelling_001.db")


def test_sp_labeller_get_valid_path_rename_skips_existing_candidates(tmp_path):
    for name in ["labelling_000.db", "labelling_001.db"]:
        (tmp_path / name).write_text("data")

    labeller = _labeller(tmp_path, "labelling_000.db", if_exists="rename")

    assert labeller._get_valid_path().endswith("labelling_002.db")


def test_sp_labeller_special_rename_handles_digits():
    assert SPLabeller._special_rename(Path("out.db")) == Path("out_000.db")
    assert SPLabeller._special_rename(Path("out_009.db")) == Path("out_010.db")
