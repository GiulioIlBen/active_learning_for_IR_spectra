import pytest
from ase import Atoms

from scm.moliterate.core import ChemDataEntry, PropertyInfo
from scm.moliterate.interfaces.params_data import ParAMSData


def test_importer_from_entries_params():
    pytest.importorskip("scm.params")
    pytest.importorskip("scm.plams")

    entries = [
        ChemDataEntry(
            system=Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]]),
            properties={"energy": -1.0},
            metadata={"dataset": "train"},
            idx_absolute=0,
            idx_origin="row0",
        ),
        ChemDataEntry(
            system=Atoms(numbers=[8], positions=[[0.0, 0.0, 0.0]]),
            properties={"energy": -2.0},
            metadata={"dataset": {"energy": "test"}},
            idx_absolute=1,
            idx_origin="row1",
        ),
    ]
    props = [PropertyInfo(name="energy", unit="eV")]

    ri = ParAMSData._importer_from_entries(entries, props, suffix_origin="orig_")

    assert len(ri.job_collection) == 2
    assert "train" in ri.data_sets
    assert "test" in ri.data_sets
    assert "energy('ChemEntry0000')" in ri.data_sets["train"]
    assert "energy('ChemEntry0001')" in ri.data_sets["test"]

    job0 = ri.job_collection["ChemEntry0000"]
    assert job0.metadata["orig_idx_origin"] == "row0"
    assert job0.metadata["orig_idx_absolute"] == 0
