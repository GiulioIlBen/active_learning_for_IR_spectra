from __future__ import annotations

from ase import Atoms
from scm.moliterate import ChemDataSetFormat, PropertyInfo
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.engines import Engine
from scm.active_learning.tasks import SPLabeller


class DummyEngine(Engine):
    def run_single_point(self, properties, dataset, parallel_settings, **kwargs):
        return [{"energy": float(idx)} for idx, _ in enumerate(dataset)]


def test_sp_labeller_preserves_entry_metadata():
    props = [PropertyInfo(name="energy", unit="eV")]
    dataset = InMemoryMolData.create(available_properties=props)
    dataset.add_system(ChemDataEntry(system=Atoms("H"), properties={"energy": 0.0}, metadata={"source": "a"}))
    dataset.add_system(
        ChemDataEntry(system=Atoms("He"), properties={"energy": 1.0}, metadata={"source": "b", "tag": 2})
    )

    labeller = SPLabeller(properties=props, out_fmt=ChemDataSetFormat.IN_MEMORY)

    ds, failures = labeller.run(engine=DummyEngine(engine_id="DummyEngine"), dataset=dataset)

    assert failures == {}
    rows = list(ds)
    default = {"FAILURE": False, "error": ""}
    assert rows[0].metadata == {"source": "a", **default}
    assert rows[1].metadata == {"source": "b", "tag": 2, **default}
