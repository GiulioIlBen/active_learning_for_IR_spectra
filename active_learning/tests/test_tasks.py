from __future__ import annotations

from typing import Iterable

import pytest
from ase import Atoms
from scm.moliterate import ChemDataSetFormat, PropertyInfo, load_dataset
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.plams import Atom, Molecule, Settings, config, from_smiles

from scm.active_learning.engines import AMSEngine, AtomisticContainer, Engine, SCMChemicalSystem
from scm.active_learning.results import SPLabellingResults
from scm.active_learning.tasks import AMSMDTask, AMSTask, SPLabeller, SPLabellerData


@pytest.fixture
def plams_erase_workdir():
    config.erase_workdir = True
    yield


class DummyAtomisticContainer(AtomisticContainer):
    def collect_atomistic_systems(self):
        return []

    def collect_molecules(self):
        mol = Molecule()
        mol.add_atom(Atom(symbol="H", coords=(0.0, 0.0, 0.0)))
        return mol


class DummyEngine(Engine):
    def run_single_point(self, properties, dataset, parallel_settings) -> Iterable[dict]:
        for idx, _ in enumerate(dataset):
            if idx == 1:
                yield {"FAILURE": "boom"}
            else:
                yield {prop.name: float(idx) for prop in properties}


def test_sp_labeller_data_run_and_system_id():
    props = [PropertyInfo(name="energy", unit="eV")]
    dataset = InMemoryMolData.create(available_properties=props)
    dataset.add_system(ChemDataEntry(system=Atoms("H"), properties={"energy": 0.0}))
    dataset.add_system(ChemDataEntry(system=Atoms("He"), properties={"energy": 1.0}))

    labeller = SPLabeller(properties=props, out_fmt=ChemDataSetFormat.IN_MEMORY)
    task = SPLabellerData(task_id="sp", sp_labeller=labeller, dataset=dataset)

    results = task.run(engine=DummyEngine(engine_id="dummy"))

    assert task.system_id == "Len2"
    assert isinstance(results, SPLabellingResults)
    assert results.failed_simulations == {1: "boom"}
    assert results.from_task == task


def test_ams_task_build_job_name_and_settings():
    container = DummyAtomisticContainer(system_id="H2O")
    task = AMSTask.model_construct(
        task_id="t1",
        source_settings={"input": {"ams": {"task": "SinglePoint"}}},
        atomistic_system=container,
    )
    engine = AMSEngine(engine_id="ML")

    job = task.build_job(engine=engine)

    assert job.name == "ML-H2O-t1"
    assert job.settings.input.ams.task == "SinglePoint"


def test_ams_task_run_rejects_wrong_engine():
    container = DummyAtomisticContainer(system_id="sys-1")
    task = AMSTask.model_construct(
        task_id="t1",
        source_settings={"input": {"ams": {"task": "SinglePoint"}}},
        atomistic_system=container,
    )

    with pytest.raises(TypeError):
        task.run(engine=Engine(engine_id="nope"))


def test_ams_task_job(plams_erase_workdir):
    pytest.importorskip("scm.base")
    pytest.importorskip("scm.plams")
    container = SCMChemicalSystem.from_molecules(system_id="sys1", systems=from_smiles("CC"))
    source_settings = Settings()
    source_settings.input.forcefield
    force_field = AMSEngine(engine_id="forcefield", source_settings=source_settings.as_dict())

    source_settings = Settings()
    source_settings.input.ams.task = "SinglePoint"
    task = AMSTask(
        task_id="SP",
        source_settings=source_settings.as_dict(),
        atomistic_system=container,
    )
    results = task.run(engine=force_field)

    assert results.plams_results.job.check()
    assert results.plams_results.get_energy() == pytest.approx(expected=0.006967547960270447, abs=1e-9)


def test_ams_md_task_job(plams_erase_workdir):
    pytest.importorskip("scm.base")
    pytest.importorskip("scm.plams")

    container = SCMChemicalSystem.from_molecules(system_id="sys1", systems=from_smiles("CC"))
    source_settings = Settings()
    source_settings.input.forcefield
    force_field = AMSEngine(engine_id="forcefield", source_settings=source_settings.as_dict())

    source_settings = Settings()
    source_settings.input.ams.task = "SinglePoint"
    task = AMSMDTask(
        task_id="SP",
        nsteps=40,
        thermostat="NHC",
        atomistic_system=container,
        samplingfreq=2,
    )
    results = task.run(engine=force_field)

    assert results.plams_results.job.check()
    assert len(load_dataset(results.plams_results.rkfpath())) == 21

    print(results.model_dump())
