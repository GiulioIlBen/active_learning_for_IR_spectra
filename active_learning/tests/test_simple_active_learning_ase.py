from __future__ import annotations

from pathlib import Path

from ase import Atoms
from scm.moliterate import ChemDataSetFormat, create_dataset
from scm.moliterate.core.chem_data_entry import ChemDataEntry

import scm.active_learning.journey_scheduler.simple_active_learning as sal_module
from scm.active_learning import SimpleActiveLearningJourney
from scm.active_learning.checker_getters.checkers import ASETrajChecker
from scm.active_learning.checker_getters.getters import ASETrajGetter
from scm.active_learning.engines import ASEEngine, PLAMSMolecule
from scm.active_learning.tasks import ASEMolecularDynamics


def _make_task() -> ASEMolecularDynamics:
    from scm.plams import Atom, Molecule

    molecule = Molecule()
    molecule.add_atom(Atom(symbol="H", coords=(0.0, 0.0, 0.0)))
    molecule.add_atom(Atom(symbol="H", coords=(0.0, 0.0, 0.74)))
    return ASEMolecularDynamics(
        atomistic_system=PLAMSMolecule.from_molecules(system_id="H2", systems=molecule),
        nsteps=20,
        samplingfreq=2,
    )


def _write_dataset(path: Path) -> None:
    dataset = create_dataset(path, fmt=ChemDataSetFormat.ASE, available_properties=[])
    dataset.add_system(ChemDataEntry(system=Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])))
    dataset.add_system(ChemDataEntry(system=Atoms("H2", positions=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.84]])))


def test_simple_active_learning_journey_runs_ase_task_without_ams_engine(monkeypatch, tmp_path):
    db_path = tmp_path / "trajectory.db"
    _write_dataset(db_path)
    engine = ASEEngine.build(engine_id="ase", calculator="ase.calculators.lj.LennardJones")
    task = _make_task()

    def fake_run(self, engine, *args, **kwargs):
        del args, kwargs
        return sal_module.ASEMolecularDynamicsResult(
            task=self,
            engine=engine,
            save_folder=str(tmp_path),
            trajectory_path=str(db_path),
            requested_steps=self.nsteps,
            executed_steps=self.nsteps,
            written_frames=2,
        )

    monkeypatch.setattr(ASEMolecularDynamics, "run", fake_run)
    journey = SimpleActiveLearningJourney(
        checker_getter=ASETrajChecker()
        + ASETrajGetter(
            condition_filters=[
                ASETrajGetter.RemoveUnreasonableTraj(),
                ASETrajGetter.RemoveInitialTraj(skip_first_n_if_m_longer=None),
                ASETrajGetter.DataSelector(
                    low_data_threshold=10,
                    low_data_selector=(None, None, 1),
                    default_data_selector=(None, None, 1),
                ),
            ]
        ),
        task=task,
        steps=SimpleActiveLearningJourney.ListSteps.from_md_task(
            task,
            cumulative_values=[10],
            sampling=1,
            min_frames=None,
        ),
        gluer=None,
    )

    results = journey.run_current_batch(engine)

    assert len(results.validity_get_results) == 1
    assert len(results.validity_get_results[0].getter_results.dataset) == 2
    assert journey.attempt_idx_ == 1
