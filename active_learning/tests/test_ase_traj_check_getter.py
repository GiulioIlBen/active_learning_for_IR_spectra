from __future__ import annotations

from pathlib import Path

from ase import Atoms
from scm.moliterate import ChemDataSetFormat, create_dataset
from scm.moliterate.core.chem_data_entry import ChemDataEntry

from scm.active_learning.checker_getters.checkers import ASETrajChecker
from scm.active_learning.checker_getters.getters import ASETrajGetter
from scm.active_learning.engines import ASEEngine, PLAMSMolecule
from scm.active_learning.results import EngineCheckResult
from scm.active_learning.results.task.ase_md import ASEMolecularDynamicsResult
from scm.active_learning.tasks import ASEMolecularDynamics


def _make_task() -> ASEMolecularDynamics:
    from scm.plams import Atom, Molecule

    molecule = Molecule()
    molecule.add_atom(Atom(symbol="H", coords=(0.0, 0.0, 0.0)))
    molecule.add_atom(Atom(symbol="H", coords=(0.0, 0.0, 0.74)))
    return ASEMolecularDynamics(
        atomistic_system=PLAMSMolecule.from_molecules(system_id="H2", systems=molecule),
        nsteps=10,
        samplingfreq=1,
    )


def _make_engine() -> ASEEngine:
    return ASEEngine.build(engine_id="ase", calculator="ase.calculators.lj.LennardJones")


def _write_dataset(path: Path, structures: list[Atoms]) -> None:
    dataset = create_dataset(path, fmt=ChemDataSetFormat.ASE, available_properties=[])
    for atoms in structures:
        dataset.add_system(ChemDataEntry(system=atoms, properties={}, metadata={}))


def _make_result(path: Path, **overrides) -> ASEMolecularDynamicsResult:
    payload = dict(
        task=_make_task(),
        engine=_make_engine(),
        save_folder=str(path.parent.parent),
        trajectory_path=str(path),
        requested_steps=10,
        executed_steps=10,
        written_frames=0,
    )
    payload.update(overrides)
    return ASEMolecularDynamicsResult(**payload)


def test_ase_traj_checker_reports_stop_triggered_frame(tmp_path):
    path = tmp_path / "trajectory.db"
    _write_dataset(
        path,
        [
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.72]]),
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.65]]),
        ],
    )
    checker = ASETrajChecker()

    results = checker.run(
        _make_result(path, stop_triggered=True, stop_reasons=["minimum_distance<0.7"], written_frames=3)
    )

    assert len(results) == 1
    assert results[0].success == "DISTANCE"
    assert results[0].value == 1
    assert results[0].target == 2


def test_ase_traj_checker_reports_broken_bond(tmp_path):
    path = tmp_path / "trajectory.db"
    _write_dataset(
        path,
        [
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        ],
    )
    checker = ASETrajChecker()

    results = checker.run(_make_result(path, written_frames=2))

    assert len(results) == 1
    assert results[0].success == "BROKEN_BOND"
    assert results[0].value == 0
    assert results[0].target == 1


def test_ase_traj_getter_loads_db_and_applies_filters(tmp_path):
    path = tmp_path / "trajectory.db"
    _write_dataset(
        path,
        [
            Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            Atoms("H2", positions=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.84]]),
            Atoms("H2", positions=[[0.0, 0.0, 0.2], [0.0, 0.0, 0.94]]),
            Atoms("H2", positions=[[0.0, 0.0, 0.3], [0.0, 0.0, 1.04]]),
            Atoms("H2", positions=[[0.0, 0.0, 0.4], [0.0, 0.0, 1.14]]),
        ],
    )
    getter = ASETrajGetter(
        condition_filters=[
            ASETrajGetter.RemoveUnreasonableTraj(),
            ASETrajGetter.RemoveInitialTraj(skip_first_n_if_m_longer=None),
            ASETrajGetter.DataSelector(
                low_data_threshold=10,
                low_data_selector=(None, None, 1),
                default_data_selector=(None, None, 1),
            ),
        ]
    )
    checker_results = [
        EngineCheckResult(
            engine_id="ase",
            task_id="ASEMolecularDynamics",
            system_id="H2",
            checker_id="ASETrajChecker",
            property="trajectory",
            units="0-index",
            metric="reasonable_final_frame",
            value=2,
            target=4,
            n_entries=5,
            success="DISTANCE",
        )
    ]

    result = getter.run(_make_result(path, written_frames=5), checker_results)

    assert len(result.dataset) == 2
    assert list(result.dataset.absolute_idxs) == [0, 1]
