"""Run three complete active-learning iterations with ASE MD and Lennard-Jones reference labels.

This is the smallest ``SimpleActiveLearningJourney`` example that needs no
AMS executable. ASE runs periodic Ar molecular dynamics; ASE Lennard-Jones
calculates the reference energy and forces; and MACE trains a small CPU model
from those labels.

Install the PyPI dependencies from the project root, then run:

    uv sync --extra mace
    uv run --extra mace tutorials/scripts/sal0_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from ase import Atoms
from ase.build import bulk
from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.filters import ConditionsFilter
from scm.plams import Atom, Molecule

from scm.active_learning import ActiveLearningLoop, SimpleActiveLearningJourney
from scm.active_learning.checker_getters.checkers import ASETrajChecker
from scm.active_learning.checker_getters.getters import ASETrajGetter
from scm.active_learning.engines import ASEEngine, PLAMSMolecule
from scm.active_learning.loop import StoppableCounter
from scm.active_learning.mlip import DataAndSplittingStrategy, MACETrainer
from scm.active_learning.tasks import ASEMolecularDynamics, SPLabeller


def build_argon_box() -> tuple[PLAMSMolecule, list[list[float]]]:
    """Build a periodic fcc Ar box for the ASE task and return its cell separately for ASE MD."""
    atoms: Atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat((2, 2, 2))
    molecule = Molecule()
    for symbol, position in zip(atoms.symbols, atoms.positions, strict=True):
        molecule.add_atom(Atom(symbol=symbol, coords=position))
    return PLAMSMolecule.from_molecules(system_id="Ar32", systems=molecule), atoms.cell.tolist()


def build_lennard_jones_engine(engine_id: str) -> ASEEngine:
    """Create the ASE calculator engine used for either MD exploration or reference single-point labels."""
    return ASEEngine.build(engine_id=engine_id, calculator="ase.calculators.lj.LennardJones")


def build_md_task(atomistic_system: PLAMSMolecule, cell: list[list[float]], output_root: Path) -> ASEMolecularDynamics:
    """Configure short periodic Langevin dynamics that proposes Ar configurations to the active-learning journey."""
    return ASEMolecularDynamics(
        atomistic_system=atomistic_system,
        save_folder=str(output_root / "ase_md"),
        nsteps=40,
        samplingfreq=10,
        timestep=1.0,
        temperature=100.0,
        thermostat="Langevin",
        friction=0.02,
        fixcm=False,
        use_cell=True,
        cell=cell,
        pbc=[True, True, True],
        exit_condition_freq=None,
        minimum_distance=None,
        error_threshold=None,
    )


def build_journey(task: ASEMolecularDynamics) -> SimpleActiveLearningJourney:
    """Create three SAL steps that check and select frames written by the ASE molecular-dynamics task."""
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
    return SimpleActiveLearningJourney(
        checker_getter=ASETrajChecker(check_bond_breaking=False) + getter,
        task=task,
        steps=SimpleActiveLearningJourney.ListSteps.from_md_task(
            task,
            cumulative_values=[40, 80, 120],
            sampling=10,
            min_frames=None,
        ),
        max_attempts=1,
        first_step_train=True,
        finish_loop_on_max_attempts=True,
        gluer=None,
    )


def build_smoke_trainer(output_root: Path) -> MACETrainer:
    """Configure a tiny CPU MACE fit so the smoke test completes one genuine active-learning training phase."""
    trainer = MACETrainer()
    trainer.data.folder = output_root / "mace_training"
    trainer.train_settings.swa = False
    trainer.train_settings.max_num_epochs = 1
    trainer.train_settings.batch_size = 4
    trainer.train_settings.valid_batch_size = 4
    trainer.train_settings.patience = 1
    trainer.train_settings.learning_rate = 0.01
    trainer.train_settings.device = "cpu"
    trainer.train_settings.architecture.hidden_irreps = "16x0e + 16x1o"
    trainer.train_settings.architecture.max_ell = 1
    trainer.train_settings.architecture.correlation = 2
    trainer.train_settings.architecture.num_interactions = 1
    trainer.train_settings.architecture.MLP_irreps = "8x0e"
    return trainer


def build_active_learning_loop(output_root: Path) -> ActiveLearningLoop:
    """Compose ASE exploration, ASE LJ labelling, and MACE training into one complete simple active-learning loop."""
    atomistic_system, cell = build_argon_box()
    task = build_md_task(atomistic_system, cell, output_root)
    return ActiveLearningLoop(
        tag="sal0_smoke_test",
        start_engine=build_lennard_jones_engine("LJ_exploration"),
        journey=build_journey(task),
        iterable_loop=StoppableCounter(stop=3),
        labeller=SPLabeller(
            properties=[
                PropertyInfo(name="energy", unit="eV"),
                PropertyInfo(name="forces", unit="eV/Ang"),
            ],
            out_path=str(output_root / "lj_labels.db"),
            if_out_path_exists="rename",
        ),
        labeller_engine=build_lennard_jones_engine("LJ_reference"),
        post_filter=ConditionsFilter(conditions="@fmax<18"),
        accuracy_checker=PairwiseDatasetMetrics(
            settings=[
                PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", per_n_atoms=True, target=0.01),
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", target=0.01),
            ]
        ),
        splitting=DataAndSplittingStrategy(splitting_policy="policy_off", splitter="sequential"),
        mlip_trainer=build_smoke_trainer(output_root),
        trainer_checker=None,
    )


def main(output_root: Path = Path("sal0_smoke_test")) -> None:
    """Run three ASE-only SAL iterations and print the trained engine identifier and final loop message."""
    ActiveLearningLoop.logging_config(clean_sinks="YES", new_sink=sys.stderr, level="INFO")
    active_learning_loop = build_active_learning_loop(output_root)
    trained_engine = active_learning_loop.run()
    print(f"Trained engine: {trained_engine.engine_id}")
    print(active_learning_loop.result.exit_message)


if __name__ == "__main__":
    main()
