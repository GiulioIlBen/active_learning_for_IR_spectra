import sys

import scm.plams as plams
from sal1_m3gnet import (
    get_al,
    get_engines,
    get_initial,
    get_molecule,
)
from scm.moliterate.filters import LinearSteppedFilter

from scm.active_learning import ActiveLearningLoop, SimpleActiveLearningJourney
from scm.active_learning.checker_getters.checkers import ASETrajChecker
from scm.active_learning.checker_getters.getters import ASETrajGetter
from scm.active_learning.engines import SCMChemicalSystem
from scm.active_learning.mlip import MACETrainer
from scm.active_learning.tasks import ASEMolecularDynamics


def get_task(systems: plams.Molecule) -> ASEMolecularDynamics:
    md_task = ASEMolecularDynamics(
        atomistic_system=SCMChemicalSystem.from_molecules(
            system_id="OCC=O",
            systems=systems,
        ),
        save_folder="ase_md_loop",
        nsteps=10000,
        timestep=0.5,
        temperature=300.0,
        thermostat="Langevin",
        exit_condition_freq=1,
        minimum_distance=0.6,
    )
    print(md_task.model_dump_json(indent=2))
    return md_task


def get_trainer() -> MACETrainer:
    mlip_trainer = MACETrainer()
    mlip_trainer.train_settings.architecture.model = "MACE"
    mlip_trainer.train_settings.max_num_epochs = 200
    mlip_trainer.train_settings.batch_size = 5
    mlip_trainer.train_settings.patience = 20
    mlip_trainer.train_settings.learning_rate = 0.001
    print(mlip_trainer)
    return mlip_trainer


def get_sal(md_task: ASEMolecularDynamics) -> SimpleActiveLearningJourney:
    step_model = SimpleActiveLearningJourney.GeometricSteps.from_md_task(
        md_task=md_task,
        start=10,
        num_steps=10,
        sampling=10,
        min_frames=100,
    )
    checker = ASETrajChecker()
    getter = ASETrajGetter()
    final_selector: ASETrajGetter.DataSelector = getter.condition_filters[2]
    final_selector.low_data_threshold = 100
    final_selector.low_data_selector = LinearSteppedFilter(
        step=-4,
        num_samples=2,
    )
    final_selector.default_data_selector = LinearSteppedFilter(
        step=-4,
        num_samples=3,
    )
    sal = SimpleActiveLearningJourney(
        checker_getter=checker + getter,
        task=md_task,
        steps=step_model,
        max_attempts=15,
        first_step_train=False,
        finish_loop_on_max_attempts=True,
        gluer=None,
    )

    print(sal.journey_table())
    return sal


def main(check: bool = False) -> None:
    print("Make sure you have installed MACE and ASE optional dependencies.")
    ActiveLearningLoop.logging_config(
        clean_sinks="YES",
        new_sink=sys.stderr,
        level="INFO",
    )
    molecule = get_molecule()
    md_task = get_task(molecule)
    sal = get_sal(md_task)

    start_engine, labeller_engine = get_engines()
    mlip_trainer = get_trainer()
    al = get_al(sal, labeller_engine, mlip_trainer)
    al_start = get_initial(al, start_engine)
    al.tag = "sal4_mace_ase"
    al_start.tag = "sal4_mace_ase-start"

    if check:
        print(al_start.journey.journey_table())
        print("")
        print(al.journey.journey_table())
        return

    initial_engine = al_start.run()
    initial_engine.engine_id = "MACEFirst"
    al.start_engine = initial_engine
    al.run()


if __name__ == "__main__":
    main()
