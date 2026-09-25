import sys
from typing import Tuple

import scm.plams as plams
from sal1_m3gnet import (
    get_initial,
    get_molecule,
    get_sal,
)
from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.filters import ConditionsFilter

from scm.active_learning import ActiveLearningLoop, SimpleActiveLearningJourney
from scm.active_learning.callbacks import FolderManagerCallback, ImportStartEngineData
from scm.active_learning.engines import AMSEngine, SCMChemicalSystem
from scm.active_learning.loop import StoppableCounter
from scm.active_learning.mlip import ConcreteMLIPTrainer, DataAndSplittingStrategy, MACETrainer
from scm.active_learning.task_parallelization import AMSParallelCPU
from scm.active_learning.tasks import AMSMDTask, SPLabeller


def get_engines() -> Tuple[AMSEngine, AMSEngine]:
    ref_s = plams.Settings()
    ref_s.input.ForceField.Type = "UFF"
    # UFF needs guessed charges to produce a non-zero dipole moment for polar systems.
    ref_s.input.ForceField.GuessCharges = True
    ref_s.runscript.nproc = 1

    start_engine = AMSEngine(engine_id="UFFStart", source_settings=ref_s.as_dict())
    labeller_engine = AMSEngine(engine_id="UFFRef", source_settings=ref_s.as_dict())

    print(plams.AMSJob(settings=start_engine.settings).get_input())

    return start_engine, labeller_engine


def get_task(systems: plams.Molecule, dipole=True) -> AMSMDTask:
    md_task = AMSMDTask(
        atomistic_system=SCMChemicalSystem.from_molecules(
            system_id="OCC=O",
            systems=systems,
        ),
        nsteps=10000,
        timestep=0.5,
        temperature=300.0,
        thermostat="NHC",
        exit_condition_freq=1,
        minimum_distance=0.6,
        writeenginegradients=True,
        binlog_dipolemoment=dipole,
        binlog_time=dipole,
    )
    print(plams.AMSJob(settings=md_task.ams_task.settings).get_input())
    return md_task


def get_trainer() -> MACETrainer:
    mlip_trainer = MACETrainer()
    mlip_trainer.train_settings.device = "cpu"
    mlip_trainer.train_settings.architecture.loss = "energy_forces_dipole"
    mlip_trainer.train_settings.architecture.error_table = "EnergyDipoleRMSE"
    mlip_trainer.train_settings.architecture.model = "EnergyDipolesMACE"
    mlip_trainer.train_settings.model_name = "dipole_mace"
    mlip_trainer.train_settings.max_num_epochs = 400
    mlip_trainer.train_settings.batch_size = 5
    mlip_trainer.train_settings.patience = 20
    mlip_trainer.train_settings.start_swa = 100
    mlip_trainer.train_settings.learning_rate = 0.001
    mlip_trainer.train_settings.architecture.num_channels = 64  # 128 #"number of embedding channels"
    mlip_trainer.train_settings.architecture.correlation = 3
    mlip_trainer.train_settings.architecture.max_L = 1
    # mlip_trainer.train_settings.architecture.r_max = 4
    mlip_trainer.train_settings.architecture.max_ell = 3
    print(mlip_trainer)
    return mlip_trainer


def get_al(
    sal: SimpleActiveLearningJourney,
    labeller_engine: AMSEngine,
    mlip_trainer: ConcreteMLIPTrainer,
) -> ActiveLearningLoop:
    al = ActiveLearningLoop(
        # start_engine=start_engine,
        journey=sal,
        iterable_loop=StoppableCounter(stop=sal.max_n_steps),
        labeller=SPLabeller(
            parallelization=AMSParallelCPU(maxjobs=5, nproc=1),
            properties=[
                PropertyInfo(name="energy", unit="eV"),
                PropertyInfo(name="forces", unit="eV/Ang"),
                PropertyInfo(name="dipole", unit="e*Ang"),
            ],
        ),
        labeller_engine=labeller_engine,
        post_filter=ConditionsFilter(conditions="@fmax<18"),
        accuracy_checker=PairwiseDatasetMetrics(
            settings=[
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="max_scaled_abs", target=0.1),
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="max_abs", target=0.6),
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", target=0.01),
                PairwiseDatasetMetrics.PropMetricEv(property="dipole", metric="mae", target=0.01),
                PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", per_n_atoms=True, target=0.01),
            ]
        ),
        splitting=DataAndSplittingStrategy(
            splitting_policy="policy_off",  # "success_in_val", "fail_in_train", "splitter_off", "policy_off"
            splitter="sequential",  # "sequential", "random"
        ),
        mlip_trainer=mlip_trainer,
    )
    al.query_callback(FolderManagerCallback).logging_file_level = "DEBUG"
    al.query_callback(FolderManagerCallback).cleanup.mode = "off"
    assert len(al.query_callbacks(ImportStartEngineData)) == 1, "The AL should collect data from previous engine"
    print(al.settings)
    return al


def main(check: bool = False) -> None:
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
    al.tag = "sal3_maceEFD"
    al_start.tag = "sal3_maceEFD-start"
    if check:
        print(al_start.journey.journey_table())
        print("")
        print(al.journey.journey_table())
        return

    initial_engine = al_start.run()
    initial_engine.engine_id = "MACEFirst"
    al.start_engine = initial_engine
    al.run()

    print(al.analysis.get_summary())
    al.analysis.plot.save_to_pdf()


if __name__ == "__main__":
    main()
