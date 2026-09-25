import sys
from typing import Tuple

import scm.plams as plams
from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.filters import ConditionsFilter, LinearSteppedFilter

from scm.active_learning import ActiveLearningLoop, SimpleActiveLearningJourney
from scm.active_learning.callbacks import FolderManagerCallback, ImportStartEngineData
from scm.active_learning.checker_getters.checkers import AMSTrajChecker, NoneChecker
from scm.active_learning.checker_getters.getters import AMSTrajGetter
from scm.active_learning.engines import AMSEngine, SCMChemicalSystem
from scm.active_learning.loop import StoppableCounter
from scm.active_learning.mlip import DataAndSplittingStrategy
from scm.active_learning.mlip.params_trainer import ParAMSTrainer
from scm.active_learning.task_parallelization import AMSParallelCPU, AMSSerialStrategy
from scm.active_learning.tasks import AMSMDTask, SPLabeller


def get_molecule() -> plams.Molecule:
    mol = plams.from_smiles("OCC=O", forcefield="uff")
    for at in mol:
        at.properties = {}
    return mol


def get_engines() -> Tuple[AMSEngine, AMSEngine]:
    ref_s = plams.Settings()
    ref_s.input.ForceField.Type = "UFF"
    ref_s.runscript.nproc = 1

    start_engine = AMSEngine(engine_id="UFFStart", source_settings=ref_s.as_dict())
    labeller_engine = AMSEngine(engine_id="UFFRef", source_settings=ref_s.as_dict())

    return start_engine, labeller_engine


def get_task(systems: plams.Molecule) -> AMSMDTask:
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
    )
    return md_task


def get_trainer() -> ParAMSTrainer:
    # batch_size
    mlip_trainer = (
        ParAMSTrainer.Builder.M3GNet_EF(
            Model="UniversalPotential",
            learning_rate=None,
        )
        # .set_committee(committee_size=1)
        .set_max_epochs(200)
        # .set_loss(forces=10, energy=1)
        # .set_target_forces(0.05)
        # .set_batch_size(2)
        .build()
    )
    return mlip_trainer


def get_sal(md_task: AMSMDTask) -> SimpleActiveLearningJourney:
    step_model = SimpleActiveLearningJourney.GeometricSteps.from_md_settings(
        md_settings=md_task.settings,
        start=10,
        num_steps=10,
        # Optional override; omit to use md_task.settings SamplingFreq
        sampling=10,
        min_frames=100,
    )
    checker = AMSTrajChecker()
    getter = AMSTrajGetter()
    final_selector = getter.condition_filters[2]
    assert isinstance(final_selector, AMSTrajGetter.DataSelector), type(final_selector)
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
        # gluer=None,
        task=md_task,
        steps=step_model,
        max_attempts=30,
        first_step_train=False,
        task_parallelization=AMSSerialStrategy(watch_ams_log_stdout=False, nproc=1, OMP_NUM_THREADS=1),
        finish_loop_on_max_attempts=True,
    )

    return sal


def get_al(
    sal: SimpleActiveLearningJourney,
    labeller_engine: AMSEngine,
    mlip_trainer: ParAMSTrainer,
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
            ],
        ),
        labeller_engine=labeller_engine,
        post_filter=ConditionsFilter(conditions="@fmax<18"),
        accuracy_checker=PairwiseDatasetMetrics(
            settings=[
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="max_scaled_abs", target=0.5),
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="max_abs", target=0.6),
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", target=0.1),
                PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", per_n_atoms=True, target=0.1),
            ]
        ),
        splitting=DataAndSplittingStrategy(
            splitting_policy="policy_off",  # "success_in_val", "fail_in_train", "splitter_off", "policy_off"
            splitter="sequential",  # "sequential", "random"
        ),
        mlip_trainer=mlip_trainer,
    )
    # al.callbacks.append(
    #     EarlyStopAccuracy(patience=3, check_results_query="property='forces',metric='max_scaled_abs'")
    # )
    al.query_callback(FolderManagerCallback).logging_file_level = "DEBUG"
    al.query_callback(FolderManagerCallback).cleanup.mode = "slim"
    assert len(al.query_callbacks(ImportStartEngineData)) == 1, "The AL should collect data from previous engine"
    return al


def get_initial(al: ActiveLearningLoop, start_engine: AMSEngine) -> ActiveLearningLoop:
    # Initial run: needed to generate initial samples
    al_start = al.model_copy(deep=True)
    al_start.iterable_loop.stop = 1
    al_start.start_engine = start_engine
    # al_start.splitting.splitter = "sequential"
    # al_start.splitting.splitting_policy = "policy_off"
    assert isinstance(al_start.journey, SimpleActiveLearningJourney)
    assert isinstance(al_start.journey.task, AMSMDTask)
    al_start.journey.task.temperature = 500.0
    al_start.journey.max_attempts = 1
    al_start.journey.checker_getter.checker = NoneChecker()
    final_selector: AMSTrajGetter.DataSelector = al_start.journey.checker_getter.getter.condition_filters[2]
    # Since in this step we will ends up with 10 frames and the low_data_threshold is 100,
    # we modify just the low_data_selector. This one will be ultimetly used.
    final_selector.low_data_selector = LinearSteppedFilter(
        step=-2,
        num_samples=4,
    )
    al_start.journey.steps = SimpleActiveLearningJourney.ListSteps(
        sampling=100,
        cumulative_values=[1000],
        min_frames=10,
    )
    # If the accuracy checks are good or skipped than the training is skipped,
    # for this specail case we have to make sure the train is lounched
    al_start.journey.first_step_train = True
    return al_start


def main(check: bool = False) -> None:
    print("Make sure you have installed m3gnet: `amspackages install m3gnet`.")

    # Journey
    molecule = get_molecule()
    md_task = get_task(molecule)
    sal = get_sal(md_task)

    # AL Loop
    start_engine, labeller_engine = get_engines()
    mlip_trainer = get_trainer()
    al = get_al(sal, labeller_engine, mlip_trainer)
    al_start = get_initial(al, start_engine)
    al_start.tag = "sal1_m3gnet-start"
    al.tag = "sal1_m3gnet"
    if check:
        print(al.settings)
        print("===============================")
        print(al_start.journey.journey_table())
        print("")
        print(al.journey.journey_table())
        return

    # Logging
    # from scm.active_learning.callbacks import ALStateLogger
    # al_start.pop_callbacks(ALStateLogger)
    # al.pop_callbacks(ALStateLogger)
    print([x.type for x in al_start.callbacks])
    ActiveLearningLoop.logging_config(
        clean_sinks="YES",
        new_sink=sys.stderr,
        level="INFO",
    )

    # Run
    initial_engine = al_start.run()
    initial_engine.engine_id = "M3GNetFirst"
    al.start_engine = initial_engine
    al.run()

    # Summary
    print(al.analysis.get_summary())
    al.analysis.plot.save_to_pdf()


if __name__ == "__main__":
    main()
