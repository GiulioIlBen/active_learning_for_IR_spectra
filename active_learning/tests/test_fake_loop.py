from __future__ import annotations

from ase import Atoms
from scm.moliterate import ConcreteInterfaces
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.filters import ConditionsFilter
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.callbacks import ALStateLogger, NoPostFilterData
from scm.active_learning.checker_getters import DecoupledCheckerGetter
from scm.active_learning.checker_getters.checkers import SPChecker
from scm.active_learning.checker_getters.concat_check_getter import ConcatCheckerGetter
from scm.active_learning.checker_getters.getters.sp_getter import SPGetter
from scm.active_learning.engines import AMSEngine, Engine
from scm.active_learning.journey_scheduler import SequentialStepsJourney
from scm.active_learning.loop.loop import ActiveLearningLoop
from scm.active_learning.loop.loop_result import IterationState, LoopResult
from scm.active_learning.loop.run_control import ActiveLearningRunControl
from scm.active_learning.loop.stoppable_counter import StoppableCounter
from scm.active_learning.mlip import DataAndSplittingStrategy, MLIPTrainer
from scm.active_learning.results import MLIPTrainerResults
from scm.active_learning.tasks import SPLabeller, SPLabellerData


class ConstantEnergyLabeller(SPLabeller):
    value: float = 0.0

    def run(self, engine: Engine, dataset: ConcreteInterfaces, **kwargs) -> tuple[ConcreteInterfaces, dict[int, str]]:
        labelled = InMemoryMolData.create(available_properties=self.properties)
        for row in dataset:
            properties = {prop.name: self.value for prop in self.properties}
            labelled.add_system(ChemDataEntry(system=row.system, properties=properties, metadata=row.metadata))
        return labelled, {}


class NoOpTrainer(MLIPTrainer[AMSEngine]):
    def validate_engine(self, engine: Engine) -> AMSEngine:
        if not isinstance(engine, AMSEngine):
            raise TypeError(f"{type(engine)=} is not AMSEngine")
        return engine

    def train(self, data_split: DataAndSplittingStrategy, engine_id: str, **kwargs) -> MLIPTrainerResults:
        return MLIPTrainerResults(engine=AMSEngine(engine_id=engine_id))

    def finetune(
        self, engine: AMSEngine, data_split: DataAndSplittingStrategy, engine_id: str, **kwargs
    ) -> MLIPTrainerResults:
        return MLIPTrainerResults(engine=engine)

    def to_data_format(self, data_split: DataAndSplittingStrategy):
        return None


def test_active_learning_loop_smoke():
    props = [PropertyInfo(name="energy", unit="eV")]
    dataset = InMemoryMolData.create(available_properties=props)
    dataset.add_system(ChemDataEntry(system=Atoms("H"), properties={"energy": 0.0}))

    sp_labeller = ConstantEnergyLabeller(properties=props)
    task = SPLabellerData(task_id="sp_data", sp_labeller=sp_labeller, dataset=dataset)
    metrics = PairwiseDatasetMetrics(
        settings=[PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", target=1.0)]
    )
    checker = SPChecker(checker_id="sp_checker", metrics=metrics)
    getter = SPGetter(getter_id="sp_getter", filters_converged_sp=ConditionsFilter(conditions=["energy>=-1"]))
    check_getter = ConcatCheckerGetter(
        check_getters=[
            DecoupledCheckerGetter(checker=checker, getter=getter),
        ],
    )
    journey = SequentialStepsJourney(couples=[(task, check_getter)], max_attempts=0)

    loop = ActiveLearningLoop.model_construct(
        start_engine=AMSEngine(engine_id="start"),
        iterable_loop=StoppableCounter(start=0, stop=5),
        journey=journey,
        labeller=sp_labeller,
        labeller_engine=AMSEngine(engine_id="labeller"),
        accuracy_checker=metrics,
        splitting=DataAndSplittingStrategy(dataset=InMemoryMolData.create(available_properties=props)),
        mlip_trainer=NoOpTrainer(),
        callbacks=[NoPostFilterData(), ALStateLogger()],
        last_run=None,
    )
    print("")
    final_engine = loop.run()

    assert isinstance(final_engine, Engine)
    assert loop.result is not None


def test_loop_with_sp_task_checker():
    props = [PropertyInfo(name="energy", unit="eV")]
    dataset = InMemoryMolData.create(available_properties=props)
    dataset.add_system(ChemDataEntry(system=Atoms("H"), properties={"energy": 0.0}))

    sp_labeller = ConstantEnergyLabeller(properties=props)
    task = SPLabellerData(task_id="", sp_labeller=sp_labeller, dataset=dataset)
    metrics = PairwiseDatasetMetrics(
        settings=[PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", target=1.0)]
    )
    checker = SPChecker(checker_id="sp_checker", metrics=metrics)
    getter = SPGetter(getter_id="sp_getter", filters_converged_sp=ConditionsFilter(conditions=["energy>=-1"]))
    check_getter = ConcatCheckerGetter(
        check_getters=[
            DecoupledCheckerGetter(checker=checker, getter=getter),
        ],
    )
    journey = SequentialStepsJourney(couples=[(task, check_getter)], max_attempts=0)

    loop = ActiveLearningLoop.model_construct(
        start_engine=AMSEngine(engine_id="start"),
        iterable_loop=StoppableCounter(start=0, stop=1),
        journey=journey,
        labeller=sp_labeller,
        labeller_engine=AMSEngine(engine_id="labeller"),
        accuracy_checker=metrics,
        splitting=DataAndSplittingStrategy(dataset=InMemoryMolData.create(available_properties=props)),
        mlip_trainer=NoOpTrainer(),
        callbacks=[NoPostFilterData()],
        last_run=None,
    )
    print("")
    final_engine = loop.run()

    assert isinstance(final_engine, Engine)
    assert loop.result is not None
    assert len(loop.result.iterations) == 1
    assert loop.result.iterations[0].accuracy_checks_results is not None


def test_run_control_resume_uses_current_iteration_and_engine():
    run_control = ActiveLearningRunControl(mode="resume")
    loop = ActiveLearningLoop.model_construct(
        start_engine=AMSEngine(engine_id="start"),
        iterable_loop=StoppableCounter(start=0, stop=5),
        current_state=IterationState(iteration_al=7, start_engine=AMSEngine(engine_id="resume")),
        result=LoopResult(final_engine=AMSEngine(engine_id="final")),
    )

    assert run_control.start_iteration(loop) == 7
    assert run_control.start_engine(loop).engine_id == "resume"
    assert run_control.should_reset_shared_dataset() is False
    assert run_control.should_import_start_engine_data() is False
