from __future__ import annotations

from types import SimpleNamespace

from ase import Atoms
from scm.moliterate import ConcreteInterfaces
from scm.moliterate.analysis import PairwiseDatasetMetrics, PairwiseResult
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.filters import ConditionsFilter
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.callbacks import NoPostFilterData
from scm.active_learning.checker_getters import DecoupledCheckerGetter
from scm.active_learning.checker_getters.checkers import SPChecker
from scm.active_learning.checker_getters.concat_check_getter import ConcatCheckerGetter
from scm.active_learning.checker_getters.getters.sp_getter import SPGetter
from scm.active_learning.engines import AMSEngine, Engine
from scm.active_learning.journey_scheduler import SequentialStepsJourney
from scm.active_learning.loop import ActiveLearningLoop, IterPhase, TrainerChecker
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
        return MLIPTrainerResults(engine=AMSEngine(engine_id=engine_id))

    def to_data_format(self, data_split: DataAndSplittingStrategy):
        return None


class TrainCheckRecorder:
    def __init__(self):
        self.calls = 0
        self.last_state = None

    def after_train_check(self, al_loop, state) -> None:
        self.calls += 1
        self.last_state = state


def _pairwise_result(
    *,
    success: str,
    units: str = "eV",
    metric="mae",
    value: float = 0.5,
    target: float = 1.0,
) -> PairwiseResult:
    return PairwiseResult(
        property="energy",
        units=units,
        metric=metric,
        value=value,
        target=target,
        n_entries=1,
        success=success,
    )


def _build_loop(*, trainer_checker=None, callbacks=None, post_filter=None) -> ActiveLearningLoop:
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
    journey = SequentialStepsJourney(couples=[(task, check_getter)], max_attempts=1)

    return ActiveLearningLoop.model_construct(
        start_engine=AMSEngine(engine_id="start"),
        iterable_loop=StoppableCounter(start=0, stop=1),
        journey=journey,
        labeller=sp_labeller,
        labeller_engine=AMSEngine(engine_id="labeller"),
        post_filter=post_filter,
        accuracy_checker=metrics,
        trainer_checker=trainer_checker,
        splitting=DataAndSplittingStrategy(dataset=InMemoryMolData.create(available_properties=props)),
        mlip_trainer=NoOpTrainer(),
        callbacks=callbacks or [],
        last_run=None,
    )


def _fake_accuracy_run_result(*, engine_id: str, value: float, success: str):
    return (
        SimpleNamespace(
            task=SimpleNamespace(task_id="=Model=", system_id="Len1"),
            engine=SimpleNamespace(engine_id=engine_id),
            failed_simulations={},
        ),
        [
            PairwiseResult(
                property="energy",
                metric="mae",
                units="eV",
                value=value,
                target=1.0,
                n_entries=1,
                success=success,
                group_info={},
            )
        ],
    )


def test_trainer_checker_passes_groupby_metadata_to_accuracy_run(monkeypatch):
    calls = {}

    def fake_sp_run(self, engine, **kwargs):
        return SimpleNamespace(task=self, engine=engine, dataset=self.dataset, failed_simulations={})

    def fake_compare_two_grouped(self, dataset_ref, dataset_cmp, metadata_grouping):
        calls["groupby_metadata"] = metadata_grouping
        return [_pairwise_result(success="OK")]

    monkeypatch.setattr("scm.active_learning.loop.trainer_check.SPLabellerData.run", fake_sp_run)
    monkeypatch.setattr(
        "scm.active_learning.loop.trainer_check.PairwiseDatasetMetrics.compare_two_grouped",
        fake_compare_two_grouped,
    )

    props = [PropertyInfo(name="energy", unit="eV")]
    dataset = InMemoryMolData.create(available_properties=props)
    dataset.add_system(ChemDataEntry(system=Atoms("H"), properties={"energy": 0.0}))
    TrainerChecker(
        groupby_metadata=["dataset", "system"],
        metrics=PairwiseDatasetMetrics(
            settings=[PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", target=1.0)]
        ),
    ).run(
        dataset=dataset,
        engine=AMSEngine(engine_id="candidate"),
        labeller=ConstantEnergyLabeller(properties=props),
    )

    assert calls["groupby_metadata"] == ["dataset", "system"]


def test_train_check_skips_with_train_skip_and_still_calls_callback():
    props = [PropertyInfo(name="energy", unit="eV")]
    recorder = TrainCheckRecorder()
    loop = _build_loop(
        trainer_checker=TrainerChecker(),
        callbacks=[NoPostFilterData(skip_training_too=True), recorder],
        post_filter=lambda dataset: InMemoryMolData.create(available_properties=props),
    )

    final_engine = loop.run()
    iteration = loop.result.iterations[0]

    assert final_engine.engine_id == "start"
    assert iteration.trainer_check_result is None
    assert iteration.skip[IterPhase.TRAIN] == "Post-filter dataset is empty"
    assert iteration.skip[IterPhase.TRAIN_CHECK] == "Post-filter dataset is empty"
    assert recorder.calls == 1
    assert recorder.last_state is iteration
