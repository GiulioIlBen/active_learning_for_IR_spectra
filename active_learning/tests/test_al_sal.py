from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.filters import BaseFilterWarning
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.interfaces.rkf_files import RKFDataWarning
from scm.moliterate.utils.progress_bar import MOLITERATE_PROGRESS_BAR_CONFIG
from scm.plams import Atom, Molecule, Settings

from scm.active_learning import ActiveLearningLoop, SimpleActiveLearningJourney
from scm.active_learning.checker_getters.checkers import AMSTrajChecker
from scm.active_learning.checker_getters.getters import AMSTrajGetter
from scm.active_learning.engines import AMSEngine, SCMChemicalSystem
from scm.active_learning.loop import StoppableCounter
from scm.active_learning.mlip import DataAndSplittingStrategy, MLIPTrainer
from scm.active_learning.results import MLIPTrainerResults
from scm.active_learning.results.task import AMSTaskResult
from scm.active_learning.tasks import AMSMDTask, SPLabeller

pytest.importorskip("scm.base")
pytest.importorskip("scm.plams")
pytest.importorskip("scm.params")


def _co_molecule() -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom(symbol="C", coords=(0.0, 0.0, 0.0)))
    molecule.add_atom(Atom(symbol="O", coords=(0.0, 0.0, 1.128)))
    return molecule.add_hatoms()


class ConstantEnergyLabeller(SPLabeller):
    value: float = 0.0

    def run(self, engine, dataset, **kwargs):
        labelled = InMemoryMolData.create(available_properties=self.properties)
        for row in dataset:
            properties = {prop.name: self.value for prop in self.properties}
            labelled.add_system(ChemDataEntry(system=row.system, properties=properties, metadata=row.metadata))
        return labelled, {}


class NoOpTrainer(MLIPTrainer[AMSEngine]):
    def validate_engine(self, engine):
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


@dataclass
class DummyJobInfo:
    settings: Settings
    path: str


@dataclass
class DummyResults:
    rkf_path: str
    job: DummyJobInfo
    history: Dict[str, List[float]]
    n_atoms: int
    exit_message: str = ""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "DummyResults":
        settings = Settings()
        settings.input.ams.moleculardynamics.NSteps = payload["n_steps"]
        settings.input.ams.moleculardynamics.Trajectory.SamplingFreq = payload["sampling_freq"]
        job = DummyJobInfo(settings=settings, path=payload["rkf_path"])
        return cls(
            rkf_path=payload["rkf_path"],
            job=job,
            history={
                "Temperature": payload["temperatures"],
                "EngineEnergy": payload["energies"],
            },
            n_atoms=payload["n_atoms"],
            exit_message=payload.get("exit_message", ""),
        )

    def rkfpath(self) -> str:
        return self.rkf_path

    def get_history_property(self, key: str, history_section: str | None = None):
        return self.history.get(key)

    def get_exit_condition_message(self) -> str:
        return self.exit_message

    def get_main_molecule(self):
        return [None] * self.n_atoms

    def readrkf(self, section: str, key: str):
        if section == "History" and key == "nEntries":
            return len(self.history.get("Temperature", []))
        raise KeyError(f"Unknown {section=} {key=}")


class DummyParallelization:
    def __init__(self, result: AMSTaskResult):
        self.result = result
        self.calls: List[List[Tuple[object, object]]] = []

    def run_tasks(self, task_engine_coll):
        self.calls.append(task_engine_coll)
        return [self.result]


def _previous_run_payload(rkf_path: Path, n_steps: int = 10, sampling_freq: int = 2) -> Dict[str, Any]:
    n_entries = int(n_steps / sampling_freq)
    return {
        "rkf_path": str(rkf_path),
        "n_steps": n_steps,
        "sampling_freq": sampling_freq,
        "temperatures": [300.0] * n_entries,
        "energies": [0.0] * n_entries,
        "n_atoms": 2,
    }


def _assert_expected_previous_results_warnings(caught_warnings: list[warnings.WarningMessage]) -> None:
    messages = [(warning.category, str(warning.message)) for warning in caught_warnings]
    assert any(category is RKFDataWarning and "MDHistory is not found" in message for category, message in messages)


def test_simple_active_learning_loop_uses_previous_results():
    MOLITERATE_PROGRESS_BAR_CONFIG.disable = True

    rkf_path = Path(__file__).resolve().parents[2] / "moliterate" / "tutorials" / "data" / "molecule.rkf"
    assert rkf_path.exists()

    sal = SimpleActiveLearningJourney(
        checker_getter=AMSTrajChecker() + AMSTrajGetter(data_selector_if_low_data=(None, None, 3)),
        task=AMSMDTask(
            samplingfreq=2,
            thermostat="NHC",
            atomistic_system=SCMChemicalSystem.from_molecules(
                system_id="CO",
                systems=_co_molecule(),
            ),
        ),
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=10, cumulative_values=[10]),
        max_attempts=1,
        first_step_train=True,
    )
    start_engine = AMSEngine(engine_id="start")
    task_result = AMSTaskResult.model_construct(
        task=sal._current_batch_tasks()[0],
        engine=start_engine,
        plams_results=DummyResults.from_payload(_previous_run_payload(rkf_path)),
    )
    parallelization = DummyParallelization(task_result)
    sal.task_parallelization = parallelization

    props = [PropertyInfo(name="energy", unit="eV")]
    loop = ActiveLearningLoop.model_construct(
        start_engine=start_engine,
        journey=sal,
        iterable_loop=StoppableCounter(stop=2),
        labeller=ConstantEnergyLabeller(properties=props),
        labeller_engine=AMSEngine(engine_id="labeller"),
        accuracy_checker=PairwiseDatasetMetrics(
            settings=[PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", target=1.0)]
        ),
        splitting=DataAndSplittingStrategy(dataset=InMemoryMolData.create(available_properties=props)),
        mlip_trainer=NoOpTrainer(),
        callbacks=[],
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        loop.run()

    assert loop.result is not None
    assert len(loop.result.iterations) == 1
    assert len(parallelization.calls) == 1
    assert len(parallelization.calls[0]) == 1
    assert parallelization.calls[0][0][1] is start_engine
    _assert_expected_previous_results_warnings(caught_warnings)
    assert len(loop.result.iterations[0].task_results.get_final_dataset()) > 0


# def test_simple_active_learning_tutorial_layout_and_cleanup(monkeypatch, tmp_path):
#     MOLITERATE_PROGRESS_BAR_CONFIG.disable = True

#     rkf_path = Path(__file__).resolve().parents[2] / "moliterate" / "tutorials" / "data" / "molecule.rkf"
#     assert rkf_path.exists()

#     results_path = tmp_path / "previous_run.json"
#     results_path.write_text(json.dumps(_previous_run_payload(rkf_path, n_steps=50, sampling_freq=2)))

#     dummy_job = DummyRunJob(results_path)

#     def fake_build_job(self, engine, extra=None):
#         return dummy_job

#     monkeypatch.setattr(AMSTask, "build_job", fake_build_job)
#     monkeypatch.setattr(simple_active_learning, "AMSTaskResult", DummyAMSTaskResult)

#     workspace_root = tmp_path / "workspace_snapshot"
#     (workspace_root / "keep_a").mkdir(parents=True)
#     (workspace_root / "keep_b" / "nested").mkdir(parents=True)
#     (workspace_root / "keep_b" / "marker.txt").write_text("marker")
#     before_other_paths = sorted(
#         p.relative_to(workspace_root).as_posix() for p in workspace_root.rglob("*")
#     )

#     # Tutorial setup from examples-simple-active-learning.ipynb (using a light trainer/labeller for test speed).
#     sal = SimpleActiveLearningJourney(
#         checker_getter=AMSTrajChecker() + AMSTrajGetter(data_selector_if_low_data=(None, None, 3)),
#         task=AMSMDTask(
#             samplingfreq=2,
#             thermostat="NHC",
#             atomistic_system=AtomisticDictStrContainer.from_molecules(
#                 system_id="CO",
#                 systems=from_smiles("CO"),
#             ),
#         ),
#         steps=[50, 120],
#         max_attempts=3,
#         first_step_train=True,
#     )
#     props = [
#         PropertyInfo(name="energy", unit="eV"),
#         PropertyInfo(name="forces", unit="eV/Ang"),
#     ]
#     run_dir = workspace_root / "tutorial_simple_active_learning_run"
#     loop = ActiveLearningLoop(
#         start_engine=AMSEngine(engine_id="start"),
#         journey=sal,
#         iterable_loop=StoppableCounter(stop=1),
#         labeller=ConstantEnergyLabeller(properties=props),
#         labeller_engine=AMSEngine(engine_id="labeller"),
#         accuracy_checker=PairwiseDatasetMetrics(
#             settings=[PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae")]
#         ),
#         splitting=DataAndSplittingStrategy(dataset=InMemoryMolData.create(available_properties=props)),
#         mlip_trainer=NoOpTrainer(),
#         callbacks=[
#             FolderManagerCallback(root_dir=RootDir(run_root=run_dir)),
#         ],
#     )

#     loop.run()

#     assert loop.last_run is not None
#     assert len(loop.last_run.iterations) == 1
#     assert dummy_job.run_calls == 1
#     assert dummy_job.loaded_path == results_path
#     assert len(loop.last_run.iterations[0].task_results.get_final_dataset()) > 0

#     # Folder layout checks for tutorial-like run output.
#     assert run_dir.exists()
#     assert run_dir.is_dir()
#     assert (run_dir / "al_start_state.json").is_file()
#     assert (run_dir / "al_final_state.json").is_file()
#     assert (run_dir / "train_validation_AL.db").is_file()
#     assert (run_dir / "iter_00").is_dir()
#     assert (run_dir / "iter_00" / "al_state.json").is_file()

#     shutil.rmtree(run_dir)
#     after_other_paths = sorted(
#         p.relative_to(workspace_root).as_posix() for p in workspace_root.rglob("*")
#     )
#     assert after_other_paths == before_other_paths
