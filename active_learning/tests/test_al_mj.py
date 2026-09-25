from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import pytest
from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.utils.progress_bar import MOLITERATE_PROGRESS_BAR_CONFIG

from scm.active_learning import ActiveLearningLoop, SimpleActiveLearningJourney
from scm.active_learning.checker_getters.checkers import AMSTrajChecker
from scm.active_learning.checker_getters.getters import AMSTrajGetter
from scm.active_learning.engines import AMSEngine, SCMChemicalSystem
from scm.active_learning.loop import StoppableCounter
from scm.active_learning.mlip import DataAndSplittingStrategy
from scm.active_learning.results.task import AMSTaskResult
from scm.active_learning.tasks import AMSMDTask

_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from test_al_sal import (  # noqa: E402
    ConstantEnergyLabeller,
    DummyResults,
    NoOpTrainer,
    _assert_expected_previous_results_warnings,
    _co_molecule,
    _previous_run_payload,
)

pytest.importorskip("scm.base")
pytest.importorskip("scm.plams")
pytest.importorskip("scm.params")


class DummyParallelization:
    def __init__(self, result: AMSTaskResult):
        self.result = result
        self.calls: List[List[Tuple[object, object]]] = []

    def run_tasks(self, task_engine_coll):
        self.calls.append(task_engine_coll)
        return [self.result]


def test_simple_active_learning_example_uses_previous_results():
    MOLITERATE_PROGRESS_BAR_CONFIG.disable = True

    rkf_path = Path(__file__).resolve().parents[2] / "moliterate" / "tutorials" / "data" / "molecule.rkf"
    assert rkf_path.exists()

    sal = SimpleActiveLearningJourney(
        checker_getter=AMSTrajChecker() + AMSTrajGetter(),
        task=AMSMDTask(
            samplingfreq=2,
            thermostat="NHC",
            atomistic_system=SCMChemicalSystem.from_molecules(
                system_id="CO",
                systems=_co_molecule(),
            ),
        ),
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=100, cumulative_values=[50, 100]),
        max_attempts=3,
        first_step_train=True,
    )
    start_engine = AMSEngine(engine_id="start")
    task_result = AMSTaskResult.model_construct(
        task=sal._current_batch_tasks()[0],
        engine=start_engine,
        plams_results=DummyResults.from_payload(_previous_run_payload(rkf_path, n_steps=50, sampling_freq=2)),
    )
    parallelization = DummyParallelization(task_result)
    sal.task_parallelization = parallelization

    props = [
        PropertyInfo(name="energy", unit="eV"),
        PropertyInfo(name="forces", unit="eV/Ang"),
    ]
    loop = ActiveLearningLoop.model_construct(
        start_engine=start_engine,
        journey=sal,
        iterable_loop=StoppableCounter(stop=1),
        labeller=ConstantEnergyLabeller(properties=props),
        labeller_engine=AMSEngine(engine_id="labeller"),
        accuracy_checker=PairwiseDatasetMetrics(
            settings=[PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae")]
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


# def test_molecules_journey_example_folder_structure(monkeypatch, tmp_path):
#     MOLITERATE_PROGRESS_BAR_CONFIG.disable = True
#     monkeypatch.chdir(tmp_path)

#     rkf_path = Path(__file__).resolve().parents[2] / "moliterate" / "tutorials" / "data" / "molecule.rkf"
#     assert rkf_path.exists()

#     results_path = tmp_path / "previous_run.json"
#     results_path.write_text(json.dumps(_previous_run_payload(rkf_path, n_steps=50, sampling_freq=2)))
#     dummy_job = DummyRunJob(results_path)

#     def fake_build_job(self, engine, extra=None):
#         return dummy_job

#     def fake_run_single_point(self, properties, dataset, parallel_settings):
#         return [{prop.name: 0.0 for prop in properties} for _ in dataset]

#     monkeypatch.setattr(AMSTask, "build_job", fake_build_job)
#     monkeypatch.setattr(AMSEngine, "run_single_point", fake_run_single_point)

#     smiles_list = ["CO", "CCO", "CC", "O"]
#     db_path = Path("molecules.db")
#     if db_path.exists():
#         db_path.unlink()

#     molecules = ASEDatabase.create(db_path, available_properties=[])
#     molecules.add_systems(
#         ChemDataEntry(system=from_smiles(smiles), metadata={"smiles": smiles}) for smiles in smiles_list
#     )

#     md_checker_getter = AMSTrajChecker() + AMSTrajGetter(data_selector_if_low_data=(None, None, 8))
#     go_simple_checker_getter = AMSTrajChecker() + AMSTrajGetter(data_selector_if_low_data=(None, None, 3))

#     md_settings = AMSMDTask.model_construct(nsteps=50, thermostat="NHC", temperature=300, samplingfreq=2).settings
#     md_settings2 = AMSMDTask.model_construct(nsteps=50, thermostat="NHC", temperature=700, samplingfreq=2).settings
#     go_settings = AMSGOIRTask.model_construct(normal_modes=False).settings

#     molecules_journey = MoleculesJourney(
#         molecules=molecules,
#         task_checker_getter={
#             "md300": ("ams", md_settings.as_dict(), md_checker_getter),
#             "md700": ("ams", md_settings2.as_dict(), md_checker_getter),
#             "go": ("ams", go_settings.as_dict(), go_simple_checker_getter),
#         },
#         max_attempts_per_task={"md300": 1, "md700": 1, "go": 2},
#         batch_size=2,
#     )

#     run_dir = tmp_path / "al_mj_run"
#     al = ActiveLearningLoop(
#         start_engine=AMSEngine.Builder.UFF().build(),
#         journey=molecules_journey,
#         iterable_loop=StoppableCounter(stop=3),
#         labeller=SPLabeller(
#             properties=[
#                 PropertyInfo(name="energy", unit="eV"),
#                 PropertyInfo(name="forces", unit="eV/Ang"),
#             ]
#         ),
#         labeller_engine=AMSEngine.Builder.DFTB_GFN1().build(),
#         accuracy_checker=PairwiseDatasetMetrics(
#             settings=[
#                 PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", per_n_atoms=True, target=0.01)
#             ]
#         ),
#         mlip_trainer=NoOpTrainer(),
#         callbacks=[FolderManagerCallback(root_dir=RootDir(run_root=run_dir))],
#     )

#     al.run()

#     folder_callback = al.query_callbacks(FolderManagerCallback)[0]
#     assert folder_callback.run_dir() == run_dir
#     assert run_dir.exists()
#     assert (run_dir / "al_start_state.json").exists()
#     assert (run_dir / "al_final_state.json").exists()
#     assert (run_dir / "train_validation_AL.db").exists()

#     iter_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith("iter_"))
#     assert len(iter_dirs) > 0

#     first_files = sorted(path.name for path in iter_dirs[0].iterdir() if path.is_file())
#     first_n_dirs = len([path for path in iter_dirs[0].iterdir() if path.is_dir()])
#     for iter_dir in iter_dirs[1:]:
#         assert sorted(path.name for path in iter_dir.iterdir() if path.is_file()) == first_files
#         assert len([path for path in iter_dir.iterdir() if path.is_dir()]) == first_n_dirs

#     shutil.rmtree(run_dir)
#     assert not run_dir.exists()
