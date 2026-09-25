import shutil
import sys

from mj1_m3gnet import get_mol_journey, get_molecules, get_start_mol_journey, get_task_checker_getter
from sal3_maceEFD import get_trainer
from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.filters import ConditionsFilter

from scm.active_learning import ActiveLearningLoop, MoleculesJourney
from scm.active_learning.callbacks import (
    FolderManagerCallback,
)
from scm.active_learning.engines import AMSEngine
from scm.active_learning.loop import StoppableCounter
from scm.active_learning.tasks import SPLabeller


def get_al(
    molecules_journey: MoleculesJourney,
    trainer,
    start_engine=None,
    stop=10,
) -> ActiveLearningLoop:
    al = ActiveLearningLoop(
        start_engine=start_engine,
        journey=molecules_journey,
        iterable_loop=StoppableCounter(stop=stop),
        labeller=SPLabeller(
            properties=[
                PropertyInfo(name="energy", unit="eV"),
                PropertyInfo(name="forces", unit="eV/Ang"),
                PropertyInfo(name="dipole", unit="e*Ang"),
            ]
        ),
        labeller_engine=AMSEngine.Builder.UFF(dipole_moment=True).build(),
        # labeller_engine=AMSEngine.Builder.DFTB_GFN1().build(),
        post_filter=ConditionsFilter(conditions="@fmax<18"),
        accuracy_checker=PairwiseDatasetMetrics(
            settings=[
                PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", per_n_atoms=True, target=0.05),
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="max_scaled_abs", target=0.5),
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="max_abs", target=0.6),
                PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", target=0.1),
                PairwiseDatasetMetrics.PropMetricEv(property="dipole", metric="mae", target=0.02),
            ]
        ),
        mlip_trainer=trainer,
    )
    al.query_callback(FolderManagerCallback).logging_file_level = "DEBUG"
    return al


def main():
    ActiveLearningLoop.logging_config(
        clean_sinks="YES",
        new_sink=sys.stderr,
        level="INFO",
    )
    molecules = get_molecules()
    task_checker_getter, start_task_checker_getter = get_task_checker_getter()
    molecules_journey_start = get_start_mol_journey(molecules=molecules, task_checker_getter=start_task_checker_getter)
    molecules_journey = get_mol_journey(molecules=molecules, task_checker_getter=task_checker_getter)

    trainer = get_trainer()
    al_start = get_al(
        molecules_journey=molecules_journey_start,
        start_engine=AMSEngine.Builder.UFF().build(),
        stop=1,
        trainer=trainer,
    )
    al_start.tag = "mj2_maceEFD-start"
    engine_0 = al_start.run()
    al = get_al(molecules_journey=molecules_journey, start_engine=engine_0, trainer=trainer)
    al.tag = "mj2_maceEFD"
    _ = al.run()


def post(al: ActiveLearningLoop):
    shutil.rmtree(al.query_callbacks(FolderManagerCallback)[0].run_dir())


if __name__ == "__main__":
    main()
