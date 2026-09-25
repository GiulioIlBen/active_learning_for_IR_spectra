import shutil
import sys

from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.filters import ConditionsFilter
from scm.moliterate.interfaces import InMemoryMolData
from scm.plams import from_smiles

from scm.active_learning import ActiveLearningLoop, MoleculesJourney
from scm.active_learning.callbacks import (
    FolderManagerCallback,
)
from scm.active_learning.checker_getters.checkers import (
    AMSTrajChecker,
    ConfJobFailChecker,
    NoneChecker,
)
from scm.active_learning.checker_getters.getters import AMSTrajGetter, ConformersGetter, NMSGetters
from scm.active_learning.engines import AMSEngine
from scm.active_learning.loop import StoppableCounter
from scm.active_learning.mlip.params_trainer import ParAMSTrainer
from scm.active_learning.task_parallelization import AMSParallelCPU
from scm.active_learning.tasks import AMSConformersTask, AMSGOIRTask, AMSMDTask, SPLabeller


def get_molecules() -> InMemoryMolData:
    smiles_list = ["CO", "CCO", "CC", "O"]
    molecules = InMemoryMolData.create(available_properties=[])
    molecules.add_systems(
        ChemDataEntry(system=from_smiles(smiles), metadata={"smiles": smiles}) for smiles in smiles_list
    )
    return molecules


def get_task_checker_getter():
    md_checker_getter = AMSTrajChecker() + AMSTrajGetter()
    md_300 = AMSMDTask.model_construct(nsteps=5000, thermostat="NHC", temperature=300.0, samplingfreq=2).settings
    md_700 = AMSMDTask.model_construct(nsteps=5000, thermostat="NHC", temperature=700.0, samplingfreq=2).settings
    conformer = AMSConformersTask.build_rdkit_settings()

    go_settings = AMSGOIRTask.model_construct(normal_modes=False, dipole_moment=False).settings
    go_simple_checker_getter = AMSTrajChecker() + AMSTrajGetter()
    # go_checker_getter = go_simple_checker_getter >> (
    #     AMSIRCommitteeAgreementChecker() + NMSGetters(max_n_structures=5)
    # )
    conf_getter = ConfJobFailChecker() + ConformersGetter()
    task_checker_getter = {
        "md300": ("ams", md_300.as_dict(), md_checker_getter),
        "md700": ("ams", md_700.as_dict(), md_checker_getter),
        "go": ("ams", go_settings.as_dict(), go_simple_checker_getter),
        "conf": ("conformers", conformer, conf_getter),
    }

    md_settings_start = AMSMDTask.model_construct(
        nsteps=1000, thermostat="NHC", temperature=300, samplingfreq=10
    ).settings
    md_none_getter = NoneChecker() + AMSTrajGetter()
    conf_none_getter = NoneChecker() + ConformersGetter()

    go_settings_start = AMSGOIRTask.model_construct(normal_modes=True, dipole_moment=False).settings
    go_none_getter_nms = go_simple_checker_getter >> (NoneChecker() + NMSGetters(max_n_structures=3))
    start_task_checker_getter = {
        "md": ("ams", md_settings_start.as_dict(), md_none_getter),
        "go": ("ams", go_settings_start.as_dict(), go_none_getter_nms),
        "conf": ("conformers", conformer, conf_none_getter),
    }
    return task_checker_getter, start_task_checker_getter


def get_mol_journey(molecules, task_checker_getter):
    molecules_journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=task_checker_getter,
        max_attempts_per_task={"md300": 5, "md700": 5, "go": 5, "conf": 5},
        batch_size=4,
        task_parallelization=AMSParallelCPU(maxjobs=4, nproc=1),
    )
    return molecules_journey


def get_start_mol_journey(molecules, task_checker_getter):
    molecules_journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=task_checker_getter,
        max_attempts_per_task={"md": 1, "go": 1, "conf": 1},
        batch_size=4,
        task_parallelization=AMSParallelCPU(maxjobs=4, nproc=1),
    )
    return molecules_journey


def get_al(molecules_journey: MoleculesJourney, start_engine=None, stop=10):
    al = ActiveLearningLoop(
        start_engine=start_engine,
        journey=molecules_journey,
        iterable_loop=StoppableCounter(stop=stop),
        labeller=SPLabeller(
            properties=[
                PropertyInfo(name="energy", unit="eV"),
                PropertyInfo(name="forces", unit="eV/Ang"),
                # PropertyInfo(name="dipole", unit="e*Ang"),
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
                # PairwiseDatasetMetrics.PropMetricEv(property="dipole", metric="mae", target=0.1),
            ]
        ),
        # mlip_trainer=ParAMSTrainer.Builder.TEST().set_committee(1).build(),
        mlip_trainer=ParAMSTrainer.Builder.M3GNet_EF().set_committee(1).build(),
    )
    al.query_callback(FolderManagerCallback).logging_file_level = "DEBUG"
    return al


def main(check: bool = False, production: bool = False):
    ActiveLearningLoop.logging_config(
        clean_sinks="YES",
        new_sink=sys.stderr,
        level="INFO",
    )
    molecules = get_molecules()
    task_checker_getter, start_task_checker_getter = get_task_checker_getter()
    molecules_journey_start = get_start_mol_journey(molecules=molecules, task_checker_getter=start_task_checker_getter)
    molecules_journey = get_mol_journey(molecules=molecules, task_checker_getter=task_checker_getter)
    al_start = get_al(molecules_journey=molecules_journey_start, start_engine=AMSEngine.Builder.UFF().build(), stop=1)
    al_start.tag = "mj1_m3gnet-start"
    engine_0 = al_start.run()
    al = get_al(molecules_journey=molecules_journey, start_engine=engine_0)
    al.tag = "mj1_m3gnet"
    final_engine = al.run()
    if check:
        print(al_start.journey.journey_table())
        print("")
        print(al.journey.journey_table())
        return
    print(al.analysis.get_summary())
    al.analysis.plot.save_to_pdf()

    # Final Run
    if production and "CONVERGED" in al.iterable_loop.reason:
        molecules_journey.restart_journey()
        molecules_journey.batch_size = 4
        molecules_journey.run_current_batch(final_engine)


def post(al: ActiveLearningLoop):
    shutil.rmtree(al.query_callbacks(FolderManagerCallback)[0].run_dir())


if __name__ == "__main__":
    main()
