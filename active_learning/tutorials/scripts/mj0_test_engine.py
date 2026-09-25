"""Tutorial: run a small `MoleculesJourney` batch and stop after validity/accuracy checks.

This tutorial is useful when you want to sanity-check an engine on a collection
of molecule tasks without running the full active-learning loop retraining step.

Run from the `active_learning` project root with:

```bash
uv run tutorials/scripts/mj0_test_engine.py
```
"""

from __future__ import annotations

import sys
from typing import List

from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.filters import ConditionsFilter
from scm.moliterate.interfaces import InMemoryMolData
from scm.plams import from_smiles
from tabulate import tabulate

from scm.active_learning import ActiveLearningLoop, MoleculesJourney
from scm.active_learning.callbacks import ALCallback, FolderManagerCallback
from scm.active_learning.checker_getters.checkers import AMSTrajChecker, ConfJobFailChecker
from scm.active_learning.checker_getters.getters import AMSTrajGetter, ConformersGetter
from scm.active_learning.engines import AMSEngine
from scm.active_learning.loop import StoppableCounter
from scm.active_learning.loop.iteration_phase import IterPhase
from scm.active_learning.mlip.params_trainer import ParAMSTrainer
from scm.active_learning.tasks import AMSConformersTask, AMSGOIRTask, AMSMDTask, SPLabeller


class StopAfterChecks(ALCallback):
    type: str = "StopAfterChecks"
    reason: str = "Tutorial engine test: skip dataset splitting and MLIP training"

    def start_iter_loop(self, al_loop: ActiveLearningLoop, state) -> None:
        del al_loop
        state.skip[IterPhase.SPLIT_AND_ADD] = self.reason
        state.skip[IterPhase.TRAIN] = self.reason


def build_molecules() -> InMemoryMolData:
    smiles_list = ["CO", "CCO", "CC", "O"]
    molecules = InMemoryMolData.create(available_properties=[])
    molecules.add_systems(
        ChemDataEntry(system=from_smiles(smiles), metadata={"smiles": smiles}) for smiles in smiles_list
    )
    return molecules


def build_task_checker_getter():
    md_checker_getter = AMSTrajChecker() + AMSTrajGetter()
    conformer_checker_getter = ConfJobFailChecker() + ConformersGetter()

    md_300 = AMSMDTask.model_construct(
        nsteps=1000,
        thermostat="NHC",
        temperature=300.0,
        samplingfreq=10,
    ).settings
    md_700 = AMSMDTask.model_construct(
        nsteps=1000,
        thermostat="NHC",
        temperature=700.0,
        samplingfreq=10,
    ).settings
    go = AMSGOIRTask.model_construct(normal_modes=False, dipole_moment=False).settings
    conformers = AMSConformersTask.build_rdkit_settings()

    return {
        "md300": ("ams", md_300.as_dict(), md_checker_getter),
        "md700": ("ams", md_700.as_dict(), md_checker_getter),
        "go": ("ams", go.as_dict(), md_checker_getter),
        "conf": ("conformers", conformers, conformer_checker_getter),
    }


def build_journey(batch_size: int = 2) -> MoleculesJourney:
    task_checker_getter = build_task_checker_getter()
    molecules = build_molecules()
    return MoleculesJourney(
        molecules=molecules,
        task_checker_getter=task_checker_getter,
        max_attempts_per_task={task_id: 1 for task_id in task_checker_getter},
        batch_size=batch_size,
    )


def build_test_engine():
    return AMSEngine.Builder.UFF().build()


def build_labeller_engine():
    return AMSEngine.Builder.UFF(dipole_moment=False).build()


def build_accuracy_settings() -> List[PairwiseDatasetMetrics.PropMetricEv]:
    return [
        PairwiseDatasetMetrics.PropMetricEv(
            property="energy",
            metric="mae",
            per_n_atoms=True,
            target=0.05,
        ),
        PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", target=0.10),
        PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="max_abs", target=0.60),
    ]


def build_loop(run_tag: str, journey: MoleculesJourney) -> ActiveLearningLoop:
    trainer = ParAMSTrainer.Builder.TEST().set_committee(committee_size=1, parallel=1).build()
    al = ActiveLearningLoop(
        tag=run_tag,
        start_engine=build_test_engine(),
        journey=journey,
        iterable_loop=StoppableCounter(stop=1),
        labeller=SPLabeller(
            properties=[
                PropertyInfo(name="energy", unit="eV"),
                PropertyInfo(name="forces", unit="eV/Ang"),
            ]
        ),
        labeller_engine=build_labeller_engine(),
        post_filter=ConditionsFilter(conditions="@fmax<18"),
        accuracy_checker=PairwiseDatasetMetrics(settings=build_accuracy_settings()),
        mlip_trainer=trainer,
        callbacks=[
            FolderManagerCallback(),
            StopAfterChecks(),
        ],
    )
    al.query_callback(FolderManagerCallback).logging_file_level = "DEBUG"
    return al


def print_summary(al: ActiveLearningLoop) -> None:
    state = al.result.iterations[-1]

    print(al.journey.journey_table())
    print("")
    print(al.journey.history_table(options="All"))

    print("")
    print("Iteration summary:")
    for key, value in state.dump_task_valid_getter_accurate_summary().items():
        print(f"  - {key}: {value}")

    if state.task_results is not None:
        print("")
        print("Validity results:")
        print(state.task_results.validity_results_table())
        print("")
        print("Getter results:")
        print(state.task_results.getter_results_table())

    if state.accuracy_checks_results is not None:
        print("")
        print("Accuracy results:")
        print(
            tabulate(
                [check.to_compact_dict() for check in state.accuracy_checks_results],
                headers="keys",
            )
        )

    print("")
    print("Loop exit:")
    print(al.result.exit_message)


def main(
    run_tag: str = "al0-test-engine",
) -> None:
    ActiveLearningLoop.logging_config(
        clean_sinks="YES",
        new_sink=sys.stderr,
        level="INFO",
    )

    journey = build_journey()
    al = build_loop(run_tag=run_tag, journey=journey)
    al.run()
    print_summary(al)


if __name__ == "__main__":
    main()
