from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, List, Literal, Optional, Tuple, Union

from ase import Atoms
from ase.io import read as ase_read
from pydantic import BaseModel, JsonValue, model_validator
from scm.moliterate import ChemDataEntry, ChemDataSetFormat, ConcreteInterfaces, PropertyInfo, create_dataset
from tabulate import tabulate

from scm.active_learning.checker_getters import ConcreteCheckerGetter
from scm.active_learning.engines.atomic_container import SCMChemicalSystem
from scm.active_learning.logging import log_level
from scm.active_learning.results import (
    CollectionCheckGetResults,
    EngineCheckResult,
    JourneyAdvanceResult,
)
from scm.active_learning.tasks import AMSConformersTask, AMSTask, ConcreteTask

from .core import JourneyScheduler

if TYPE_CHECKING:
    from scm.moliterate.interfaces import ASEMolData

StatusOptions = Literal["Pending", "Running", "Finished", "Succeeded"]


class TaskStatus(BaseModel):
    status: StatusOptions = "Pending"
    max_attempt: int = -1
    attempt: int = 0
    attempt_messages: Dict[int, str] = {}

    def record_message(self, message: str):
        assert self.attempt not in self.attempt_messages
        self.attempt_messages[self.attempt] = message

    def end_task(
        self,
        message: str = "",
        status: Literal["Finished", "Succeeded"] = "Finished",
    ):
        if self.status != "Running":
            raise ValueError("Not Running...")
        self.status = status
        self.record_message(message)

    def record_failure(self, message: str):
        if self.attempt > self.max_attempt:
            self.status = "Finished"
        self.record_message(message)


class MolStatus(BaseModel):
    status: StatusOptions = "Pending"
    tasks: Dict[str, TaskStatus] = {}
    attempt: int = 0

    @classmethod
    def init_mol(cls, tasks: Dict[str, int]):
        return cls(tasks={k: TaskStatus(max_attempt=max_attempt) for k, max_attempt in tasks.items()})

    def tasks_query(self, status: StatusOptions):
        for couple_idx, x in self.tasks.items():
            if x.status == status:
                yield couple_idx, x

    def set_running(self):
        self.status = "Running"
        for _, x in self.tasks_query("Pending"):
            x.status = "Running"

    def register_attempt(self):
        self.attempt += 1
        for _, x in self.tasks_query("Running"):
            x.attempt += 1

    def set_remaining_tasks_succeeded(self):
        if self.status != "Running":
            raise ValueError(f"{self.status=} | {self}")
        for _, t in self.tasks_query("Running"):
            t.end_task(message="Succeeded", status="Succeeded")
        return self.is_it_finished()

    def set_task_succeeded(self, task_id: str):
        task = self.tasks[task_id]
        if task.status != "Running":
            raise ValueError(f"{task.status=} | {task_id=} | {self}")
        task.end_task(message="Succeeded", status="Succeeded")
        return self.is_it_finished()

    def record_task_failure(self, task_id: str, message: str):
        self.tasks[task_id].record_failure(message)
        return self.is_it_finished()

    def is_it_finished(self):
        n_couples = len(self.tasks)
        n_succeeded = len(list(self.tasks_query("Succeeded")))
        if n_succeeded == n_couples:
            self.status = "Succeeded"
            return True
        n_finished = len(list(self.tasks_query("Finished"))) + n_succeeded
        if n_finished == n_couples:
            self.status = "Finished"
            return True
        return False


class MoleculeJourneyRegistry(BaseModel):
    molecules: List[MolStatus] = []
    steps_record: List[List[int]] = []

    def molecules_query(self, status: Union[StatusOptions, Literal["ExcludePending", "All"]]):
        if status == "ExcludePending":
            for mol_idx, x in enumerate(self.molecules):
                if x.status == "Pending":
                    continue
                yield mol_idx, x
            return
        if status == "All":
            for mol_idx, x in enumerate(self.molecules):
                yield mol_idx, x
            return
        for mol_idx, x in enumerate(self.molecules):
            if x.status == status:
                yield mol_idx, x

    def clear_init_journey(self, n_molecules: int, tasks: Dict[str, int]):
        # Create independent MolStatus instances for each molecule.
        self.molecules = [MolStatus.init_mol(tasks=tasks) for _ in range(n_molecules)]
        # self.steps_record = []

    def n_pending_molecules(self) -> int:
        return len(list(self.molecules_query("Pending")))

    def n_inactivated_molecules(self) -> int:
        n_inactivated = len(list(self.molecules_query("Finished"))) + len(list(self.molecules_query("Succeeded")))
        return n_inactivated

    def n_running(self) -> int:
        return len(list(self.molecules_query("Running")))

    def current_running(self):
        for mol_idx, m in self.molecules_query("Running"):
            for couple_idx, _ in m.tasks_query("Running"):
                yield (mol_idx, couple_idx)

    def activate_n_running(self, batch: int, batch_by: Literal["molecules"] = "molecules"):
        running = list(self.molecules_query("Running"))
        active_limit = batch - len(running)
        if active_limit <= 0:
            return
        # sampling is ordered
        for _, m in self.molecules_query("Pending"):
            m.set_running()
            active_limit -= 1
            if active_limit <= 0:
                return

    def register_attempt(self):
        mols_running = []
        for mols_idx, m in self.molecules_query("Running"):
            m.register_attempt()
            mols_running.append(mols_idx)
        self.steps_record.append(mols_running)

    def set_molecule_succeeded(self, mol_idx: int):
        self.molecules[mol_idx].set_remaining_tasks_succeeded()

    def set_molecule_task_succeeded(self, mol_idx: int, task_id: str):
        return self.molecules[mol_idx].set_task_succeeded(task_id)

    def record_molecule_failure_on_task(self, mol_idx: int, task_id: str, message: str):
        return self.molecules[mol_idx].record_task_failure(task_id, message)


class MJState(BaseModel):
    type: Literal["MJState"] = "MJState"
    tot_pending: int = -1
    tot_inactivated: int = 0
    tot_succeeded: int = 0
    n_running: int = -1
    n_inactivated: int = 0
    n_succeeded: int = 0
    finished_msg: str = ""
    all_running_succeeded: bool = False


class MoleculeImportConfig(BaseModel):
    enabled: bool = True
    backend: Literal["ase"] = "ase"
    relative_path: str = "journey_molecules.db"
    original_molecules: Optional[Dict[str, JsonValue]] = None

    @staticmethod
    def _is_json_compatible(value: Any) -> bool:
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, dict):
            return all(
                isinstance(key, str) and MoleculeImportConfig._is_json_compatible(val) for key, val in value.items()
            )
        if isinstance(value, (list, tuple)):
            return all(MoleculeImportConfig._is_json_compatible(val) for val in value)
        return False

    @classmethod
    def _read_molecule_entries(cls, source_path: Path) -> Iterable[ChemDataEntry]:
        if not source_path.exists():
            raise FileNotFoundError(f"Molecule XYZ file not found: {source_path}")

        atoms_collection = ase_read(source_path, index=":")
        if isinstance(atoms_collection, Atoms):
            atoms_collection = [atoms_collection]

        for index, atoms in enumerate(atoms_collection):
            metadata = {str(key): val for key, val in atoms.info.items() if cls._is_json_compatible(val)}
            metadata.setdefault("name", metadata.get("name") or f"{atoms.get_chemical_formula()}-{index}")
            yield ChemDataEntry(system=atoms, metadata=metadata)

    def prepare_for_run_folder(
        self,
        journey: MoleculesJourney,
        run_dir: Path,
        available_properties: Optional[List[PropertyInfo]] = None,
    ) -> None:
        path_input = isinstance(journey.molecules, (str, Path))
        if (not self.enabled and not path_input) or self.original_molecules is not None:
            return
        if self.backend != "ase":
            raise ValueError(f"Unsupported molecule import backend: {self.backend}")

        dataset_path = (run_dir / self.relative_path).expanduser().resolve(strict=False)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.unlink(missing_ok=True)

        if path_input:
            source_path = Path(journey.molecules).expanduser().resolve()
            original_payload: Dict[str, JsonValue] = {"type": "xyz", "data_source": source_path.as_posix()}
            molecule_entries = self._read_molecule_entries(source_path)
            properties = available_properties or []
            distance_unit = "Ang"
        else:
            original_payload = journey.molecules.model_dump(mode="json", round_trip=True)
            molecule_entries = journey.molecules
            properties = journey.molecules.out_properties if available_properties is None else available_properties
            distance_unit = journey.molecules.distance_unit

        imported: ASEMolData = create_dataset(
            dataset_path,
            fmt=ChemDataSetFormat.ASE,
            available_properties=properties,
            distance_unit=distance_unit,
        )
        imported.add_systems(molecule_entries)

        self.original_molecules = original_payload
        journey.molecules = imported
        journey.restart_journey()


class MoleculesJourney(JourneyScheduler):
    """
    Current implementation support only scm native tasks!
    """

    type: Literal["MoleculesJourney"] = "MoleculesJourney"
    start_separator: ClassVar[str] = " MoleculesJourney Scheduler "
    history_separator: ClassVar[str] = " MoleculesJourney History "
    molecules: Union[ConcreteInterfaces, str, Path]
    molecule_import: MoleculeImportConfig = MoleculeImportConfig()
    batch_size: int = -1
    batch_by: Literal["molecules"] = "molecules"
    task_checker_getter: Dict[
        str,
        Tuple[Literal["ams", "conformers"], Dict[str, JsonValue], ConcreteCheckerGetter],
    ]
    max_attempts_per_task: Dict[str, int]
    history: List[MJState] = []
    registry_: MoleculeJourneyRegistry = MoleculeJourneyRegistry()

    def model_post_init(self, __context) -> None:
        # A newly constructed journey needs an initial registry.  A restored
        # checkpoint already carries the per-molecule/task state, however, and
        # must not be reset while Pydantic deserializes it.
        if not isinstance(self.molecules, (str, Path)) and not self.registry_.molecules:
            self.restart_journey()

    @model_validator(mode="after")
    def enable_molecule_import_for_path_input(self) -> MoleculesJourney:
        if isinstance(self.molecules, (str, Path)):
            self.molecule_import.enabled = True
        return self

    def restart_journey(self):
        self.registry_.clear_init_journey(n_molecules=len(self.molecules), tasks=self.max_attempts_per_task)

    def n_active_molecules(self) -> int:
        return len(self.registry_.molecules) - self.registry_.n_inactivated_molecules()

    @property
    def max_n_steps(self):
        coeff = 1 if self.max_attempts_per_task else max(list(self.max_attempts_per_task.values()))
        return self.n_steps() * coeff

    def n_steps(self) -> int:
        batched = -1
        if self.batch_by == "molecules":
            batched = -(-len(self.molecules) // self.batch_size)
        return batched

    def n_steps_left(self) -> int:
        batched = 0
        if self.batch_by == "molecules":
            batched = -(-self.n_active_molecules() // self.batch_size)
        return self.n_steps() - batched

    def is_finished(self) -> bool:
        return self.n_active_molecules() <= 0

    def finished_reason(self) -> str:
        n_molecules = len(self.registry_.molecules)
        n_succeeded = len(list(self.registry_.molecules_query("Succeeded")))
        if n_molecules > 0 and n_succeeded == n_molecules:
            return super().finished_reason()

        n_stopped = len(list(self.registry_.molecules_query("Finished")))
        return (
            "MoleculesJourney did not converge: "
            f"{n_succeeded}/{n_molecules} molecules succeeded; "
            f"{n_stopped} stopped without succeeding"
        )

    def register_attempt(self) -> None:
        self.registry_.activate_n_running(batch=self.batch_size, batch_by=self.batch_by)
        self.registry_.register_attempt()

    def prepare_for_run_folder(self, run_dir: Path) -> None:
        available_properties = [] if isinstance(self.molecules, (str, Path)) else None
        self.molecule_import.prepare_for_run_folder(self, run_dir, available_properties=available_properties)

    def _current_batch_tasks(self) -> List[ConcreteTask]:
        tasks: List[ConcreteTask] = []
        molecules = self.molecules
        couples = self.task_checker_getter
        for mol_idx, task_idx in self.registry_.current_running():
            entry = molecules[mol_idx]
            atomistic_system = SCMChemicalSystem.from_molecules(
                system_id=self.molecule_id_converter(mol_idx),
                systems=entry.chemical_system,
            )
            tasks.append(
                self._build_task(
                    task_idx,
                    couples[task_idx][0],
                    couples[task_idx][1],
                    atomistic_system,
                )
            )
        return tasks

    def _build_task(self, task_id, task_type, settings, system: SCMChemicalSystem) -> ConcreteTask:
        if task_type == "ams":
            return AMSTask(task_id=task_id, source_settings=settings, atomistic_system=system)
        if task_type == "conformers":
            return AMSConformersTask(task_id=task_id, source_settings=settings, atomistic_system=system)
        raise ValueError(f"{task_type=} not valid")

    def _current_batch_check_getters(self) -> List[ConcreteCheckerGetter]:
        return [self.task_checker_getter[task_idx][2] for _, task_idx in self.registry_.current_running()]

    def molecule_id_converter(self, idx: Union[str, int]) -> Union[str, int]:
        if isinstance(idx, int):
            return f"M{idx:04}"
        elif isinstance(idx, str):
            return int(idx.replace("M", ""))
        raise NotImplementedError

    ###############################################################################
    ###############################################################################
    ###############################################################################

    def advance(
        self,
        validity_results: CollectionCheckGetResults,
        accuracy_results: List[EngineCheckResult],
    ) -> JourneyAdvanceResult:
        """_summary_

        :param validity_results: _description_
        :type validity_results: CollectionCheckGetResults
        :param accuracy_results: _description_
        :type accuracy_results: List[EngineCheckResult]
        :return: return True if all the current tasks are successful. Therefore you can skip retraining the MLIP
        :rtype: bool
        """
        validity_grouped_system = defaultdict(list)
        for v in validity_results.validity_get_results:
            for vv in v.validity_results:
                validity_grouped_system[vv.system_id].append(vv)

        accuracy_grouped_system = defaultdict(list)
        for vv in accuracy_results:
            accuracy_grouped_system[vv.system_id].append(vv)

        def check_tasks_failure(checks: List[EngineCheckResult]):
            checks_grouped_task: Dict[str, List[EngineCheckResult]] = defaultdict(list)
            for x in checks:
                checks_grouped_task[x.task_id].append(x)
            for k, checks_grouped in checks_grouped_task.items():
                if not all(xx.is_success() for xx in checks_grouped):
                    yield k

        n_inactivated = 0
        n_succeeded = 0
        n_running = len(validity_grouped_system)
        failed_systems_current_batch = []
        for system_id, validity in validity_grouped_system.items():
            validity_failures = set(check_tasks_failure(validity))
            accuracy_failures = set(check_tasks_failure(accuracy_grouped_system.get(system_id, [])))
            mol_idx = self.molecule_id_converter(system_id)
            if len(validity_failures) == 0 and len(accuracy_failures) == 0:
                self.registry_.set_molecule_succeeded(mol_idx)
                n_succeeded += 1
                n_inactivated += 1
                continue
            failed_systems_current_batch.append(system_id)
            failed_tasks = validity_failures.union(accuracy_failures)
            successful_tasks = {
                task_id for task_id, task in self.registry_.molecules[mol_idx].tasks.items() if task.status == "Running"
            }.difference(failed_tasks)

            for k in successful_tasks:
                n_inactivated += self.registry_.set_molecule_task_succeeded(mol_idx, k)

            only_valid_failure = validity_failures.difference(accuracy_failures)
            only_accuracy_failure = accuracy_failures.difference(validity_failures)
            both_failure = validity_failures.intersection(accuracy_failures)
            for k in only_valid_failure:
                n_inactivated += self.registry_.record_molecule_failure_on_task(mol_idx, k, "ValidityFailure")
            for k in both_failure:
                n_inactivated += self.registry_.record_molecule_failure_on_task(mol_idx, k, "ValidityAccuracyFailure")
            for k in only_accuracy_failure:
                n_inactivated += self.registry_.record_molecule_failure_on_task(mol_idx, k, "AccuracyFailure")

        all_running_succeeded = False
        if len(failed_systems_current_batch) == 0:
            all_running_succeeded = True

        self._record_state(
            n_running,
            n_inactivated,
            n_succeeded,
            all_running_succeeded=all_running_succeeded,
        )
        self.log_last()
        return JourneyAdvanceResult(
            skip_training_reason=("All current tasks converged" if all_running_succeeded else "")
        )

    #############################################################################
    #############################################################################
    #############################################################################

    def _current_state(self) -> MJState:
        tot_pending = 0
        tot_inactivated = 0
        tot_succeeded = 0
        n_running = 0
        for mol in self.registry_.molecules:
            status = mol.status
            if status == "Pending":
                tot_pending += 1
            elif status == "Running":
                n_running += 1
            elif status == "Succeeded":
                tot_succeeded += 1
                tot_inactivated += 1
            elif status == "Finished":
                tot_inactivated += 1
        return MJState(
            tot_pending=tot_pending,
            tot_inactivated=tot_inactivated,
            tot_succeeded=tot_succeeded,
        )

    def _record_state(
        self,
        n_running: int,
        n_inactivated: int,
        n_succeeded: int,
        all_running_succeeded: bool = False,
    ) -> MJState:
        state = self._current_state()
        state.n_running = n_running
        state.n_inactivated = n_inactivated
        state.n_succeeded = n_succeeded
        state.all_running_succeeded = all_running_succeeded
        self.history.append(state)
        return state

    def log_last(self) -> None:
        n_running = self.history[-1].n_running
        n_inactivated = self.history[-1].n_inactivated
        log_level(
            "MoleculesJourney: {n_inactivated}/{n_running} got inactivated!",
            level="SUCCESS",
            n_running=n_running,
            n_inactivated=n_inactivated,
            depth=2,
        )

    def journey_table(self) -> str:
        rows = []
        for task_id, (_, _, checker) in self.task_checker_getter.items():
            rows.append(
                {
                    "task": task_id,
                    "checker": checker.checker_id,
                    "max_attempts": self.max_attempts_per_task[task_id],
                }
            )
        summary = f"{self.__class__.__name__} | NMolecules={len(self.molecules)}"
        table = tabulate(rows, headers="keys") if rows else ""
        return self.wrap_table(table=table, title=self.start_separator, subtitle=summary)

    def history_table(self, options: Literal["ExcludePending", "Running", "All"] = "Running") -> str:
        simple_history = (
            tabulate(
                [{"Idx": ii, **x.model_dump(exclude={"type"})} for ii, x in enumerate(self.history)], headers="keys"
            )
            if self.history
            else ""
        )
        attempts_history_list = self.get_history_attempts(options=options)
        attempts_history = tabulate(attempts_history_list, headers="keys") if attempts_history_list else ""
        history_body = simple_history or self._history_empty_message(attempts_history_list)
        sections = [self.wrap_table(table=history_body, title=self.history_separator)]
        if attempts_history_list:
            sections.append(self.wrap_table(table=attempts_history, title=" Recent Failures "))
        return "\n".join(sections)

    def _history_empty_message(self, attempts_history_list: List[Dict[str, str]]) -> str:
        if attempts_history_list:
            return (
                "No journey history has been recorded yet.\n"
                "Tasks are active, but convergence/advance has not produced a summary snapshot for this iteration.\n"
                "See 'Recent Failures' below for the current per-task status."
            )
        return (
            "No journey history has been recorded yet.\n"
            "No molecule/task attempts are available for the selected history view."
        )

    def get_history_attempts(self, options: Literal["ExcludePending", "Running", "All"] = "Running"):
        attempts_history_list = []
        for mol_idx, m in self.registry_.molecules_query(status=options):
            for t_id, t in m.tasks.items():
                attempts_history_list.append(
                    {
                        "Molecule": self.molecule_id_converter(mol_idx),
                        "MStatus": m.status,
                        "Task": t_id,
                        "TStatus": t.status,
                        "MaxAttempts": t.max_attempt,
                        "History": "-".join(t.attempt_messages.values()),
                    }
                )
        return attempts_history_list
