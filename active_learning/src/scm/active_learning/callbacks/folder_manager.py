from __future__ import annotations

import os
import re
import shutil
import string
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, List, Literal, Optional, Set, Tuple

import scm.plams as plams
from pydantic import AliasChoices, BaseModel, Field, PrivateAttr, field_validator
from scm.moliterate import ChemDataSetFormat, create_dataset

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.logging import add_sink, log_level
from scm.active_learning.logging.state_paths import collect_runtime_paths
from scm.active_learning.logging.state_serialization import StateFormat
from scm.active_learning.loop.run_control import ActiveLearningRunControl

if TYPE_CHECKING:
    from scm.active_learning.loop.iteration_state import IterationState
    from scm.active_learning.loop.loop import ActiveLearningLoop


class RootDir(BaseModel):
    run_root: Optional[Path] = None
    base_dir: Path = Path("ALruns")
    run_name: Optional[str] = None
    timestamp_format: str = "%Y%m%d_%H%M%S"

    def get_root_dir(self) -> Path:
        if self.run_root is None:
            run_name = self.run_name or time.strftime(self.timestamp_format)
            self.run_root = self.base_dir / run_name
        return self.run_root


class IterationCleanup(BaseModel):
    mode: Literal["off", "slim", "delete"] = "off"
    keep_last_full_iterations: int = 2
    preserve_root_entries: List[str] = Field(
        default_factory=lambda: ["al_state.json", "al_state.yaml", "al_state.yml", "job_logfile.csv", "logfile"]
    )

    @field_validator("keep_last_full_iterations")
    @classmethod
    def validate_keep_last_full_iterations(cls, value: int) -> int:
        if value < 0:
            raise ValueError("keep_last_full_iterations must be >= 0")
        return value

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def cleanup_old_iteration_dirs(
        self,
        al_loop: "ActiveLearningLoop",
        run_dir: Path,
        current_iteration: int,
        iter_dir_template: str,
    ) -> None:
        if not self.enabled:
            return
        iteration_dirs = self.iteration_dirs(run_dir=run_dir, iter_dir_template=iter_dir_template)
        if not iteration_dirs:
            return

        keep_dirs = {
            path for _, path in iteration_dirs[-self.keep_last_full_iterations :] if self.keep_last_full_iterations > 0
        }
        current_dir = run_dir.joinpath(iter_dir_template.format(current_iteration)).resolve(strict=False)
        keep_dirs.add(current_dir)

        protected_paths = self._protected_run_paths(al_loop=al_loop, run_dir=run_dir)
        slimmed_dirs = []
        for _, iter_dir in iteration_dirs:
            if iter_dir in keep_dirs:
                continue
            if self._cleanup_iteration_dir(iter_dir=iter_dir, protected_paths=protected_paths):
                slimmed_dirs.append(iter_dir)
        if slimmed_dirs:
            common_path = Path(os.path.commonpath([path.parent for path in slimmed_dirs])).as_posix().rstrip("/") + "/"
            folder_names = f"[{', '.join(path.name for path in slimmed_dirs)}]"
            log_level(
                "Slimmed old iteration folders: {common_path} | {folder_names}",
                level="DEBUG",
                common_path=common_path,
                folder_names=folder_names,
            )

    def iteration_dirs(self, run_dir: Path, iter_dir_template: str) -> List[Tuple[int, Path]]:
        run_dir = run_dir.resolve(strict=False)
        if not run_dir.exists():
            return []
        iteration_dirs = []
        for path in run_dir.iterdir():
            if not path.is_dir():
                continue
            iteration = self._parse_iteration_dir_name(name=path.name, iter_dir_template=iter_dir_template)
            if iteration is None:
                continue
            iteration_dirs.append((iteration, path.resolve(strict=False)))
        iteration_dirs.sort(key=lambda item: item[0])
        return iteration_dirs

    def _parse_iteration_dir_name(self, name: str, iter_dir_template: str) -> Optional[int]:
        parts = list(string.Formatter().parse(iter_dir_template))
        field_count = sum(field_name is not None for _, field_name, _, _ in parts)
        if field_count != 1:
            return None
        pattern = "^"
        inserted_iteration = False
        for literal_text, field_name, _, _ in parts:
            pattern += re.escape(literal_text)
            if field_name is not None and not inserted_iteration:
                pattern += r"(?P<iteration>\d+)"
                inserted_iteration = True
        pattern += "$"
        match = re.match(pattern, name)
        if match is None:
            return None
        return int(match.group("iteration"))

    def _protected_run_paths(self, al_loop: "ActiveLearningLoop", run_dir: Path) -> Set[Path]:
        run_dir = run_dir.expanduser().resolve(strict=False)
        payload = al_loop.model_dump(mode="json", round_trip=True)
        protected = set()
        for path in collect_runtime_paths(payload):
            if self._is_relative_to(path, run_dir):
                protected.add(path)
        return protected

    def _cleanup_iteration_dir(self, iter_dir: Path, protected_paths: Set[Path]) -> bool:
        protected_in_dir = {path for path in protected_paths if self._is_relative_to(path, iter_dir)}
        if self.mode == "delete" and len(protected_in_dir) == 0:
            shutil.rmtree(iter_dir)
            log_level("Deleted old iteration folder: {iter_dir}", level="DEBUG", iter_dir=iter_dir)
            return False
        self._slim_iteration_dir(iter_dir=iter_dir, protected_paths=protected_in_dir)
        return True

    def _slim_iteration_dir(self, iter_dir: Path, protected_paths: Set[Path]) -> None:
        for child in iter_dir.iterdir():
            if child.name in self.preserve_root_entries:
                continue
            self._prune_path(path=child.resolve(strict=False), protected_paths=protected_paths)

    def _prune_path(self, path: Path, protected_paths: Set[Path]) -> None:
        if path in protected_paths:
            return

        protected_descendants = [p for p in protected_paths if self._is_relative_to(p, path)]
        if len(protected_descendants) == 0:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            return

        if not path.is_dir():
            return

        for child in path.iterdir():
            self._prune_path(path=child.resolve(strict=False), protected_paths=protected_paths)

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True


class FolderManagerCallback(ALCallback):
    type: Literal["FolderManagerCallback"] = "FolderManagerCallback"
    reset_shared_dataset: bool = True
    make_dirs: bool = True
    set_plams_jobmanager_folder: bool = True
    set_ase_md_save_folder: bool = True
    root_dir: RootDir = RootDir()
    iter_dir_template: str = "iter_{:02d}"
    shared_dataset_name: str = "train_validation_AL.db"
    labeller_name: str = "labelled_data.db"
    mlip_dataname: str = "mlip_data"
    log_al_state: bool = Field(
        default=True,
        # TODO: Remove the log_al_state_json validation alias after old snapshots are migrated.
        validation_alias=AliasChoices("log_al_state", "log_al_state_json"),
    )
    state_format: StateFormat = "yaml"
    logging_file: Optional[str] = "loop.log"
    logging_file_level: Literal["INFO", "SUCCESS", "DEBUG"] = "INFO"
    cleanup: IterationCleanup = IterationCleanup()
    _original_plams_jobmanager: Any = PrivateAttr(default=None)
    _original_ase_md_save_folders: List[Tuple[object, str]] = PrivateAttr(default_factory=list)
    _original_labeller_out_path: Any = PrivateAttr(default=None)
    _has_labeller_out_path: bool = PrivateAttr(default=False)
    _original_trainer_data_folder: Any = PrivateAttr(default=None)
    _has_trainer_data_folder: bool = PrivateAttr(default=False)
    _original_trainer_datapath: Any = PrivateAttr(default=None)
    _has_trainer_datapath: bool = PrivateAttr(default=False)

    def run_dir(self):
        return self.root_dir.get_root_dir()

    @property
    # TODO: Remove after downstream callers migrate to log_al_state.
    def log_al_state_json(self) -> bool:
        """Compatibility alias for state files created before YAML support."""
        return self.log_al_state

    @log_al_state_json.setter
    def log_al_state_json(self, value: bool) -> None:
        self.log_al_state = value

    def state_path(self, directory: Path, stem: str) -> Path:
        return directory / f"{stem}.{self.state_format}"

    def iteration_dir(self, iteration: int) -> Path:
        return self.run_dir() / self.iter_dir_template.format(iteration)

    def shared_dataset_path(self) -> Path:
        return self.run_dir() / self.shared_dataset_name

    def labeller_path(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / self.labeller_name

    def mlip_datapath(self, iteration: int) -> Path:
        return self.iteration_dir(iteration) / self.mlip_dataname

    def _iter_ase_md_tasks(self, obj: object, visited: Optional[Set[int]] = None) -> Iterator[object]:
        from scm.active_learning.tasks.ase_md import ASEMolecularDynamics

        if obj is None or isinstance(obj, (str, bytes, Path)):
            return
        if visited is None:
            visited = set()
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, ASEMolecularDynamics):
            yield obj
            return

        if isinstance(obj, BaseModel):
            for field_name in obj.__class__.model_fields:
                yield from self._iter_ase_md_tasks(getattr(obj, field_name), visited)
            return

        if isinstance(obj, dict):
            values = obj.values()
        elif isinstance(obj, (list, tuple, set)):
            values = obj
        else:
            return

        for value in values:
            yield from self._iter_ase_md_tasks(value, visited)

    def plams_jobmanager_folder(self, iteration: int) -> None:
        folder_name = self.iter_dir_template.format(iteration)
        plams.config.default_jobmanager = plams.JobManager(
            plams.config.jobmanager,
            path=str(self.run_dir()),
            folder=folder_name,
            use_existing_folder=True,
        )

    @property
    def logging_path(self):
        if self.logging_file is None:
            return
        run_dir = self.run_dir()
        return run_dir / self.logging_file

    @staticmethod
    def _run_control(al_loop: "ActiveLearningLoop") -> ActiveLearningRunControl:
        return getattr(al_loop, "run_control", ActiveLearningRunControl())

    def cleanup_resume_iteration_dir(self, al_loop: "ActiveLearningLoop") -> None:
        run_control = self._run_control(al_loop)
        if not run_control.should_cleanup_resume_iteration():
            return
        iteration = run_control.start_iteration(al_loop)
        iter_dir = self.iteration_dir(iteration)
        if not iter_dir.exists():
            return
        if not iter_dir.is_dir():
            raise ValueError(f"Cannot clean resume iteration path because it is not a directory: {iter_dir}")
        shutil.rmtree(iter_dir)
        log_level("Resume cleanup removed incomplete iteration folder: {iter_dir}", iter_dir=iter_dir)

    def _capture_initial_folder_state(self, al_loop: "ActiveLearningLoop") -> None:
        if self.set_plams_jobmanager_folder and self._original_plams_jobmanager is None:
            self._original_plams_jobmanager = plams.config.default_jobmanager

        if self.set_ase_md_save_folder and not self._original_ase_md_save_folders:
            self._original_ase_md_save_folders = [
                (task, task.save_folder) for task in self._iter_ase_md_tasks(al_loop.journey)
            ]

        self._has_labeller_out_path = hasattr(al_loop.labeller, "out_path")
        if self._has_labeller_out_path:
            self._original_labeller_out_path = al_loop.labeller.out_path

        trainer_data = getattr(al_loop.mlip_trainer, "data", None)
        self._has_trainer_data_folder = trainer_data is not None and hasattr(trainer_data, "folder")
        if self._has_trainer_data_folder:
            self._original_trainer_data_folder = trainer_data.folder

        self._has_trainer_datapath = hasattr(al_loop.mlip_trainer, "datapath")
        if self._has_trainer_datapath:
            self._original_trainer_datapath = al_loop.mlip_trainer.datapath

    def _restore_initial_folder_state(self, al_loop: "ActiveLearningLoop") -> None:
        if self.set_plams_jobmanager_folder and self._original_plams_jobmanager is not None:
            plams.config.default_jobmanager = self._original_plams_jobmanager

        if self.set_ase_md_save_folder:
            for task, save_folder in self._original_ase_md_save_folders:
                task.save_folder = save_folder

        if self._has_labeller_out_path:
            al_loop.labeller.out_path = self._original_labeller_out_path

        trainer_data = getattr(al_loop.mlip_trainer, "data", None)
        if self._has_trainer_data_folder and trainer_data is not None and hasattr(trainer_data, "folder"):
            trainer_data.folder = self._original_trainer_data_folder

        if self._has_trainer_datapath and hasattr(al_loop.mlip_trainer, "datapath"):
            al_loop.mlip_trainer.datapath = self._original_trainer_datapath

        self._original_plams_jobmanager = None
        self._original_ase_md_save_folders = []
        self._original_labeller_out_path = None
        self._has_labeller_out_path = False
        self._original_trainer_data_folder = None
        self._has_trainer_data_folder = False
        self._original_trainer_datapath = None
        self._has_trainer_datapath = False

    def before_loop(self, al_loop: "ActiveLearningLoop", state=None) -> None:
        self._capture_initial_folder_state(al_loop)
        run_dir = self.run_dir()
        if self.make_dirs:
            run_dir.mkdir(parents=True, exist_ok=True)
            if self.logging_path is not None:
                add_sink(
                    file_path=self.logging_path,
                    level=self.logging_file_level,
                    format="verbose",
                )
            log_level("AL Loop Folder: {run_dir}", run_dir=run_dir)
        run_control = self._run_control(al_loop)
        if run_control.is_resume:
            self.cleanup_resume_iteration_dir(al_loop)
            log_level("Resuming AL Loop Folder: {run_dir}", run_dir=run_dir)
            return
        prepare_for_run_folder = getattr(al_loop.journey, "prepare_for_run_folder", None)
        if callable(prepare_for_run_folder):
            prepare_for_run_folder(run_dir)
        start_payload = al_loop.model_dump(mode="json", round_trip=True)
        if self.log_al_state:
            al_loop.write_payload(start_payload, self.state_path(run_dir, "al_start_state"))
        dataset_path = self.shared_dataset_path()
        if self.make_dirs:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if self.reset_shared_dataset and run_control.should_reset_shared_dataset():
            dataset_path.unlink(missing_ok=True)
        al_loop.splitting.dataset = create_dataset(
            dataset_path.as_posix(),
            fmt=ChemDataSetFormat.ASE,
            available_properties=al_loop.labeller.properties,
        )
        log_level("Train Val created: {dataset_path}", dataset_path=dataset_path)

    def start_iter_loop(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        iter_dir = self.iteration_dir(state.iteration_al)
        mlip_data_dir = self.mlip_datapath(state.iteration_al)
        if self.make_dirs:
            iter_dir.mkdir(parents=True, exist_ok=True)
        if self.set_ase_md_save_folder:
            for task in self._iter_ase_md_tasks(al_loop.journey):
                task.save_folder = str(iter_dir)
        if self.log_al_state:
            al_loop.write_model(self.state_path(iter_dir, "al_state"))
        if self.set_plams_jobmanager_folder:
            self.plams_jobmanager_folder(state.iteration_al)
        if hasattr(al_loop.labeller, "out_path"):
            al_loop.labeller.out_path = str(self.labeller_path(state.iteration_al))

        trainer_data = getattr(al_loop.mlip_trainer, "data", None)
        if trainer_data is not None and hasattr(trainer_data, "folder"):
            trainer_data.folder = mlip_data_dir

        if hasattr(al_loop.mlip_trainer, "datapath"):
            al_loop.mlip_trainer.datapath = str(mlip_data_dir)

    def after_loop(self, al_loop: "ActiveLearningLoop", state=None) -> None:
        if self.log_al_state:
            al_loop.write_model(self.state_path(self.run_dir(), "al_final_state"))
        self._restore_initial_folder_state(al_loop)

    def after_train(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        self.cleanup.cleanup_old_iteration_dirs(
            al_loop=al_loop,
            run_dir=self.run_dir(),
            current_iteration=state.iteration_al,
            iter_dir_template=self.iter_dir_template,
        )
