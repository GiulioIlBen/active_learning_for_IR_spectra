from __future__ import annotations

import re
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from scm.active_learning.engines import ASEEngine, ConcreteEngines
from scm.active_learning.mlip.core import MLIPTrainer
from scm.active_learning.mlip.mace_trainer._optional_dependencies import _load_yaml_module
from scm.active_learning.mlip.mace_trainer.config import MACEConfigBuilder
from scm.active_learning.mlip.mace_trainer.metrics import MACETrainingMetrics
from scm.active_learning.mlip.mace_trainer.runner import (
    _patched_mace_load_foundations,
    run_train_mace_subprocess,
    train_mace,
)
from scm.active_learning.mlip.mace_trainer.settings import (
    MACEDataSettings,
    MACEModelSettings,
    MACEPreparedTrainingContext,
    MACETrainerSaveSettings,
    MACETrainSettings,
)
from scm.active_learning.mlip.splitting_strategy import DataAndSplittingStrategy
from scm.active_learning.results import MLIPTrainerResults


class _MACEMemberTrainingResult(BaseModel):
    member_index: int
    run_dir: Path
    model_dir: Path
    config: Dict[str, Any]
    model_path: str
    actual_num_epochs: Optional[int] = None
    log_metrics: Dict[str, List[Any]] = Field(default_factory=dict)


class _MACEMemberJob(BaseModel):
    """A committee member whose MACE config is written and ready to train."""

    member_index: int
    run_dir: Path
    context: MACEPreparedTrainingContext
    model_dir: Path
    config_path: Path
    config: Dict[str, Any]
    patch_load_foundations: bool

    @property
    def subprocess_log_path(self) -> Path:
        return self.run_dir / "mace_train_subprocess.log"


class MACETrainer(MLIPTrainer[ASEEngine]):
    type: Literal["MACETrainer"] = "MACETrainer"
    data: MACEDataSettings = Field(default_factory=MACEDataSettings)
    train_settings: MACETrainSettings = Field(default_factory=MACETrainSettings)
    property_aliases: Dict[str, Tuple[str, ...]] = {
        "energy": ("REF_energy", "energy", "potential_energy"),
        "forces": ("REF_forces", "forces", "gradients"),
        "stress": ("REF_stress", "stress"),
        "dipole": ("REF_dipole", "dipole", "dipole_moment"),
    }
    extra_config: Dict[str, Any] = Field(default_factory=dict)
    suppress_warnings: bool = True
    save: MACETrainerSaveSettings = Field(default_factory=MACETrainerSaveSettings)

    @property
    def ml_name(self) -> str:
        return "MACE"

    def validate_engine(
        self,
        engine: ConcreteEngines,
    ) -> ASEEngine:
        if isinstance(engine, ASEEngine):
            return engine
        raise TypeError(f"{type(engine)=} is not ASEEngine")

    def train(self, data_split: DataAndSplittingStrategy, engine_id: str, **kwargs) -> MLIPTrainerResults:
        if kwargs:
            raise ValueError("MACETrainer settings must be provided at initialization, not in train().")
        return self._run_training(data_split=data_split, engine_id=engine_id, base_model_files=None)

    def finetune(
        self,
        engine: ASEEngine,
        data_split: DataAndSplittingStrategy,
        engine_id: str,
        **kwargs,
    ) -> MLIPTrainerResults:
        if kwargs:
            raise ValueError("MACETrainer settings must be provided at initialization, not in finetune().")
        validated = self.validate_engine(engine)
        base_model_files = self._extract_model_files_from_ase_engine(validated)
        if len(base_model_files) == 0:
            raise ValueError("Could not determine model path from ASEEngine.calculator_kwargs for MACE fine-tuning.")
        return self._run_training(
            data_split=data_split,
            engine_id=engine_id,
            base_model_files=base_model_files,
            base_model_architecture=self._extract_model_architecture_from_ase_engine(validated),
        )

    def to_data_format(
        self,
        data_split: DataAndSplittingStrategy,
        run_dir: Optional[Path] = None,
    ) -> Path:
        resolved_run_dir = (self.data.folder if run_dir is None else run_dir).expanduser().resolve(strict=False)
        return self.data.export_dataset(data_split=data_split, run_dir=resolved_run_dir)

    def _run_training(
        self,
        data_split: DataAndSplittingStrategy,
        engine_id: str,
        base_model_files: Optional[Sequence[str]],
        base_model_architecture: Optional[Dict[str, Any]] = None,
    ) -> MLIPTrainerResults:
        member_base_model_files = self._resolve_member_base_model_files(base_model_files)
        run_dir = (self.data.folder / engine_id).expanduser().resolve(strict=False)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.to_data_format(data_split=data_split, run_dir=run_dir)
        context = self.data.prepare_training_context(data_split=data_split, run_dir=run_dir)

        settings_path = self.save.persist_trainer(trainer=self, run_dir=run_dir)
        member_jobs = [
            self._prepare_member_training(
                root_context=context,
                root_run_dir=run_dir,
                member_index=member_index,
                member_count=len(member_base_model_files),
                member_seed=self.train_settings.committee.member_seed(member_index),
                base_model_file=base_model_file,
                base_model_architecture=base_model_architecture,
            )
            for member_index, base_model_file in enumerate(member_base_model_files)
        ]
        if self.train_settings.committee.trains_in_parallel and len(member_jobs) > 1:
            self._execute_members_in_parallel(member_jobs)
        else:
            for job in member_jobs:
                self._execute_member_in_process(job)
        member_results = [self._finalize_member_training(job) for job in member_jobs]
        model_paths = [member_result.model_path for member_result in member_results]
        engine_model_paths: str | List[str] = model_paths[0] if len(model_paths) == 1 else model_paths
        train_infos = self._build_train_infos(
            context=context,
            run_dir=run_dir,
            member_results=member_results,
        )
        log_metrics = self._merge_member_log_metrics(member_results)
        results = MLIPTrainerResults(
            engine=ASEEngine(
                engine_id=engine_id,
                calculator="scm.active_learning.mlip.mace_trainer.calculator.AMSMACECalculator",
                calculator_kwargs=self._build_calculator_kwargs(
                    model_path=engine_model_paths,
                    data_split=data_split,
                    config=member_results[0].config,
                ),
                finetunable=True,
            ),
            log_metrics=log_metrics,
            train_infos=train_infos,  # ty:ignore[invalid-argument-type]
        )
        self.save.persist_results(results=results, run_dir=run_dir, settings_path=settings_path)
        return results

    def _prepare_member_training(
        self,
        *,
        root_context: MACEPreparedTrainingContext,
        root_run_dir: Path,
        member_index: int,
        member_count: int,
        member_seed: Optional[int],
        base_model_file: Optional[str],
        base_model_architecture: Optional[Dict[str, Any]],
    ) -> _MACEMemberJob:
        member_run_dir = self._member_run_dir(root_run_dir, member_index, member_count)
        member_run_dir.mkdir(parents=True, exist_ok=True)
        context = self._member_context(root_context, member_run_dir, member_count)
        model_dir = self.data.model_dir(member_run_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.data.config_path(member_run_dir)
        config = self._build_config(
            context=context,
            model_dir=model_dir,
            base_model_file=base_model_file,
            base_model_architecture=base_model_architecture,
            seed=member_seed,
        )
        self._write_yaml(config_path, config)
        return _MACEMemberJob(
            member_index=member_index,
            run_dir=member_run_dir,
            context=context,
            model_dir=model_dir,
            config_path=config_path,
            config=config,
            patch_load_foundations=base_model_file is not None and self.train_settings.patch_load_foundations,
        )

    def _execute_member_in_process(self, job: _MACEMemberJob) -> None:
        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.simplefilter("ignore")
            if not job.patch_load_foundations:
                train_mace(config_file_path=job.config_path, workdir=job.context.run_dir)
            else:
                train_mace(
                    config_file_path=job.config_path,
                    workdir=job.context.run_dir,
                    patched_load_foundations=_patched_mace_load_foundations,
                )

    def _execute_members_in_parallel(self, jobs: Sequence[_MACEMemberJob]) -> None:
        committee = self.train_settings.committee

        def run(job: _MACEMemberJob) -> None:
            run_train_mace_subprocess(
                config_file_path=job.config_path,
                workdir=job.context.run_dir,
                log_path=job.subprocess_log_path,
                patch_load_foundations=job.patch_load_foundations,
                gpu_id=committee.member_gpu(job.member_index),
                suppress_warnings=self.suppress_warnings,
            )

        with ThreadPoolExecutor(max_workers=committee.parallel_members) as pool:
            # list() re-raises the first member failure after all started members have finished
            list(pool.map(run, jobs))

    def _finalize_member_training(self, job: _MACEMemberJob) -> _MACEMemberTrainingResult:
        model_path = self._resolve_model_path(job.model_dir)
        actual_num_epochs = self._resolve_completed_epochs(job.context.run_dir / "checkpoints")
        if self.train_settings.delete_checkpoints_after_training:
            self._delete_training_checkpoints(job.context.run_dir / "checkpoints")
        return _MACEMemberTrainingResult(
            member_index=job.member_index,
            run_dir=job.run_dir,
            model_dir=job.model_dir,
            config=job.config,
            model_path=model_path,
            actual_num_epochs=actual_num_epochs,
            log_metrics=MACETrainingMetrics.collect_validation_log_metrics(job.model_dir),
        )

    def _resolve_member_base_model_files(
        self,
        base_model_files: Optional[Sequence[str]],
    ) -> List[Optional[str]]:
        member_count = self.train_settings.committee.size
        if base_model_files is None:
            return [None] * member_count
        files = [path for path in base_model_files if path]
        if len(files) == 0:
            return [None] * member_count
        if member_count == 1:
            return [files[0]]
        if len(files) == 1:
            return files * member_count
        if len(files) == member_count:
            return files
        raise ValueError(
            "Cannot fine-tune MACE committee: start engine provides "
            f"{len(files)} model path(s), but train_settings.committee.size={member_count}. "
            "Provide one base model to broadcast or one base model per committee member."
        )

    @staticmethod
    def _member_run_dir(root_run_dir: Path, member_index: int, member_count: int) -> Path:
        if member_count == 1:
            return root_run_dir
        return root_run_dir / f"member_{member_index:03d}"

    @staticmethod
    def _member_context(
        root_context: MACEPreparedTrainingContext,
        member_run_dir: Path,
        member_count: int,
    ) -> MACEPreparedTrainingContext:
        if member_count == 1:
            return root_context
        return root_context.model_copy(update={"run_dir": member_run_dir})

    @staticmethod
    def _build_train_infos(
        *,
        context: MACEPreparedTrainingContext,
        run_dir: Path,
        member_results: Sequence[_MACEMemberTrainingResult],
    ) -> Dict[str, int | float | str]:
        train_infos: Dict[str, int | float | str] = {
            "training_set_size": context.training_set_size,
            "validation_set_size": context.validation_set_size,
            "run_directory": str(run_dir),
        }
        if len(member_results) == 1:
            actual_num_epochs = member_results[0].actual_num_epochs
            if actual_num_epochs is not None:
                train_infos["actual_num_epochs"] = actual_num_epochs
            return train_infos

        train_infos["committee_size"] = len(member_results)
        for member_result in member_results:
            prefix = f"member_{member_result.member_index}"
            train_infos[f"{prefix}_run_directory"] = str(member_result.run_dir)
            if member_result.actual_num_epochs is not None:
                train_infos[f"{prefix}_actual_num_epochs"] = member_result.actual_num_epochs
        return train_infos

    @staticmethod
    def _merge_member_log_metrics(
        member_results: Sequence[_MACEMemberTrainingResult],
    ) -> Dict[str, List[Any]]:
        if len(member_results) == 1:
            return member_results[0].log_metrics

        metric_keys: List[str] = []
        for member_result in member_results:
            for key in member_result.log_metrics:
                if key not in metric_keys:
                    metric_keys.append(key)

        if len(metric_keys) == 0:
            return {}

        merged: Dict[str, List[Any]] = {"member": []}
        for key in metric_keys:
            merged[key] = []

        for member_result in member_results:
            row_count = max((len(values) for values in member_result.log_metrics.values()), default=0)
            for row_index in range(row_count):
                merged["member"].append(member_result.member_index)
                for key in metric_keys:
                    values = member_result.log_metrics.get(key, [])
                    merged[key].append(values[row_index] if row_index < len(values) else "")
        return merged

    def iter_tidy_error_metric_rows(self, path: Optional[Path] = None) -> Iterable[Dict[str, Any]]:
        train_log_path = self._resolve_tidy_metrics_train_log_path(path)
        yield from MACETrainingMetrics.iter_tidy_error_metric_rows_from_train_log(train_log_path)

    @classmethod
    def plot_training_metrics(
        cls,
        path: Path | str,
        metrics: Optional[Sequence[str]] = None,
        output_path: Path | str | None = None,
        figsize: Optional[Tuple[float, float]] = None,
        max_columns: int = 3,
    ) -> Any:
        """Plot numeric MACE training-log metrics against epoch.

        ``path`` may point to a MACE run directory, its ``results`` directory,
        or a concrete ``*_train.txt`` file.
        """
        return MACETrainingMetrics.plot_training_metrics(
            path=path,
            metrics=metrics,
            output_path=output_path,
            figsize=figsize,
            max_columns=max_columns,
        )

    def _resolve_tidy_metrics_train_log_path(self, path: Optional[Path]) -> Path:
        if path is not None:
            candidate = path.expanduser().resolve(strict=False)
            if candidate.name.endswith("_train.txt"):
                if not candidate.exists():
                    raise FileNotFoundError(f"MACE training log file not found: {candidate}")
                return candidate
            run_dir = candidate
        else:
            run_dir = self.data.folder.expanduser().resolve(strict=False)

        model_dir = self.data.model_dir(run_dir)
        train_logs = sorted(model_dir.glob("*_train.txt"))
        if len(train_logs) == 0:
            raise FileNotFoundError(f"MACETrainer could not find a training log matching '*_train.txt' in {model_dir}.")
        return train_logs[-1]

    def _config_builder(self) -> MACEConfigBuilder:
        return MACEConfigBuilder(
            train_settings=self.train_settings,
            property_aliases=self.property_aliases,
            extra_config=self.extra_config,
        )

    def _build_config(
        self,
        context: MACEPreparedTrainingContext,
        model_dir: Path,
        base_model_file: Optional[str],
        base_model_architecture: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._config_builder().build(
            context=context,
            model_dir=model_dir,
            base_model_file=base_model_file,
            base_model_architecture=base_model_architecture,
            seed=seed,
        )

    def _build_calculator_kwargs(
        self,
        model_path: str | List[str],
        data_split: DataAndSplittingStrategy,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._config_builder().build_calculator_kwargs(
            model_path=model_path,
            data_split=data_split,
            config=config,
        )

    @staticmethod
    def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
        yaml_module = _load_yaml_module()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(yaml_module.safe_dump(data, sort_keys=False))

    @staticmethod
    def _resolve_model_path(model_dir: Path) -> str:
        model_files = sorted(x for x in model_dir.glob("*.model") if "_compiled" not in x.name)
        if model_files:
            prioritized = [x for x in model_files if "_swa" in x.stem] or model_files
            return str(prioritized[-1].resolve(strict=False))

        compiled_models = sorted(model_dir.glob("*_compiled.model"))
        if compiled_models:
            prioritized = [x for x in compiled_models if "_swa_compiled" in x.stem] or compiled_models
            return str(prioritized[-1].resolve(strict=False))

        checkpoints = sorted(model_dir.glob("*.pt"))
        if checkpoints:
            prioritized = [x for x in checkpoints if "swa" in x.name] or checkpoints
            return str(prioritized[-1].resolve(strict=False))
        raise FileNotFoundError(f"No trained model file found in {model_dir}")

    @staticmethod
    def _extract_model_from_ase_engine(engine: ASEEngine) -> Optional[str]:
        model_files = MACETrainer._extract_model_files_from_ase_engine(engine)
        if model_files:
            return model_files[0]
        return None

    @staticmethod
    def _extract_model_files_from_ase_engine(engine: ASEEngine) -> List[str]:
        candidates = (
            "model_file",
            "model_path",
            "model_paths",
            "checkpoint_path",
            "foundation_model",
        )
        for key in candidates:
            value = engine.calculator_kwargs.get(key)
            if isinstance(value, str) and value:
                return [value]
            if isinstance(value, (list, tuple)):
                model_files = [item for item in value if isinstance(item, str) and item]
                if model_files:
                    return model_files
        return []

    @classmethod
    def _extract_model_architecture_from_ase_engine(cls, engine: ASEEngine) -> Optional[Dict[str, Any]]:
        raw_value = engine.calculator_kwargs.get("mace_model_architecture")
        if not isinstance(raw_value, dict):
            return None
        architecture: Dict[str, Any] = {}
        for key in MACEModelSettings._ARCHITECTURE_KEYS:
            if key in raw_value:
                architecture[key] = raw_value[key]
        return architecture or None

    @staticmethod
    def _delete_training_checkpoints(checkpoints_dir: Path) -> None:
        if checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)

    @staticmethod
    def _resolve_completed_epochs(checkpoints_dir: Path) -> Optional[int]:
        if not checkpoints_dir.exists():
            return None

        max_epoch: Optional[int] = None
        for checkpoint in checkpoints_dir.glob("epoch-*.pt"):
            match = re.search(r"epoch-(\d+)", checkpoint.stem)
            if match is None:
                continue
            epoch = int(match.group(1))
            if max_epoch is None or epoch > max_epoch:
                max_epoch = epoch
        return max_epoch
