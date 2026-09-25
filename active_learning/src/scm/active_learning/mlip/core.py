from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, Optional, TypeVar

from pydantic import BaseModel, field_validator

from scm.active_learning.engines import ConcreteEngines, Engine
from scm.active_learning.logging.path_serializable_model import PathSerializableModel
from scm.active_learning.results import MLIPTrainerResults

from .splitting_strategy import DataAndSplittingStrategy

T = TypeVar("T", bound=Engine)


class MLIPTrainerSaveSettings(BaseModel):
    settings: bool = False
    results: bool = False
    settings_file_name: str = "mlip_trainer_settings.json"
    results_file_name: str = "mlip_trainer_results.json"

    @field_validator("settings_file_name", "results_file_name")
    @classmethod
    def _validate_non_empty_file_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Save file names must not be blank.")
        return stripped

    def settings_path(self, run_dir: Path) -> Path:
        return run_dir / self.settings_file_name

    def results_path(self, run_dir: Path) -> Path:
        return run_dir / self.results_file_name

    def persist_trainer(self, *, trainer: PathSerializableModel, run_dir: Path) -> Optional[Path]:
        if not self.settings:
            return None
        settings_path = self.settings_path(run_dir)
        trainer.write_model_json(settings_path)
        return settings_path

    def persist_results(
        self,
        *,
        results: MLIPTrainerResults,
        run_dir: Path,
        settings_path: Optional[Path] = None,
    ) -> Optional[Path]:
        if settings_path is not None:
            results.train_infos["settings_json_path"] = str(settings_path)
        if not self.results:
            return None
        results_path = self.results_path(run_dir)
        results.write_model_json(results_path)
        return results_path


class MLIPTrainer(PathSerializableModel, Generic[T]):
    # type: Literal[""] = ""
    save: MLIPTrainerSaveSettings = MLIPTrainerSaveSettings()

    @property
    def ml_name(self) -> str:
        return "MLIP"

    @abstractmethod
    def validate_engine(
        self,
        engine: ConcreteEngines,
    ) -> T: ...

    @abstractmethod
    def train(self, data_split: DataAndSplittingStrategy, engine_id: str, **kwargs) -> MLIPTrainerResults: ...

    @abstractmethod
    def finetune(
        self, engine: T, data_split: DataAndSplittingStrategy, engine_id: str, **kwargs
    ) -> MLIPTrainerResults: ...

    @abstractmethod
    def to_data_format(
        self,
        data_split: DataAndSplittingStrategy,
    ) -> Path: ...

    def iter_tidy_error_metric_rows(self, path: Optional[Path] = None) -> Iterable[Dict[str, Any]]:
        del path
        raise NotImplementedError(f"{self.__class__.__name__} does not implement iter_tidy_error_metric_rows().")
