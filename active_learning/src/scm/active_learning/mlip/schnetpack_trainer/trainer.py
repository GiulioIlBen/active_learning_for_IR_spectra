from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel
from scm.moliterate.interfaces import ASEMolData

from scm.active_learning.engines import ASEEngine, ConcreteEngines
from scm.active_learning.mlip.core import MLIPTrainer
from scm.active_learning.mlip.schnetpack_trainer._optional_dependencies import (
    _check_installation,
    _load_schnetpack_training_deps,
)
from scm.active_learning.mlip.splitting_strategy import DataAndSplittingStrategy
from scm.active_learning.results import MLIPTrainerResults


def _train(overrides: list[str]):
    """train schnetpack

    :param overrides: same overrides you'd pass on the CLI,
        e.g.: ["run.data_dir=/path/to/data","experiment=qm9_atomwise"]
    :type overrides: List[str]
    """
    from importlib.resources import files

    compose, initialize_config_dir, train = _load_schnetpack_training_deps()

    # Locate schnetpack's packaged config folder (the one referenced by config_path="configs")
    config_dir = files("schnetpack").joinpath("configs")

    # Compose config (optionally with overrides), then call the original function
    with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
        cfg = compose(
            config_name="train",
            overrides=overrides,
        )
        print(cfg)

    train.__wrapped__(cfg)  # <-- calls def train(config: DictConfig): ... :contentReference[oaicite:1]{index=1}


class SPKSettings(BaseModel):
    def collect_overrides(self) -> Dict[str, str]: ...


class DataSettings(SPKSettings):
    folder: Path = Path("train")

    def _get_ds_and_split(self) -> Tuple[Path, Path]:
        datapath = (self.folder / "dataset.db").expanduser()
        splitpath = datapath.with_name("split.npz")
        return datapath, splitpath

    def collect_overrides(self):
        overrides = {}
        datapath, splitpath = self._get_ds_and_split()
        overrides["data.datapath"] = f"data.datapath='{datapath}'"
        overrides["data.split_file"] = f"+data.split_file='{splitpath}'"
        return overrides


class RunSettings(SPKSettings):
    folder: Path = Path("train")
    work_dir_name: str = "run1"

    def collect_overrides(self):
        overrides = {}
        overrides["run.data_dir"] = f"run.data_dir='{self.folder}'"
        overrides["run.work_dir"] = f"run.work_dir='{self.folder / self.work_dir_name}'"
        return overrides


class SchNetPackTrainer(MLIPTrainer[ASEEngine]):
    type: Literal["SchNetPackTrainer"] = "SchNetPackTrainer"
    data: DataSettings = DataSettings()
    run: RunSettings = RunSettings()
    overrides: Dict[str, str] = {}
    separate_env: Literal[False] = False

    @property
    def ml_name(self) -> str:
        return "SchNetPack"

    def validate_engine(
        self,
        engine: ConcreteEngines,
    ) -> ASEEngine:
        if isinstance(engine, ASEEngine):
            return engine
        raise TypeError(f"{type(engine)=} is not ASEEngine")

    def _is_installed(self):
        error_msg = "schnetpack is required for SchNetPackTrainer, but the installation went wrong:\n"
        installation_error = _check_installation()
        if installation_error != "":
            raise ModuleNotFoundError(error_msg + installation_error)

    def train(self, data_split: DataAndSplittingStrategy, engine_id: str, **kwargs) -> MLIPTrainerResults:
        if kwargs:
            raise ValueError("SchNetPackTrainer configuration must be provided at initialization, not in train().")
        return self._run_training(data_split=data_split, engine_id=engine_id, base_model_file=None)

    def finetune(
        self,
        engine: ASEEngine,
        data_split: DataAndSplittingStrategy,
        engine_id: str,
        **kwargs,
    ) -> MLIPTrainerResults:
        if kwargs:
            raise ValueError("SchNetPackTrainer configuration must be provided at initialization, not in finetune().")
        validated = self.validate_engine(engine)
        # TODO implement _extract_model_from_ase_engine
        base_model_file = self._extract_model_from_ase_engine(validated)
        return self._run_training(
            data_split=data_split,
            engine_id=engine_id,
            base_model_file=base_model_file,
        )

    def to_data_format(
        self,
        data_split: DataAndSplittingStrategy,
    ) -> Path:
        if len(data_split.dataset) == 0:
            raise ValueError("Dataset is empty, cannot train SchNetPack.")
        datapath, splitpath = self.data._get_ds_and_split()
        datapath.parent.mkdir(parents=True, exist_ok=True)
        if datapath.exists():
            datapath.unlink()
        if splitpath.exists():
            splitpath.unlink()

        dataset_writer = ASEMolData.create(
            data_source=str(datapath),
            available_properties=data_split.dataset.out_properties,
            distance_unit=data_split.dataset.distance_unit,
        )
        md = {
            "_distance_unit": dataset_writer.distance_unit,
            "_property_unit_dict": {p.name: p.unit for p in data_split.dataset.out_properties},
        }
        dataset_writer.update_metadata(**md)
        dataset_writer.add_systems(data_split.dataset, disable=True)
        train_idx, val_idx, test_idx = self._build_split_indices(data_split)
        np.savez(splitpath, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        return datapath.parent

    @staticmethod
    def _build_split_indices(
        data_split: DataAndSplittingStrategy,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []

        for idx, row in enumerate(data_split.dataset):
            split = row.metadata.get(data_split.split_key, data_split.training_key)
            if split == data_split.training_key:
                train_indices.append(idx)
            elif split == data_split.validation_key:
                val_indices.append(idx)
            elif split in {"test", "test_set", "testing_set"}:
                test_indices.append(idx)
            else:
                raise ValueError(
                    f"Unknown split label {split!r} found in metadata key {data_split.split_key!r} for index {idx}."
                )

        return (
            np.array(train_indices, dtype=np.int64),
            np.array(val_indices, dtype=np.int64),
            np.array(test_indices, dtype=np.int64),
        )

    def _run_training(
        self,
        data_split: DataAndSplittingStrategy,
        engine_id: str,
        base_model_file: Optional[str],
    ) -> MLIPTrainerResults:
        self.to_data_format(data_split=data_split)
        overrides_dict = self.overrides.copy()
        overrides_dict.update(self.data.collect_overrides())
        overrides_dict.update(self.run.collect_overrides())

        if base_model_file is not None:
            raise NotImplementedError(base_model_file)
        _train(overrides=list(overrides_dict.values()))

        model_path = "Find it"
        metrics = {}
        return MLIPTrainerResults(
            engine=ASEEngine(
                engine_id=engine_id,
                calculator="",
                calculator_kwargs={"model_file": model_path},
            ),
            log_metrics=metrics,
        )
