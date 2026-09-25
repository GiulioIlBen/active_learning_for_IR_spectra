from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, ClassVar, Dict, FrozenSet, Iterable, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scm.moliterate.core.chem_data_entry import ChemDataEntry

from scm.active_learning.mlip.core import MLIPTrainerSaveSettings
from scm.active_learning.mlip.mace_trainer._optional_dependencies import _load_ase_dependencies
from scm.active_learning.mlip.splitting_strategy import DataAndSplittingStrategy

MACEDevice = Literal["cpu", "cuda", "mps", "xpu"]
MACELogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
MACEErrorTable = Literal[
    "PerAtomRMSE",
    "TotalRMSE",
    "PerAtomRMSEstressvirials",
    "PerAtomMAEstressvirials",
    "PerAtomMAE",
    "TotalMAE",
    "DipoleRMSE",
    "DipoleMAE",
    "DipolePolarRMSE",
    "EnergyDipoleRMSE",
]
MACELoss = Literal[
    "ef",
    "weighted",
    "forces_only",
    "virials",
    "stress",
    "dipole",
    "dipole_polar",
    "huber",
    "universal",
    "energy_forces_dipole",
    "l1l2energyforces",
]
MACEModel = Literal[
    "BOTNet",
    "MACE",
    "ScaleShiftMACE",
    "MACELES",
    "ScaleShiftBOTNet",
    "AtomicDipolesMACE",
    "AtomicDielectricMACE",
    "EnergyDipolesMACE",
]


class MACETrainerSaveSettings(MLIPTrainerSaveSettings):
    settings_file_name: str = "mace_trainer_settings.json"


class MACEDataSettings(BaseModel):
    folder: Path = Path("mace_training")
    dataset_subfolder: str = "datasets"
    training_file_name: str = "training_set.xyz"
    validation_file_name: str = "validation_set.xyz"
    config_file_name: str = "config_file_path.yaml"
    model_subfolder: str = "results"

    def dataset_dir(self, run_dir: Path) -> Path:
        return run_dir / self.dataset_subfolder

    def train_file(self, run_dir: Path) -> Path:
        return self.dataset_dir(run_dir) / self.training_file_name

    def valid_file(self, run_dir: Path) -> Path:
        return self.dataset_dir(run_dir) / self.validation_file_name

    def config_path(self, run_dir: Path) -> Path:
        return run_dir / self.config_file_name

    def model_dir(self, run_dir: Path) -> Path:
        return run_dir / self.model_subfolder

    def prepare_training_context(
        self,
        data_split: DataAndSplittingStrategy,
        run_dir: Path,
    ) -> MACEPreparedTrainingContext:
        context, _, _ = self._collect_training_data(data_split=data_split, run_dir=run_dir)
        return context

    def export_dataset(
        self,
        data_split: DataAndSplittingStrategy,
        run_dir: Path,
    ) -> Path:
        context, train_rows, valid_rows = self._collect_training_data(data_split=data_split, run_dir=run_dir)
        self._write_dataset_files(context=context, train_rows=train_rows, valid_rows=valid_rows)
        return context.dataset_dir

    def _collect_training_data(
        self,
        data_split: DataAndSplittingStrategy,
        run_dir: Path,
    ) -> Tuple[MACEPreparedTrainingContext, list[ChemDataEntry], list[ChemDataEntry]]:
        if len(data_split.dataset) == 0:
            raise ValueError("Dataset is empty, cannot train MACE.")

        dataset_dir = self.dataset_dir(run_dir)
        train_file = self.train_file(run_dir)
        valid_file = self.valid_file(run_dir)
        train_rows, valid_rows = self._split_rows(data_split)
        if len(train_rows) == 0:
            raise ValueError("No training entries found in dataset split.")

        context = MACEPreparedTrainingContext(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            train_file=train_file,
            valid_file=valid_file,
            training_set_size=len(train_rows),
            validation_set_size=len(valid_rows),
            available_property_names=frozenset(
                self._export_property_name(prop.name) for prop in data_split.dataset.out_properties
            ),
            dataset_atomic_numbers=frozenset(
                int(atomic_number) for row in (*train_rows, *valid_rows) for atomic_number in row.atoms.numbers
            ),
        )
        return context, train_rows, valid_rows

    def _write_dataset_files(
        self,
        context: MACEPreparedTrainingContext,
        train_rows: Iterable[ChemDataEntry],
        valid_rows: Iterable[ChemDataEntry],
    ) -> None:
        context.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._write_xyz(context.train_file, train_rows)
        valid_rows_list = list(valid_rows)
        if len(valid_rows_list) > 0:
            self._write_xyz(context.valid_file, valid_rows_list)
        elif context.valid_file.exists():
            context.valid_file.unlink()

    @classmethod
    def _write_xyz(cls, path: Path, rows: Iterable[ChemDataEntry]) -> None:
        ase_io = _load_ase_dependencies()
        atoms_collection = []
        for row in rows:
            atoms = row.atoms.copy()
            atoms.info.update(cls._sanitize_extxyz_metadata(row.metadata))
            cls._attach_exported_properties(atoms=atoms, properties=row.properties)
            atoms_collection.append(atoms)
        if len(atoms_collection) == 0:
            raise ValueError(f"No rows to export into dataset file: {path}")
        ase_io.write(str(path), atoms_collection, format="extxyz")

    @classmethod
    def _export_properties(cls, properties: Dict[str, Any]) -> Dict[str, Any]:
        return {cls._export_property_name(key): value for key, value in properties.items()}

    @classmethod
    def _attach_exported_properties(cls, atoms: Any, properties: Dict[str, Any]) -> None:
        for key, value in cls._export_properties(properties).items():
            if value is None:
                continue
            array_value = np.asarray(value)
            if array_value.ndim >= 1 and len(array_value) == len(atoms):
                atoms.arrays[key] = array_value
                continue
            atoms.info[key] = value

    @staticmethod
    def _export_property_name(name: str) -> str:
        legacy_to_safe = {
            "energy": "REF_energy",
            "forces": "REF_forces",
            "stress": "REF_stress",
            "dipole": "REF_dipole",
            "dipole_moment": "REF_dipole",
        }
        return legacy_to_safe.get(name, name)

    @staticmethod
    def _sanitize_extxyz_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, str):
                compact = " ".join(value.split())
                if compact:
                    sanitized[key] = compact
                continue
            if isinstance(value, (int, float, bool)):
                sanitized[key] = value
        return sanitized

    def _split_rows(self, data_split: DataAndSplittingStrategy) -> Tuple[list[ChemDataEntry], list[ChemDataEntry]]:
        train_rows: list[ChemDataEntry] = []
        valid_rows: list[ChemDataEntry] = []
        for idx, row in enumerate(data_split.dataset):
            split_label = self._resolve_split_label(row.metadata.get(data_split.split_key), data_split, row_idx=idx)
            if split_label == data_split.training_key:
                train_rows.append(row)
            elif split_label == data_split.validation_key:
                valid_rows.append(row)
            elif split_label in {"test", "test_set", "testing_set"}:
                continue
            else:
                raise ValueError(
                    f"Unknown split label {split_label!r} found in metadata "
                    f"key {data_split.split_key!r} at index {idx}."
                )
        return train_rows, valid_rows

    @staticmethod
    def _resolve_split_label(
        raw_label: Any,
        data_split: DataAndSplittingStrategy,
        row_idx: int,
    ) -> str:
        if raw_label is None:
            return data_split.training_key
        if isinstance(raw_label, str):
            return raw_label
        if isinstance(raw_label, dict):
            labels = {value for value in raw_label.values() if isinstance(value, str)}
            if len(labels) == 0:
                return data_split.training_key
            if len(labels) == 1:
                return next(iter(labels))
            raise ValueError(f"Row {row_idx} has inconsistent split mapping: {raw_label!r}")
        raise TypeError(f"Split label must be str or dict at row {row_idx}; got {type(raw_label)!r} ({raw_label!r}).")


class MACEModelSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    _ARCHITECTURE_KEYS: ClassVar[Tuple[str, ...]] = (
        "hidden_irreps",
        "num_channels",
        "max_L",
        "max_ell",
        "correlation",
        "num_interactions",
        "MLP_irreps",
        "radial_MLP",
        "gate",
        "pair_repulsion",
        "distance_transform",
        "num_radial_basis",
        "num_cutoff_basis",
        "radial_type",
        "interaction",
        "interaction_first",
        "r_max",
    )

    model: MACEModel = "MACE"
    loss: MACELoss = "weighted"
    error_table: MACEErrorTable = "PerAtomRMSE"
    dipole_weight: Annotated[float, Field(gt=0.0)] = 10.0
    forces_weight: Annotated[float, Field(gt=0.0)] = 100.0
    energy_weight: Annotated[float, Field(gt=0.0)] = 1.0
    hidden_irreps: str = "128x0e + 128x1o"
    max_ell: int = Field(default=3, ge=0)
    correlation: int = Field(default=3, gt=0)
    num_interactions: int = Field(default=2, gt=0)
    MLP_irreps: str = "16x0e"
    radial_MLP: Optional[List[int]] = None
    gate: Literal["silu", "tanh", "abs", "None"] = "silu"
    num_channels: Optional[int] = Field(default=None, gt=0)
    max_L: Optional[int] = Field(default=None, ge=0)
    pair_repulsion: Optional[bool] = None
    distance_transform: Optional[str] = None
    num_radial_basis: Optional[int] = Field(default=None, gt=0)
    num_cutoff_basis: Optional[int] = Field(default=None, gt=0)
    radial_type: Optional[str] = None
    interaction: Optional[str] = None
    interaction_first: Optional[str] = None
    r_max: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _set_model_specific_error_table_default(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("model") == "EnergyDipolesMACE":
            return {"error_table": "EnergyDipoleRMSE", **data}
        return data

    @model_validator(mode="after")
    def _validate_model_loss(self) -> "MACEModelSettings":
        if self.model == "EnergyDipolesMACE" and self.loss != "energy_forces_dipole":
            raise ValueError("model=EnergyDipolesMACE, so loss must be energy_forces_dipole")
        if self.model == "EnergyDipolesMACE" and self.error_table != "EnergyDipoleRMSE":
            raise ValueError("model=EnergyDipolesMACE, so error_table must be EnergyDipoleRMSE")
        return self

    def build_architecture_metadata(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata = self.model_dump(
            exclude={"loss", "error_table", "dipole_weight", "forces_weight", "energy_weight"},
            exclude_none=True,
        )
        if config is not None:
            for key in self._ARCHITECTURE_KEYS:
                if key in config:
                    metadata[key] = config[key]
        return metadata


class MACECommitteeSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    size: int = Field(default=1, gt=0)
    seed: Optional[int] = Field(default=None, ge=0)
    seed_stride: int = Field(default=1, gt=0)
    parallel_members: int = Field(
        default=1,
        gt=0,
        description=(
            "Number of committee members trained at the same time, each in its own process. "
            "1 trains them one after another in the current process."
        ),
    )
    gpu_ids: Optional[List[int]] = Field(
        default=None,
        description=(
            "GPU index per member, cycled (member i uses gpu_ids[i % len(gpu_ids)]) through CUDA_VISIBLE_DEVICES. "
            "Only used when parallel_members > 1; None keeps the environment's GPU selection."
        ),
    )

    _DEFAULT_BASE_SEED: ClassVar[int] = 123

    def member_seed(self, index: int) -> Optional[int]:
        if index < 0 or index >= self.size:
            raise IndexError(f"Committee member index {index} outside range 0..{self.size - 1}.")
        if self.size == 1 and self.seed is None:
            return None
        base_seed = self._DEFAULT_BASE_SEED if self.seed is None else self.seed
        return base_seed + index * self.seed_stride

    def member_seeds(self) -> List[Optional[int]]:
        return [self.member_seed(index) for index in range(self.size)]

    def member_gpu(self, index: int) -> Optional[int]:
        if not self.gpu_ids:
            return None
        return self.gpu_ids[index % len(self.gpu_ids)]

    @property
    def trains_in_parallel(self) -> bool:
        return self.parallel_members > 1 and self.size > 1


class MACETrainSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    max_num_epochs: int = Field(default=30, gt=0)
    batch_size: int = Field(default=10, gt=0)
    valid_batch_size: int = Field(default=10, gt=0)
    eval_interval: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=0.01, gt=0.0)
    patience: int = Field(default=256, gt=0)
    device: MACEDevice = "cpu"
    log_level: MACELogLevel = "INFO"
    default_dtype: Literal["float64", "float32"] = "float64"
    save_all_checkpoints: bool = False
    delete_checkpoints_after_training: bool = True
    valid_fraction: float = Field(default=0.1, ge=0.0, lt=1.0)
    model_name: str = Field(default="model", min_length=1)
    patch_load_foundations: bool = True
    swa: bool = True
    start_swa: int = Field(default=20, ge=0)
    swa_energy_weight: float = Field(default=10.0, gt=0.0)
    architecture: MACEModelSettings = Field(default_factory=MACEModelSettings)
    committee: MACECommitteeSettings = Field(default_factory=MACECommitteeSettings)

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model_name cannot be empty or whitespace only.")
        return stripped

    @model_validator(mode="after")
    def _validate_swa_settings(self) -> MACETrainSettings:
        if self.swa and self.start_swa >= self.max_num_epochs:
            raise ValueError("start_swa must be smaller than max_num_epochs when swa is enabled.")
        return self


class MACEPreparedTrainingContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_dir: Path
    dataset_dir: Path
    train_file: Path
    valid_file: Path
    training_set_size: int
    validation_set_size: int
    available_property_names: FrozenSet[str]
    dataset_atomic_numbers: FrozenSet[int]

    _RUN_ROOT_CONFIG_KEYS: ClassVar[Tuple[str, ...]] = (
        "log_dir",
        "results_dir",
        "checkpoints_dir",
        "downloads_dir",
        "wandb_dir",
    )

    @property
    def has_validation_data(self) -> bool:
        return self.validation_set_size > 0

    def inferred_validation_size(self, valid_fraction: float) -> int:
        if self.has_validation_data:
            return self.validation_set_size
        if self.training_set_size <= 0 or valid_fraction <= 0.0:
            return 0
        return int(valid_fraction * self.training_set_size)

    def effective_train_size(self, valid_fraction: float) -> int:
        if self.training_set_size <= 0:
            return 0
        if self.has_validation_data:
            return self.training_set_size
        return self.training_set_size - self.inferred_validation_size(valid_fraction)

    def resolve_train_batch_size(self, requested: int, valid_fraction: float) -> int:
        effective_train_size = self.effective_train_size(valid_fraction)
        if effective_train_size <= 0:
            raise ValueError("No training entries available for MACE after applying the validation split.")
        return min(requested, effective_train_size)

    def resolve_valid_batch_size(self, requested: int, valid_fraction: float) -> int:
        if self.has_validation_data:
            return min(requested, self.validation_set_size)
        inferred_valid_size = self.inferred_validation_size(valid_fraction)
        if inferred_valid_size > 0:
            return min(requested, inferred_valid_size)
        return requested

    def build_zero_e0s(self) -> str:
        if len(self.dataset_atomic_numbers) == 0:
            raise ValueError("Could not determine dataset atomic numbers to build zero E0s for MACE.")
        return str({atomic_number: 0.0 for atomic_number in sorted(self.dataset_atomic_numbers)})

    def default_avg_num_neighbors(self) -> float:
        return 1.0

    def resolve_run_root_config_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        resolved = dict(config)
        for key in self._RUN_ROOT_CONFIG_KEYS:
            value = resolved.get(key)
            if value is None:
                continue
            resolved[key] = str(self.resolve_run_root_path(value))
        return resolved

    def resolve_run_root_path(self, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.run_dir / path
        return path.resolve(strict=False)
