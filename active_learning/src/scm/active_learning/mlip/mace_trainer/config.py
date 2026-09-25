from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from scm.active_learning.mlip.mace_trainer.settings import MACEPreparedTrainingContext, MACETrainSettings
from scm.active_learning.mlip.splitting_strategy import DataAndSplittingStrategy


class MACEConfigBuilder(BaseModel):
    """Translate active-learning trainer settings into MACE runtime config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_settings: MACETrainSettings
    property_aliases: Dict[str, Tuple[str, ...]]
    extra_config: Dict[str, Any] = Field(default_factory=dict)

    def build(
        self,
        *,
        context: MACEPreparedTrainingContext,
        model_dir: Path,
        base_model_file: Optional[str],
        base_model_architecture: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        energy_key = self.resolve_property(
            context.available_property_names,
            self.property_aliases["energy"],
            required=True,
        )
        forces_key = self.resolve_property(
            context.available_property_names,
            self.property_aliases["forces"],
            required=True,
        )
        stress_key = self.resolve_property(
            context.available_property_names,
            self.property_aliases["stress"],
            required=False,
        )
        dipole_key = self.resolve_property(
            context.available_property_names,
            self.property_aliases["dipole"],
            required=False,
        )
        self.validate_dipole_configuration(dipole_key)

        config: Dict[str, Any] = {
            "name": self.train_settings.model_name,
            "model": self.train_settings.architecture.model,
            "train_file": str(context.train_file),
            "energy_key": energy_key,
            "forces_key": forces_key,
            "batch_size": context.resolve_train_batch_size(
                requested=self.train_settings.batch_size,
                valid_fraction=self.train_settings.valid_fraction,
            ),
            "valid_batch_size": context.resolve_valid_batch_size(
                requested=self.train_settings.valid_batch_size,
                valid_fraction=self.train_settings.valid_fraction,
            ),
            "max_num_epochs": self.train_settings.max_num_epochs,
            "eval_interval": self.train_settings.eval_interval,
            "lr": self.train_settings.learning_rate,
            "patience": self.train_settings.patience,
            "loss": self.train_settings.architecture.loss,
            "device": self.train_settings.device,
            "log_level": self.train_settings.log_level,
            "default_dtype": self.train_settings.default_dtype,
            "forces_weight": self.train_settings.architecture.forces_weight,
            "energy_weight": self.train_settings.architecture.energy_weight,
            "error_table": self.train_settings.architecture.error_table,
            "save_all_checkpoints": self.train_settings.save_all_checkpoints,
            "model_dir": str(model_dir),
            "results_dir": str(model_dir),
            "log_dir": str(context.run_dir / "logs"),
            "checkpoints_dir": str(context.run_dir / "checkpoints"),
            "downloads_dir": str(context.run_dir / "downloads"),
            "wandb_dir": str(context.run_dir / "wandb"),
            "E0s": context.build_zero_e0s(),
            "compute_avg_num_neighbors": False,
            "avg_num_neighbors": context.default_avg_num_neighbors(),
            "dipole_weight": self.train_settings.architecture.dipole_weight,
        }
        config.update(
            self.train_settings.architecture.model_dump(
                exclude={"model", "loss", "error_table", "dipole_weight", "forces_weight", "energy_weight"},
                exclude_none=True,
            )
        )

        if stress_key is not None:
            config["stress_key"] = stress_key
        if dipole_key is not None:
            config["dipole_key"] = dipole_key
        if context.validation_set_size > 0:
            config["valid_file"] = str(context.valid_file)
        else:
            config["valid_fraction"] = self.train_settings.valid_fraction
        if base_model_file is not None:
            config["foundation_model"] = str(Path(base_model_file).expanduser().resolve(strict=False))
        if base_model_architecture:
            config.update(base_model_architecture)
        if self.train_settings.swa:
            config["swa"] = True
            config["start_swa"] = self.train_settings.start_swa
            config["swa_energy_weight"] = self.train_settings.swa_energy_weight

        config.update(self.extra_config)
        if seed is not None:
            config["seed"] = seed
        return context.resolve_run_root_config_paths(config=config)

    def build_calculator_kwargs(
        self,
        *,
        model_path: str | list[str],
        data_split: DataAndSplittingStrategy,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_paths": model_path,
            "device": self.train_settings.device,
            "default_dtype": self.train_settings.default_dtype,
            "mace_model_architecture": self.train_settings.architecture.build_architecture_metadata(config=config),
        }
        model_type = self.calculator_model_type()
        if model_type != "MACE":
            kwargs["model_type"] = model_type
            kwargs["dipole_units"] = self.resolve_dipole_unit(data_split)
        return kwargs

    def validate_dipole_configuration(self, dipole_key: Optional[str]) -> None:
        architecture = self.train_settings.architecture
        if architecture.loss == "energy_forces_dipole" and dipole_key is None:
            raise ValueError("MACETrainer.loss='energy_forces_dipole' requires dipole labels in the dataset.")

        if architecture.model in {"AtomicDipolesMACE", "EnergyDipolesMACE"} and dipole_key is None:
            raise ValueError(f"MACETrainer model {architecture.model!r} requires dipole labels in the dataset.")

    def calculator_model_type(self) -> Literal["MACE", "DipoleMACE", "EnergyDipoleMACE"]:
        model = self.train_settings.architecture.model
        if model == "AtomicDipolesMACE":
            return "DipoleMACE"
        if model == "EnergyDipolesMACE":
            return "EnergyDipoleMACE"
        return "MACE"

    def resolve_dipole_unit(self, data_split: DataAndSplittingStrategy) -> str:
        for prop in data_split.dataset.out_properties:
            if prop.name in self.property_aliases["dipole"]:
                assert prop.unit is not None
                return prop.unit
        return "Debye"

    @staticmethod
    def resolve_property(
        available_property_names: frozenset[str],
        aliases: Tuple[str, ...],
        required: bool,
    ) -> Optional[str]:
        for alias in aliases:
            if alias in available_property_names:
                return alias
        if required:
            raise ValueError(f"Could not find required property for aliases {aliases!r}.")
        return None
