from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Sequence, Union

import scm.plams as plams

from scm.active_learning.mlip.params_trainer._optional_dependencies import _load_input_classes_drivers

if TYPE_CHECKING:
    from scm.input_classes.drivers import ParAMS as ParAMSDriver

PresetName = Literal["MACE_EFD", "MACE_EF", "FieldSchNet_EFD", "M3GNet_EF", "TEST"]

# -----------------------------------------------------------------------------
# Step 2: Fluent builder that accumulates Settings and finally returns plams.Settings
# -----------------------------------------------------------------------------


@dataclass
class ParAMSTrainerBuilder:
    """
    Fluent builder that composes plams.Settings and returns them with build().

    This assumes the preset created a ParAMS MachineLearning input under settings.input.
    """

    settings: plams.Settings = field(default_factory=plams.Settings)
    preset: Optional[PresetName] = None
    backend: Optional[str] = None
    task: Optional[Literal["EF", "EFD"]] = None  # "EF" or "EFD"

    def __post_init__(self) -> None:
        self._ensure_base_driver()

    def add_settings(self, other: plams.Settings) -> ParAMSTrainerBuilder:
        self.settings += other
        return self

    def _ensure_base_driver(self) -> None:
        if "input" not in self.settings:
            self.settings.input = plams.Settings()
        if "Task" not in self.settings.input:
            self.settings.input.Task = "MachineLearning"
        if "MachineLearning" not in self.settings.input:
            self.settings.input.MachineLearning = plams.Settings()
        if "RunAMSAtEnd" not in self.settings.input.MachineLearning:
            self.settings.input.MachineLearning.RunAMSAtEnd = True
        if "ParallelLevels" not in self.settings.input:
            self.settings.input.ParallelLevels = plams.Settings()

    def patch(self, fn: Callable[[plams.Settings], None]) -> ParAMSTrainerBuilder:
        fn(self.settings)
        return self

    def set_committee(self, committee_size: int, parallel: Optional[int] = None) -> ParAMSTrainerBuilder:
        self._ensure_base_driver()
        if parallel is None:
            parallel = committee_size
        self.settings.input.ParallelLevels.CommitteeMembers = parallel
        self.settings.input.MachineLearning.CommitteeSize = committee_size
        return self

    def set_max_epochs(self, max_epochs: int) -> ParAMSTrainerBuilder:
        self._ensure_base_driver()
        self.settings.input.MachineLearning.MaxEpochs = max_epochs
        return self

    def set_loss(self, energy: Optional[float] = None, forces: Optional[float] = None) -> ParAMSTrainerBuilder:
        self._ensure_base_driver()
        lc = self.settings.input.MachineLearning.LossCoeffs
        if energy is not None:
            lc.Energy = energy
        if forces is not None:
            lc.Forces = forces
        return self

    def set_random_seed(self, random_seed: int) -> ParAMSTrainerBuilder:
        self._ensure_base_driver()
        self.settings.input.RandomSeed = random_seed
        return self

    def set_target_forces(self, mae_eV_Ang: float | None = 0.05) -> ParAMSTrainerBuilder:
        self._ensure_base_driver()
        if mae_eV_Ang is None:
            self.settings.input.MachineLearning.Target.Forces.Enabled = "No"
        else:
            self.settings.input.MachineLearning.Target.Forces.Enabled = "Yes"
            self.settings.input.MachineLearning.Target.Forces.MAE = mae_eV_Ang
        return self

    def set_batch_size(self, batch: int = 5) -> ParAMSTrainerBuilder:
        self._ensure_base_driver()
        if "dataset" not in self.settings.input:
            self.settings.input.dataset = [plams.Settings()]
        if len(self.settings.input.dataset) == 0:
            raise ValueError(f"{self.settings.input.dataset=}")
        self.settings.input.dataset[0].BatchSize = batch
        return self

    def build_settings(self) -> plams.Settings:
        out = plams.Settings()
        out += self.settings
        return out

    def build(self, datapath: str = "params_data"):
        from scm.active_learning.mlip.params_trainer import ParAMSTrainer

        return ParAMSTrainer(
            datapath=datapath,
            source_settings=self.build_settings().as_dict(),
        )


# -----------------------------------------------------------------------------
# Shared helpers (no dependency on scm.params.machine_learning.tests.*)
# -----------------------------------------------------------------------------


def _driver_to_settings(d: "ParAMSDriver") -> plams.Settings:
    s = plams.Settings()
    s.input = d.to_settings()
    return s


def _new_params_driver() -> "ParAMSDriver":
    drivers = _load_input_classes_drivers()
    return drivers.ParAMS()


# -----------------------------------------------------------------------------
# FieldSchNet implementation (ported from your second file)
# -----------------------------------------------------------------------------


def _fieldschnet_fitjob_settings(
    *,
    add_remove_energy_mean: bool = False,
    remove_atomrefs: Union[bool, Path, Dict[str, float], Literal["estimate", "learnable"]] = "learnable",
    max_atomic_number_atomrefs: int = 100,
    cutoff: float = 5,
    lr_initial: float = 1e-3,
    batch_size: int = 3,
    n_interactions: int = 2,
    n_atom_basis: int = 50,
    cpu_num_nodes: Optional[int] = None,
) -> plams.Settings:
    """
    Produces a Settings() with fieldschnet fitjob base_settings under settings.input.
    This matches the structure you were using via:
        base_settings = settings.input
        base_settings.hparams_architecture...
        base_settings.hparams_training...
    """
    s = plams.Settings()
    base = s.input

    arch = base.hparams_architecture
    train = base.hparams_training

    arch.n_atom_basis = n_atom_basis
    arch.n_rbf = 32
    arch.n_interactions = n_interactions
    arch.cutoff = cutoff

    # atomrefs
    arch.add_remove_energy_mean = add_remove_energy_mean  # type: ignore
    arch.remove_atomrefs = remove_atomrefs  # type: ignore
    arch.max_atomic_number_atomrefs = max_atomic_number_atomrefs  # type: ignore
    arch.initialize_with_estimate_atomsrefs = True  # type: ignore
    arch.atomrefs_callbacks_debug = False  # type: ignore
    arch.make_learnable_atoms_refs_learnable = True  # type: ignore

    # training
    train.lr_initial = lr_initial
    train.patience_lr_scheduler = 50
    train.batch_size = batch_size
    train.log_every_n_steps = 100
    train.optimizer = "AdamW"  # type: ignore
    train.accelerator = "auto"
    train.strategy = None  # type: ignore
    train.num_nodes = 1
    train.devices = None

    # parallel cpu
    if cpu_num_nodes is not None:
        train.accelerator = "cpu"
        train.strategy = "ddp"  # type: ignore
        train.num_nodes = 1
        train.devices = cpu_num_nodes

    # early stop / targets / schedulers
    train.early_stop_target_on = True
    train.target_patience = 1
    train.target_energy_mae = 0.1
    train.target_forces_mae = 0.1
    train.loss_weights_scheduler_on = True
    train.loss_weights_reduction = 100
    train.loss_weights_scheduler_energy_target = 100
    train.loss_weights_scheduler_forces_target = None
    train.loss_weights_scheduler_dipole_moment_target = None
    train.loss_weights_scheduler_relative_change = False
    train.tb_logger = False

    return s


def _fieldschnet_driver_from_fitjob_base(
    *,
    base_settings: plams.Settings,
) -> "ParAMSDriver":
    """
    Build ParAMS driver for FieldSchNet using base_settings.input.* as the fitjob config.
    """
    d = _new_params_driver()
    d.Task = "MachineLearning"
    d.MachineLearning.RunAMSAtEnd = True
    d.MachineLearning.Backend = "FieldSchNet"

    # base_settings here is the 'fitjob settings object', we want its .input
    base = base_settings.input
    d.MachineLearning.FieldSchNet.Arguments = f"""
        response_properties_and_loss = {str(base.response_properties_and_loss.as_dict())}
        hparams_architecture = {str(base.hparams_architecture.as_dict())}
        hparams_training = {str(base.hparams_training.as_dict())}
    """
    return d


def _fieldschnet_dipole_settings(
    DipoleLoss: float,
    cutoff: float,
    lr_initial: float,
    batch_size: int,
    n_interactions: int,
    n_atom_basis: int,
    cpu_num_nodes_each_committee: Optional[int],
    add_remove_energy_mean: bool,
    remove_atomrefs: Union[bool, Path, Dict[str, float], Literal["estimate", "learnable"]],
    max_atomic_number_atomrefs: int,
) -> plams.Settings:
    # Build base settings with response_properties_and_loss set
    s = plams.Settings()
    base = s.input
    base.response_properties_and_loss = {"dipole_moment": DipoleLoss}

    # match your tweaks to training settings
    train = base.hparams_training
    train.loss_weights_scheduler_on = True
    train.loss_weights_scheduler_dipole_moment_target = None

    # merge in the generic fitjob defaults
    s_fit = _fieldschnet_fitjob_settings(
        add_remove_energy_mean=add_remove_energy_mean,
        remove_atomrefs=remove_atomrefs,
        max_atomic_number_atomrefs=max_atomic_number_atomrefs,
        cutoff=cutoff,
        lr_initial=lr_initial,
        batch_size=batch_size,
        n_interactions=n_interactions,
        n_atom_basis=n_atom_basis,
        cpu_num_nodes=cpu_num_nodes_each_committee,
    )
    s += s_fit

    d = _fieldschnet_driver_from_fitjob_base(
        base_settings=s,
    )
    return _driver_to_settings(d)


# -----------------------------------------------------------------------------
# M3GNet implementation (ported from your second file)
# -----------------------------------------------------------------------------


def _m3gnet_en_forces_settings(
    Model: Literal["UniversalPotential", "Custom", "ModelDir"],
    lr: float | None = 0.001,
) -> plams.Settings:
    d = _new_params_driver()
    d.Task = "MachineLearning"
    d.MachineLearning.RunAMSAtEnd = True
    d.MachineLearning.Backend = "M3GNet"
    d.MachineLearning.M3GNet.Model = Model
    if lr:
        d.MachineLearning.M3GNet.LearningRate = lr
    return _driver_to_settings(d)


# -----------------------------------------------------------------------------
# MACE implementation (ported from your second file)
# -----------------------------------------------------------------------------


def _mace_fitjob_en_forces(
    patience: int = 100,
    start_swa: int = 20,
    atomic_numbers: Optional[Sequence[int]] = None,
    num_channels: int = 64,
    batch_size: int = 10,
    learning_rate: float = 0.01,
) -> plams.Settings:
    s = plams.Settings()
    base = s.input

    arch = base.hparams_architecture
    arch.num_interactions = 2
    arch.max_ell = 3
    arch.num_channels = num_channels
    arch.r_max = 4
    if isinstance(atomic_numbers, (list, tuple)):
        arch.atomic_numbers = list(atomic_numbers)
    arch.model = "MACE"  # type: ignore

    train = base.hparams_training
    train.devices = "cuda"
    train.eval_interval = 5
    train.learning_rate = learning_rate
    train.swa = True
    train.start_swa = start_swa
    train.patience = patience
    train.batch_size = batch_size
    return s


def _mace_fitjob_dipole(DipoleLoss: float = 1.0) -> plams.Settings:
    s = plams.Settings()
    base = s.input

    arch = base.hparams_architecture
    arch.model = "EnergyDipolesMACE"  # type: ignore

    train = base.hparams_training
    train.loss = "energy_forces_dipole"
    train.error_table = "EnergyDipoleRMSE"
    train.dipole_weight = DipoleLoss
    return s


def _mace_params_settings_from_fitjob(
    fitjob_settings: plams.Settings,
) -> plams.Settings:
    d = _new_params_driver()
    d.Task = "MachineLearning"
    d.MachineLearning.RunAMSAtEnd = True
    d.MachineLearning.Backend = "MACE"
    d.MachineLearning.MACE.Arguments = f"settings = {str(fitjob_settings.as_dict())}"
    return _driver_to_settings(d)


def _mace_en_forces_settings(
    patience: int,
    start_swa: int,
    atomic_numbers: Optional[Sequence[int]],
    batch_size: int,
) -> plams.Settings:
    fit = _mace_fitjob_en_forces(
        patience=patience,
        start_swa=start_swa,
        atomic_numbers=atomic_numbers,
        batch_size=batch_size,
    )
    return _mace_params_settings_from_fitjob(
        fitjob_settings=fit,
    )


def _mace_efd_settings(
    patience: int,
    start_swa: int,
    DipoleLoss: float,
    atomic_numbers: Optional[Sequence[int]],
    num_channels: int,
    batch_size: int,
) -> plams.Settings:
    fit = _mace_fitjob_dipole(DipoleLoss=DipoleLoss)
    fit += _mace_fitjob_en_forces(
        patience=patience,
        start_swa=start_swa,
        atomic_numbers=atomic_numbers,
        num_channels=num_channels,
        batch_size=batch_size,
    )
    return _mace_params_settings_from_fitjob(
        fitjob_settings=fit,
    )


# -----------------------------------------------------------------------------
# Step 1: Preset selector (your requested API)
# -----------------------------------------------------------------------------


class MLIPParAMSSettingsBuilder:
    """
    Two-step builder:
      1) choose preset -> returns ParAMSTrainerBuilder
      2) configure -> .build() returns plams.Settings

    No dependency on scm.params.machine_learning.tests.test_params_all.
    """

    @staticmethod
    def MACE_EFD(
        atomic_numbers: Optional[Sequence[int]] = None,
        EnergyLoss: float = 5.0,
        ForcesLoss: float = 5.0,
        DipoleLoss: float = 1.0,
        patience: int = 200,
        start_swa: int = 20,
        num_channels: int = 64,
        batch_size: int = 10,
    ) -> ParAMSTrainerBuilder:
        s = _mace_efd_settings(
            patience=patience,
            start_swa=start_swa,
            DipoleLoss=DipoleLoss,
            atomic_numbers=atomic_numbers,
            num_channels=num_channels,
            batch_size=batch_size,
        )
        return (
            ParAMSTrainerBuilder(settings=s, preset="MACE_EFD", backend="MACE", task="EFD")
            .set_max_epochs(1000)
            .set_loss(energy=EnergyLoss, forces=ForcesLoss)
        )

    @staticmethod
    def MACE_EF(
        *,
        atomic_numbers: Optional[Sequence[int]] = None,
        MaxEpochs: int = 1000,
        CommitteeSize: int = 2,
        Parallel: Optional[int] = None,
        EnergyLoss: float = 5.0,
        ForcesLoss: float = 5.0,
        RandomSeed: int = 94030,
        patience: int = 200,
        start_swa: int = 20,
        batch_size: int = 10,
    ) -> ParAMSTrainerBuilder:
        if Parallel is None:
            Parallel = CommitteeSize
        s = _mace_en_forces_settings(
            patience=patience,
            start_swa=start_swa,
            atomic_numbers=atomic_numbers,
            batch_size=batch_size,
        )
        return (
            ParAMSTrainerBuilder(settings=s, preset="MACE_EF", backend="MACE", task="EF")
            .set_committee(committee_size=CommitteeSize, parallel=Parallel)
            .set_max_epochs(MaxEpochs)
            .set_loss(energy=EnergyLoss, forces=ForcesLoss)
            .set_random_seed(RandomSeed)
        )

    @staticmethod
    def FieldSchNet_EFD(
        *,
        DipoleLoss: float = 0.2,
        cutoff: float = 5,
        lr_initial: float = 1e-3,
        batch_size: int = 3,
        n_interactions: int = 3,
        n_atom_basis: int = 50,
        cpu_num_nodes_each_committee: Optional[int] = None,
        add_remove_energy_mean: bool = False,
        remove_atomrefs: Union[bool, Path, Dict[str, float], Literal["estimate", "learnable"]] = "learnable",
        max_atomic_number_atomrefs: int = 100,
    ) -> ParAMSTrainerBuilder:
        s = _fieldschnet_dipole_settings(
            DipoleLoss=DipoleLoss,
            cutoff=cutoff,
            lr_initial=lr_initial,
            batch_size=batch_size,
            n_interactions=n_interactions,
            n_atom_basis=n_atom_basis,
            cpu_num_nodes_each_committee=cpu_num_nodes_each_committee,
            add_remove_energy_mean=add_remove_energy_mean,
            remove_atomrefs=remove_atomrefs,
            max_atomic_number_atomrefs=max_atomic_number_atomrefs,
        )
        return (
            ParAMSTrainerBuilder(settings=s, preset="FieldSchNet_EFD", backend="FieldSchNet", task="EFD")
            .set_max_epochs(1000)
            .set_loss(energy=5.0, forces=5.0)
        )

    @staticmethod
    def M3GNet_EF(
        Model: Literal["UniversalPotential", "Custom", "ModelDir"] = "UniversalPotential",
        learning_rate: float | None = 0.001,
    ) -> ParAMSTrainerBuilder:
        s = _m3gnet_en_forces_settings(
            Model=Model,
            lr=learning_rate,
        )
        return ParAMSTrainerBuilder(settings=s, preset="M3GNet_EF", backend="M3GNet", task="EF").set_max_epochs(1000)

    @staticmethod
    def TEST() -> ParAMSTrainerBuilder:
        d = _new_params_driver()
        d.Task = "MachineLearning"
        d.MachineLearning.RunAMSAtEnd = True
        d.MachineLearning.Backend = "Test"
        return (
            ParAMSTrainerBuilder(
                settings=_driver_to_settings(d),
                preset="TEST",
                backend="TEST",
                task="EF",
            )
            .set_max_epochs(1000)
            .set_loss(energy=1.0, forces=100.0)
        )


# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    atomic_numbers = [1, 6, 7, 8]

    settings = (
        MLIPParAMSSettingsBuilder.MACE_EFD(
            atomic_numbers=atomic_numbers,
            DipoleLoss=0.5,
        )
        .set_committee(committee_size=3)
        .set_max_epochs(600)
        .set_loss(energy=2.0, forces=10.0)
        .build_settings()
    )

    # settings is plams.Settings with ParAMS ML driver input under settings.input
    # settings.input.print_tree()  # if you have such helper
    print(settings)
