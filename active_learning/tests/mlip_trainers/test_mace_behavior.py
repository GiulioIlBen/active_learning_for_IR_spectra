from __future__ import annotations

from pathlib import Path

from scm.moliterate import PropertyInfo

from scm.active_learning.mlip.mace_trainer.trainer import MACETrainer

from .conformance import (
    MLIPTrainerCase,
    assert_engine_runs_in_accuracy_run,
    assert_engine_runs_md,
    assert_training_loss_shows_learning,
    get_ds,
    skip_if_missing_modules,
)

skip_if_missing_modules("mace", "e3nn", "torch")


ENERGY = PropertyInfo(name="energy", unit="eV")
FORCES = PropertyInfo(name="forces", unit="eV/Ang")
DIPOLE = PropertyInfo(name="dipole", unit="e*Ang")


def _make_mace_efd_trainer(root: Path) -> MACETrainer:
    trainer = MACETrainer()
    trainer.data.folder = root
    trainer.suppress_warnings = True
    trainer.train_settings.swa = False
    trainer.train_settings.max_num_epochs = 3
    trainer.train_settings.batch_size = 2
    trainer.train_settings.valid_batch_size = 2
    trainer.train_settings.learning_rate = 1.0e-2
    trainer.train_settings.device = "cpu"
    trainer.train_settings.default_dtype = "float32"
    trainer.train_settings.architecture.loss = "energy_forces_dipole"
    trainer.train_settings.architecture.error_table = "EnergyDipoleRMSE"
    trainer.train_settings.architecture.model = "EnergyDipolesMACE"
    trainer.train_settings.architecture.hidden_irreps = "8x0e + 8x1o"
    trainer.train_settings.architecture.MLP_irreps = "8x0e"
    trainer.train_settings.architecture.num_interactions = 1
    trainer.train_settings.architecture.correlation = 1
    trainer.train_settings.architecture.max_ell = 1
    trainer.train_settings.architecture.num_radial_basis = 4
    trainer.train_settings.architecture.num_cutoff_basis = 3
    return trainer


def test_mace_efd_trainer_learns_and_returns_engine_that_runs_accuracy_and_md(mlip_trainer_case_dir):
    case = MLIPTrainerCase(
        name="mace-efd",
        make_trainer=_make_mace_efd_trainer,
        training_properties=(ENERGY, FORCES, DIPOLE),
        accuracy_properties=(ENERGY, FORCES, DIPOLE),
        loss_metric_names=("loss",),
        engine_id="MACEEFDfast",
    )

    result = case.make_trainer(mlip_trainer_case_dir / case.name).train(
        case.make_data_split(), engine_id=case.engine_id
    )

    assert_training_loss_shows_learning(result, case.loss_metric_names)
    assert_engine_runs_in_accuracy_run(
        result,
        dataset=get_ds(case.accuracy_properties),
        properties=case.accuracy_properties,
        work_dir=mlip_trainer_case_dir,
    )
    assert_engine_runs_md(result, work_dir=mlip_trainer_case_dir)
