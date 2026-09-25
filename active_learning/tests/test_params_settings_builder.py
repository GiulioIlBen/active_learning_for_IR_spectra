from __future__ import annotations

import pytest

from scm.active_learning.mlip.params_trainer import settings_builder as psb


def test_trainer_builder_sets_values_and_build_settings():
    builder = psb.ParAMSTrainerBuilder()
    builder.set_committee(committee_size=3, parallel=2)
    builder.set_max_epochs(50)
    builder.set_loss(energy=1.2, forces=3.4)
    builder.set_random_seed(123)

    settings = builder.build_settings()

    assert settings is not builder.settings
    assert settings.input.MachineLearning.CommitteeSize == 3
    assert settings.input.ParallelLevels.CommitteeMembers == 2
    assert settings.input.MachineLearning.MaxEpochs == 50
    assert settings.input.MachineLearning.LossCoeffs.Energy == 1.2
    assert settings.input.MachineLearning.LossCoeffs.Forces == 3.4
    assert settings.input.RandomSeed == 123


def test_fieldschnet_fitjob_cpu_settings():
    settings = psb._fieldschnet_fitjob_settings(cpu_num_nodes=4, cutoff=6.0, lr_initial=1e-4)
    train = settings.input.hparams_training

    assert settings.input.hparams_architecture.cutoff == 6.0
    assert train.lr_initial == 1e-4
    assert train.accelerator == "cpu"
    assert train.strategy == "ddp"
    assert train.devices == 4
    assert train.num_nodes == 1


# def test_fieldschnet_driver_settings_contains_arguments():
#     settings = psb._fieldschnet_dipole_settings(
#         DipoleLoss=0.2,
#         cutoff=5.0,
#         lr_initial=1e-3,
#         batch_size=2,
#         n_interactions=2,
#         n_atom_basis=32,
#         cpu_num_nodes_each_committee=None,
#         add_remove_energy_mean=False,
#         remove_atomrefs="learnable",
#         max_atomic_number_atomrefs=10,
#     )

#     assert settings.input.MachineLearning.Backend == "FieldSchNet"
#     args = settings.input.MachineLearning.FieldSchNet.Arguments
#     assert "response_properties_and_loss" in args
#     assert "hparams_architecture" in args
#     assert "hparams_training" in args


# def test_mace_en_forces_settings_sets_backend_and_args():
#     settings = psb._mace_en_forces_settings(
#         patience=1,
#         start_swa=2,
#         atomic_numbers=[1, 8],
#         batch_size=4,
#     )

#     assert settings.input.MachineLearning.Backend == "MACE"
#     assert "settings" in settings.input.MachineLearning.MACE.Arguments


def test_builder_test_preset_sets_expected_defaults():
    pytest.importorskip("scm.input_classes", reason="requires optional dependency scm.input_classes")
    builder = psb.MLIPParAMSSettingsBuilder.TEST()

    assert builder.preset == "TEST"
    assert builder.backend == "TEST"
    assert builder.task == "EF"
    assert builder.settings.input.MachineLearning.Backend == "Test"
    assert builder.settings.input.MachineLearning.MaxEpochs == 1000
    assert builder.settings.input.MachineLearning.LossCoeffs.Energy == 1.0
    assert builder.settings.input.MachineLearning.LossCoeffs.Forces == 100.0
