import builtins

import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core import ChemDataEntry, PropertyInfo
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.transforms import PropertyTransform
from scm.moliterate.visualize.property_distribution import (
    PropertyDistributionPlot,
    _import_matplotlib_pyplot,
    collect_property_values,
)


def test_collect_property_values_flattens_array_entries():
    dataset = InMemoryMolData.create(available_properties=[PropertyInfo(name="forces", unit="eV/Ang")])
    dataset.add_system(
        ChemDataEntry(
            system=Atoms(numbers=[1], positions=[[0, 0, 0]]),
            properties={"forces": np.array([[1.0, 2.0, 3.0]])},
        )
    )
    dataset.add_system(
        ChemDataEntry(
            system=Atoms(numbers=[1], positions=[[0, 0, 0.7]]),
            properties={"forces": np.array([[4.0, 5.0, np.nan]])},
        )
    )

    values = collect_property_values(dataset, "forces")

    assert np.array_equal(values, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_property_distribution_plot_none_uses_all_dataset_properties():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    dataset = InMemoryMolData.create(
        available_properties=[PropertyInfo(name="energy", unit="eV"), PropertyInfo(name="dipole", unit="Debye")]
    )
    dataset.add_system(
        ChemDataEntry(
            system=Atoms(numbers=[1], positions=[[0, 0, 0]]),
            properties={"energy": -1.0, "dipole": np.array([0.0, 0.0, 1.0])},
        )
    )
    dataset.add_system(
        ChemDataEntry(
            system=Atoms(numbers=[1], positions=[[0, 0, 0.7]]),
            properties={"energy": -2.0, "dipole": np.array([0.0, 1.0, 0.0])},
        )
    )
    dataset.add_system(
        ChemDataEntry(
            system=Atoms(numbers=[1], positions=[[0, 0, 1.4]]),
            properties={"energy": -1.5, "dipole": np.array([1.0, 0.0, 0.0])},
        )
    )

    fig, axes = PropertyDistributionPlot(bins=2).run(dataset)

    axes_flat = [ax for ax in axes.reshape(-1) if ax in fig.axes]
    assert len(axes_flat) == 2
    assert axes_flat[0].get_title() == "energy [eV]"
    assert axes_flat[1].get_title() == "dipole [Debye]"
    fig.canvas.draw()


def test_property_distribution_plot_accepts_strings_and_property_transforms():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    dataset = InMemoryMolData.create(
        available_properties=[PropertyInfo(name="energy", unit="eV"), PropertyInfo(name="forces", unit="eV/Ang")]
    )
    dataset.add_system(
        ChemDataEntry(
            system=Atoms(numbers=[1], positions=[[0, 0, 0]]),
            properties={"energy": -1.0, "forces": np.array([[1.0, 2.0, 2.0]])},
        )
    )
    dataset.add_system(
        ChemDataEntry(
            system=Atoms(numbers=[1], positions=[[0, 0, 0.7]]),
            properties={"energy": -2.0, "forces": np.array([[0.0, 3.0, 4.0]])},
        )
    )

    fig, axes = PropertyDistributionPlot(
        properties=[
            "energy",
            PropertyTransform(property_key="forces", post_process="l2_max", property_key_out="fmax"),
        ],
        bins=2,
    ).run(dataset)

    axes_flat = [ax for ax in axes.reshape(-1) if ax in fig.axes]
    assert len(axes_flat) == 2
    assert axes_flat[0].get_title() == "energy [eV]"
    assert axes_flat[1].get_title() == "fmax"
    assert len(axes_flat[0].patches) == 2
    assert len(axes_flat[1].patches) == 2
    fig.canvas.draw()


def test_import_matplotlib_pyplot_raises_helpful_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib":
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError, match="Install the visualization extras"):
        _import_matplotlib_pyplot()
