import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.analysis.pairwise_dataset_metrics import (
    PairwiseDatasetMetrics,
    PairwiseResult,
    PropMetricEvaluator,
    _NPMetricAccumulator,
)
from scm.moliterate.core import ChemDataEntry, PropertyInfo
from scm.moliterate.interfaces.in_memory import InMemoryMolData


def _make_dataset(energies, forces_list, systems):
    props_info = [
        PropertyInfo(name="energy", unit="eV", shape="float"),
        PropertyInfo(name="forces", unit="eV/Ang", shape=("nAtoms", 3)),
    ]
    dataset = InMemoryMolData.create(available_properties=props_info)
    rows = []
    for system, energy, forces in zip(systems, energies, forces_list):
        rows.append(ChemDataEntry(system=system, properties={"energy": energy, "forces": forces}))
    dataset.add_systems(rows)
    return dataset


@pytest.fixture
def ds_ref_cmp():
    systems = [
        Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        Atoms(numbers=[2], positions=[[0.0, 0.0, 0.0]]),
    ]
    ref = _make_dataset(
        energies=[1.0, 2.0],
        forces_list=[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ],
        systems=systems,
    )
    cmp = _make_dataset(
        energies=[1.2, 1.7],
        forces_list=[
            [[0.0, 0.0, 0.2], [1.1, 0.0, 0.0]],
            [[0.0, 0.0, 1.5]],
        ],
        systems=systems,
    )
    return ref, cmp


def test_impossible_states():
    acc = _NPMetricAccumulator()
    acc.update(np.array([1.0]))
    with pytest.raises(ValueError, match="Unsupported metric: nope"):
        acc.metric_value("nope")  # type: ignore[arg-type]


def test_np_metric_accumulator_max_scaled_abs():
    acc = _NPMetricAccumulator()
    acc.update(np.array([-0.3, 0.2]))
    acc.update(np.array([[0.0, -0.7]]))
    assert acc.metric_value("max_scaled_abs") == pytest.approx(0.7, rel=1e-7)


def test_prop_metric_evaluator_max_scaled_abs_scaling():
    evaluator = PropMetricEvaluator(property="energy", metric="max_scaled_abs")
    assert evaluator.transform_settings["max_scaled_abs"] == {"L": 3.0, "x0": 7.0, "k": 0.5}

    value_ref = np.array([1.0, 2.0], dtype=float)
    value_cmp = np.array([1.3, 1.7], dtype=float)
    expected_scaled = np.abs(value_cmp - value_ref) - (
        3.0 / (1.0 + np.exp(-0.5 * (np.abs(value_ref) - 7.0))) - 3.0 / (1.0 + np.exp(-0.5 * (-7.0)))
    )
    np.testing.assert_allclose(
        evaluator._np_property_diff(value_ref=value_ref, value_cmp=value_cmp, n_atoms=None),
        expected_scaled,
    )
    np.testing.assert_allclose(
        evaluator._np_property_diff(value_ref=value_ref, value_cmp=value_cmp, n_atoms=2),
        expected_scaled / 2.0,
    )


def test_pairwise_dataset_metrics(ds_ref_cmp):
    ref, cmp = ds_ref_cmp
    metrics = PairwiseDatasetMetrics.from_dataset(
        ref,
        metrics=["mae", "mse", "rmse", "max_abs", "mean_error"],
        per_n_atoms=["energy"],
    )
    results = metrics.compare_two(dataset_ref=ref, dataset_cmp=cmp)
    result_map = {(r.property, r.metric): r for r in results}
    assert len(result_map) == 10
    assert result_map[("energy", "mae")].per_n_atoms is True
    assert result_map[("forces", "mae")].per_n_atoms is False
    assert result_map[("energy", "mae")].n_entries == 2
    assert result_map[("forces", "mae")].n_entries == 9
    assert result_map[("energy", "mae")].success == "FAIL"
    assert result_map[("forces", "mae")].success == "FAIL"

    assert result_map[("energy", "mae")].value == pytest.approx(0.2, rel=1e-7)
    assert result_map[("energy", "mse")].value == pytest.approx(0.05, rel=1e-7)
    assert result_map[("energy", "rmse")].value == pytest.approx(0.2236067977, rel=1e-7)
    assert result_map[("energy", "max_abs")].value == pytest.approx(0.3, rel=1e-7)
    assert result_map[("energy", "mean_error")].value == pytest.approx(-0.1, rel=1e-7)

    assert result_map[("forces", "mae")].value == pytest.approx(0.8 / 9.0, rel=1e-7)
    assert result_map[("forces", "mse")].value == pytest.approx(0.3 / 9.0, rel=1e-7)
    assert result_map[("forces", "rmse")].value == pytest.approx((0.3 / 9.0) ** 0.5, rel=1e-7)
    assert result_map[("forces", "max_abs")].value == pytest.approx(0.5, rel=1e-7)
    assert result_map[("forces", "mean_error")].value == pytest.approx(0.8 / 9.0, rel=1e-7)


def test_atom_type_metrics_selects_requested_atomic_number(ds_ref_cmp):
    ref, cmp = ds_ref_cmp
    metrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", atom_type=1, components=-1),
        ],
    )
    results = metrics.compare_two(dataset_ref=ref, dataset_cmp=cmp)
    result_map = {(r.property, r.atom_type): r for r in results}
    assert set(result_map) == {("forces", 1)}
    assert result_map[("forces", 1)].value == pytest.approx(0.3 / 6.0, rel=1e-7)


def test_atom_type_metric_without_matching_entries_can_opt_in_to_success(ds_ref_cmp):
    ref, cmp = ds_ref_cmp
    metrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(
                property="forces",
                metric="mae",
                atom_type=16,
                no_data_is_success=True,
            ),
        ],
    )

    results = metrics.compare_two(dataset_ref=ref, dataset_cmp=cmp)

    assert len(results) == 1
    assert results[0].n_entries == 0
    assert results[0].success == "NoData"
    assert np.isnan(results[0].value)
    assert results[0].no_data_is_success is True


def test_prop_not_exists(ds_ref_cmp):
    ref, cmp = ds_ref_cmp
    metrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(property="NoExists", metric="mae", atom_type=0, components=-1),
        ],
    )
    results = metrics.compare_two(dataset_ref=ref, dataset_cmp=cmp)

    assert len(results) == 1
    assert results[0].n_entries == 0
    assert results[0].success == "NoData"


def test_atom_type_metrics_requires_an_atom_resolved_property(ds_ref_cmp):
    ref, cmp = ds_ref_cmp
    metrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", atom_type=1),
        ],
    )
    with pytest.raises(
        ValueError,
        match=(
            r"no axis in diff shape \(\) matches number of atoms 2 for atom_type 1 "
            r"| Property 'energy' | At relative index: 0"
        ),
    ):
        results = metrics.compare_two(dataset_ref=ref, dataset_cmp=cmp)


def test_pairwise_result_validation():
    result_no_data = PairwiseResult(
        property="energy",
        metric="mae",
        units="eV",
        value=0.0,
        n_entries=0,
    )
    assert result_no_data.success == "NoData"

    result_ok = PairwiseResult(
        property="energy",
        metric="mae",
        units="eV",
        value=0.1,
        n_entries=2,
        target=0.2,
    )
    assert result_ok.success == "OK"

    result_fail = PairwiseResult(
        property="energy",
        metric="mae",
        units="eV",
        value=0.2,
        n_entries=2,
        target=0.1,
    )
    assert result_fail.success == "FAIL"

    result_higher_is_better = PairwiseResult(
        property="energy",
        metric="mae",
        units="eV",
        value=0.2,
        n_entries=2,
        target=0.1,
        lower_is_better=False,
    )
    assert result_higher_is_better.success == "OK"

    result_custom = PairwiseResult(
        property="energy",
        metric="mae",
        units="eV",
        value=0.1,
        n_entries=0,
        success="Custom",
    )
    assert result_custom.success == "Custom"


def test_components_metrics_all_atoms(ds_ref_cmp):
    ref, cmp = ds_ref_cmp
    metrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", atom_type=-1, components=1),
        ],
    )
    results = metrics.compare_two(dataset_ref=ref, dataset_cmp=cmp)
    result_map = {(r.atom_type, r.components): r for r in results}
    assert result_map[(-1, 0)].value == pytest.approx(0.1 / 3.0, rel=1e-7)
    assert result_map[(-1, 1)].value == pytest.approx(0.0, rel=1e-7)
    assert result_map[(-1, 2)].value == pytest.approx(0.7 / 3.0, rel=1e-7)
    assert result_map[(-1, 0)].n_entries == 3
    assert result_map[(-1, 1)].n_entries == 3
    assert result_map[(-1, 2)].n_entries == 3


def test_components_metrics_all_atoms_error(ds_ref_cmp):
    ref, cmp = ds_ref_cmp
    metrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", atom_type=-1, components=0),
        ],
    )
    with pytest.raises(
        ValueError,
        match=r"components axis 0 size 2 matches number of atoms 2 \| Property 'forces' \| At relative index: 0",
    ):
        metrics.compare_two(dataset_ref=ref, dataset_cmp=cmp)


def test_atom_type_component_metrics(ds_ref_cmp):
    ref, cmp = ds_ref_cmp
    metrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", atom_type=1, components=1),
        ],
    )
    results = metrics.compare_two(dataset_ref=ref, dataset_cmp=cmp)
    result_map = {(r.atom_type, r.components): r for r in results}
    assert result_map[(1, 0)].value == pytest.approx(0.05, rel=1e-7)
    assert result_map[(1, 1)].value == pytest.approx(0.0, rel=1e-7)
    assert result_map[(1, 2)].value == pytest.approx(0.1, rel=1e-7)
