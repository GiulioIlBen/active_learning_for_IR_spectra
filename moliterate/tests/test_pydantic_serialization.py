import json
import warnings

import numpy as np
import pytest

try:
    import apricot  # noqa: F401

    HAS_APRICOT = True
except ImportError:
    HAS_APRICOT = False

from scm.moliterate.core import (
    ConcatChemDataSet,
    PropertyInfo,
)
from scm.moliterate.filters import (
    AMSConformersFilter,
    AndFilter,
    ApricotFilter,
    ConditionsFilter,
    NotFilter,
    OrFilter,
    PipeFilter,
    RemoveDuplicates,
)
from scm.moliterate.filters.conformers_ams import (
    UniqueConformersAMSSettings,
    UniqueConformersCrestSettings,
    UniqueConformersRMSDSettings,
    UniqueConformersTFDSettings,
)
from scm.moliterate.filters.scalar_filters import FarthestPointFilter, LinearSteppedFilter, RandomFilter, TopNFilter
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.partition_criteria.criteria_1d import NumberOfAtomsCriterion
from scm.moliterate.transforms.composed_transforms import (
    ComposedTransform,
)


def _canonical_json(model) -> str:
    return json.dumps(json.loads(model.model_dump_json()), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def test_property_info_json_roundtrip_stable():
    info = PropertyInfo(name="energy", unit="eV", shape=("nAtoms", 3), description="Electronic energy")
    dumped = _canonical_json(info)
    expected = json.dumps(
        {
            "description": "Electronic energy",
            "name": "energy",
            "shape": ["nAtoms", 3],
            "unit": "eV",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert dumped == expected
    loaded = PropertyInfo.model_validate_json(info.model_dump_json())
    assert loaded == info


def test_filter_json_roundtrip_stable():
    flt = RandomFilter(num_samples=1, seed=0)
    dumped = _canonical_json(flt)
    expected = json.dumps(
        {"make_warning": True, "num_samples": 1, "seed": 0, "type": "RandomFilter"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert dumped == expected
    loaded = RandomFilter.model_validate_json(flt.model_dump_json())
    assert loaded == flt


def test_ams_conformers_filter_json_roundtrip_stable():
    flt = AMSConformersFilter(
        settings=UniqueConformersAMSSettings(
            energy_threshold=0.3,
            dihedral_threshold=25.0,
            distance_threshold=0.12,
            use_torsions=False,
            irrelevant_atoms=[1, 2],
            max_dist=4.5,
            compare_across_molecules=False,
        ),
        grouping_criterion=None,
        reorder_on_hash_connectivity=False,
        energy_property_key="energy",
        max_energy=1.5,
        log_data=True,
    )
    dumped = _canonical_json(flt)
    expected = json.dumps(
        {
            "energy_extractor": None,
            "energy_property_key": "energy",
            "grouping_criterion": None,
            "log_data": True,
            "max_energy": 1.5,
            "reorder_on_hash_connectivity": False,
            "settings": {
                "compare_across_molecules": False,
                "dihedral_threshold": 25.0,
                "distance_threshold": 0.12,
                "energy_threshold": 0.3,
                "irrelevant_atoms": [1, 2],
                "max_dist": 4.5,
                "use_torsions": False,
            },
            "type": "AMSConformersFilter",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert dumped == expected
    loaded = AMSConformersFilter.model_validate_json(flt.model_dump_json())
    assert loaded == flt


@pytest.mark.parametrize(
    ("settings", "expected_keys"),
    [
        (
            UniqueConformersAMSSettings(),
            {
                "energy_threshold",
                "dihedral_threshold",
                "distance_threshold",
                "use_torsions",
                "irrelevant_atoms",
                "max_dist",
                "compare_across_molecules",
            },
        ),
        (UniqueConformersRMSDSettings(), {"energy_threshold", "rmsd_threshold"}),
        (
            UniqueConformersCrestSettings(),
            {
                "energy_threshold",
                "rmsd_threshold",
                "bconst_threshold",
                "bthr_max",
                "bthr_shift",
                "bconst_scaling_factor",
                "size_threshold",
                "follow_paper",
            },
        ),
        (UniqueConformersTFDSettings(), {"energy_threshold", "tfd_threshold", "use_energy_threshold", "use_weights"}),
    ],
)
def test_ams_conformers_filter_json_serialization_settings_variants(settings, expected_keys):
    flt = AMSConformersFilter(
        settings=settings,
        grouping_criterion=None,
    )
    payload = json.loads(flt.model_dump_json())
    assert set(payload["settings"].keys()) == expected_keys


def test_property_extractor_json_roundtrip_stable():
    extractor = NumberOfAtomsCriterion()
    dumped = _canonical_json(extractor)
    expected = json.dumps({"type": "NumberOfAtomsCriterion"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert dumped == expected
    loaded = NumberOfAtomsCriterion.model_validate_json(extractor.model_dump_json())
    assert loaded == extractor


def test_composed_transform_json_roundtrip_stable_empty():
    composed = ComposedTransform(transforms=[])
    dumped = _canonical_json(composed)
    expected = json.dumps({"transforms": []}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert dumped == expected
    loaded = ComposedTransform.model_validate_json(composed.model_dump_json())
    assert loaded.transforms == []


def test_dataset_json_roundtrip_stable():
    dataset = InMemoryMolData(
        data_source="unit-test",
        absolute_idxs=np.array([0, 2], dtype=np.int64),
        transforms=[],
    )
    dumped = _canonical_json(dataset)
    expected = json.dumps(
        {
            "absolute_idxs": [0, 2],
            "data_source": "unit-test",
            "transforms": [],
            "type": "InMemoryMolData",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert dumped == expected
    loaded = InMemoryMolData.model_validate_json(dataset.model_dump_json())
    np.testing.assert_array_equal(loaded.absolute_idxs, np.array([0, 2], dtype=np.int64))


def test_dataset_writer_json_roundtrip_stable():
    writer = InMemoryMolData(
        data_source="dummy.db",
        absolute_idxs=None,
        transforms=[],
    )
    dumped = _canonical_json(writer)
    expected = json.dumps(
        {
            "absolute_idxs": None,
            "data_source": "dummy.db",
            "transforms": [],
            "type": "InMemoryMolData",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    assert dumped == expected
    loaded = InMemoryMolData.model_validate_json(writer.model_dump_json())
    assert loaded.data_source == "dummy.db"


def test_concat_dataset_json_roundtrip_stable():
    ds_one = InMemoryMolData(data_source="a", absolute_idxs=None, transforms=[])
    ds_two = InMemoryMolData(data_source="b", absolute_idxs=None, transforms=[])
    concat = ConcatChemDataSet(data_source=[ds_one, ds_two])
    dumped = json.dumps(concat.model_dump(mode="json"), sort_keys=True)
    expected = json.dumps(
        {
            "absolute_idxs": None,
            "data_source": [
                {
                    "absolute_idxs": None,
                    "data_source": "a",
                    "transforms": [],
                    "type": "InMemoryMolData",
                },
                {
                    "absolute_idxs": None,
                    "data_source": "b",
                    "transforms": [],
                    "type": "InMemoryMolData",
                },
            ],
            "transforms": [],
            "type": "ConcatChemDataSet",
        },
        sort_keys=True,
    )
    assert dumped == expected


def test_concat_dataset_json_roundtrip_empty():
    concat = ConcatChemDataSet(data_source=[])
    dumped = concat.model_dump_json()
    loaded = ConcatChemDataSet.model_validate_json(dumped)
    assert loaded.data_source == []


@pytest.mark.parametrize(
    "model",
    [
        PropertyInfo(name="energy", unit="eV", shape="float", description=""),
        RandomFilter(num_samples=1, seed=0),
        NumberOfAtomsCriterion(),
        ComposedTransform(transforms=[]),
    ],
)
def test_core_models_json_idempotent(model):
    dumped = model.model_dump_json()
    loaded = model.__class__.model_validate_json(dumped)
    assert loaded.model_dump_json() == dumped


def _assert_filter_roundtrip(model):
    dumped = model.model_dump()
    loaded = model.__class__.model_validate(dumped)
    assert loaded.model_dump() == dumped

    dumped_json = model.model_dump_json()
    loaded_json = model.__class__.model_validate_json(dumped_json)
    assert loaded_json.model_dump_json() == dumped_json


@pytest.mark.parametrize(
    "model",
    [
        RandomFilter(num_samples=1, seed=0),
        TopNFilter(partition_criterion=NumberOfAtomsCriterion(), num_samples=1),
        FarthestPointFilter(partition_criterion=NumberOfAtomsCriterion(), num_samples=1, seed=0),
        LinearSteppedFilter(step=2, num_samples=2),
        ConditionsFilter(conditions="foo=1"),
        ConditionsFilter(conditions=["foo=1"]),
        RemoveDuplicates(
            clustering_algorithm=NumberOfAtomsCriterion(),
            selection_strategy=RandomFilter(num_samples=1, seed=0),
        ),
        AndFilter(
            filters={
                RandomFilter(num_samples=1, seed=0),
                TopNFilter(partition_criterion=NumberOfAtomsCriterion(), num_samples=1),
            }
        ),
        OrFilter(
            filters={
                RandomFilter(num_samples=1, seed=0),
                TopNFilter(partition_criterion=NumberOfAtomsCriterion(), num_samples=1),
            }
        ),
        NotFilter(filter=RandomFilter(num_samples=1, seed=0)),
        PipeFilter(
            filters=[
                RandomFilter(num_samples=1, seed=0),
                TopNFilter(partition_criterion=NumberOfAtomsCriterion(), num_samples=1),
            ]
        ),
    ],
)
def test_roundtrip_dump_filters(model):
    # this is because: https://stackoverflow.com/questions/75345190/pytest-overrides-existing-warning-filters
    from scm.moliterate.filters.bitwise_filters import __filter_known_warnings

    __filter_known_warnings()
    _assert_filter_roundtrip(model)


def test_ams_conformers_filter_hash_stable_for_same_config():
    base = AMSConformersFilter(
        settings=UniqueConformersRMSDSettings(energy_threshold=0.2, rmsd_threshold=0.3),
        grouping_criterion=None,
        reorder_on_hash_connectivity=False,
        energy_property_key="energy",
        max_energy=0.8,
    )
    same = AMSConformersFilter(
        settings=UniqueConformersRMSDSettings(energy_threshold=0.2, rmsd_threshold=0.3),
        grouping_criterion=None,
        reorder_on_hash_connectivity=False,
        energy_property_key="energy",
        max_energy=0.8,
    )
    assert hash(base) == hash(same)


def test_ams_conformers_filter_hash_changes_with_settings():
    base = AMSConformersFilter(
        settings=UniqueConformersRMSDSettings(energy_threshold=0.2, rmsd_threshold=0.3),
        grouping_criterion=None,
    )
    updated = AMSConformersFilter(
        settings=UniqueConformersRMSDSettings(energy_threshold=0.2, rmsd_threshold=0.4),
        grouping_criterion=None,
    )
    assert hash(base) != hash(updated)


def test_ams_conformers_filter_hash_includes_energy_extractor():
    def energy_a(_entry):
        return 1.0

    def energy_b(_entry):
        return 1.0

    base = AMSConformersFilter(
        settings=UniqueConformersRMSDSettings(),
        grouping_criterion=None,
        energy_extractor=energy_a,
    )
    updated = AMSConformersFilter(
        settings=UniqueConformersRMSDSettings(),
        grouping_criterion=None,
        energy_extractor=energy_b,
    )
    assert hash(base) != hash(updated)


@pytest.mark.skipif(not HAS_APRICOT, reason="apricot is not installed")
def test_apricot_filter_roundtrip():
    model = ApricotFilter(
        n_samples=1,
        partition_criterion=NumberOfAtomsCriterion(),
    )
    _assert_filter_roundtrip(model)


# def test_model_coerce_for_filters():
#     data = {
#         'type': 'TopN',
#         'partition_criterion': {
#             'type': 'PropertyCriterion',
#             'property_key': 'energy',
#             'post_process': None,
#             'unit_transform': None
#         },
#         'num_samples': 0.5
#     }

#     class DoNothing(BaseFilter):
#         def apply_filter(self, db: BaseChemDataSet):
#             return None

#     and_filter = {'type': 'AND', 'filters': [data, DoNothing().model_dump()]}

#     #TODO I would like to improve this error message:
#     # TypeError: Can't instantiate abstract class BaseFilter with abstract methods apply_filter,
#     # where should I implement this? such that It says I was not able to find a
#     AndFilter(**and_filter)
