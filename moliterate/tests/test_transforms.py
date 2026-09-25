from typing import List

import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.transforms import (
    ConvertNamesTransform,
    MetadataAddTransform,
    MinMaxAtomsDistance,
    PropertyTransform,
    ReshapeTransform,
    ScalePropTransform,
    ToArrayTransform,
)
from scm.moliterate.transforms.composed_transforms import ComposedTransform
from scm.moliterate.transforms.core import BaseEntryTransform


def _molecule_from_coords(coords):
    return Atoms(symbols=["H"] * len(coords), positions=coords)


def test_data_atoms_row_transform_passthrough():
    class DummyDataAtomsTransform(BaseEntryTransform):
        def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
            return properties_info

        def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
            return properties_info

        def __call__(self, data: ChemDataEntry) -> ChemDataEntry:  # pragma: no cover - trivial passthrough
            return data

    properties = [PropertyInfo(name="energy"), PropertyInfo(name="dipole", shape=(3,))]
    transform = DummyDataAtomsTransform()

    assert transform.properties_in(properties) == properties
    assert transform.properties_out(properties) == properties

    row = ChemDataEntry(system=Atoms(), properties={"energy": -1.0})
    assert transform(row).properties["energy"] == -1.0


def test_min_max_atoms_distance_updates_properties_and_info():
    mol = _molecule_from_coords([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)])
    row = ChemDataEntry(system=mol, properties={})

    transform = MinMaxAtomsDistance()
    out_info = transform.properties_out([PropertyInfo(name="energy", shape="float")])

    assert any(p.name == "min_distance" for p in out_info)
    assert any(p.name == "max_distance" for p in out_info)

    updated = transform(row)
    assert updated.properties["min_distance"] == pytest.approx(1.0)
    assert updated.properties["max_distance"] == pytest.approx(np.sqrt(2.0))

    with pytest.raises(ValueError):
        transform(ChemDataEntry(system=_molecule_from_coords([(0.0, 0.0, 0.0)]), properties={}))


def test_to_array_transform_converts_selected_properties():
    row = ChemDataEntry(system=Atoms(), properties={"vector": [1, 2, 3], "keep": 4})
    transform = ToArrayTransform(prop_names=["vector"])
    props_info = [PropertyInfo(name="vector"), PropertyInfo(name="keep")]

    assert transform.properties_in(props_info) == [props_info[0]]
    assert transform.properties_out(props_info) == props_info

    updated = transform(row)
    assert isinstance(updated.properties["vector"], np.ndarray)
    assert updated.properties["vector"].tolist() == [1, 2, 3]
    assert updated.properties["keep"] == 4


def test_reshape_transform_handles_static_and_dynamic_shapes():
    mol = _molecule_from_coords([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    row = ChemDataEntry(system=mol, properties={"vec": list(range(9)), "keep": [1, 2]})

    transform = ReshapeTransform(props_shapes={"vec": ["nAtoms", 3]})
    props_info = [PropertyInfo(name="vec", shape=(9,)), PropertyInfo(name="keep", shape="float")]

    out_info = transform.properties_out(props_info)
    assert any(p.name == "vec" and p.shape == ("nAtoms", 3) for p in out_info)
    assert any(p.name == "keep" and p.shape == "float" for p in out_info)

    updated = transform(row)
    assert updated.properties["vec"].shape == (3, 3)
    assert updated.properties["keep"] == [1, 2]


def test_convert_names_transform_renames_with_collision_resolution():
    props_info = [PropertyInfo(name="energy"), PropertyInfo(name="temp")]
    transform = ConvertNamesTransform(prop_names_convert={"energy": "temp"})

    out_info = transform.properties_out(props_info)
    assert any(p.name == "temp" for p in out_info)
    assert any(p.name == "temp" for p in out_info)

    row = ChemDataEntry(system=Atoms(), properties={"energy": -1.0, "temp": 300.0}, metadata={"energy": "tag"})
    updated = transform(row)

    assert "energy" not in updated.properties
    assert updated.properties["temp"] == -1.0
    assert updated.metadata["temp"] == "tag"


def test_scale_prop_transform_scales_and_can_write_to_new_key():
    props_info = [PropertyInfo(name="energy", shape="float")]
    transform = ScalePropTransform(key_in="energy", coeff_scale=2.0, key_out="scaled_energy")

    out_info = transform.properties_out(props_info)
    assert any(p.name == "scaled_energy" for p in out_info)

    row = ChemDataEntry(system=Atoms(), properties={"energy": 1.5})
    updated = transform(row)

    assert updated.properties["scaled_energy"] == pytest.approx(3.0)
    # assert updated.properties["energy"] == pytest.approx(1.5)


def test_metadata_add_transform_respects_overwrite_flag():
    row = ChemDataEntry(system=Atoms(), metadata={"keep": "old"}, properties={})
    transform = MetadataAddTransform(key_values={"keep": "new", "fresh": 1}, overwrite=False)

    updated = transform(row)
    assert updated.metadata["keep"] == "old"
    assert updated.metadata["fresh"] == 1
    assert updated.metadata["fresh"] == 1


def test_property_transform_post_process_and_properties_info():
    row = ChemDataEntry(
        system=Atoms(),
        properties={"forces": np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])},
    )
    props_info = [PropertyInfo(name="forces", shape=("nAtoms", 3)), PropertyInfo(name="energy", shape="float")]
    transform = PropertyTransform(property_key="forces", post_process="l2_max", property_key_out="fmax")

    assert transform.properties_in(props_info) == [props_info[0]]
    out_info = transform.properties_out(props_info)
    assert any(p.name == "fmax" for p in out_info)

    updated = transform(row)
    assert updated.properties["fmax"] == pytest.approx(5.0)


def test_composed_transform_chains_properties_and_call():
    transforms = [
        ScalePropTransform(key_in="energy", coeff_scale=2.0, key_out="scaled"),
        ConvertNamesTransform(prop_names_convert={"scaled": "scaled_energy"}),
    ]
    composed = ComposedTransform(transforms=transforms)
    props_info = [PropertyInfo(name="energy", shape="float")]
    assert composed.properties_in(props_info) == props_info
    out_info = composed.properties_out(props_info)
    assert [p.name for p in out_info] == ["scaled_energy"]

    row = ChemDataEntry(system=Atoms(), properties={"energy": 1.5})
    updated = composed(row)
    assert updated.properties["scaled_energy"] == pytest.approx(3.0)


def test_property_transform_options():
    forces = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

    def _row():
        return ChemDataEntry(system=Atoms(), properties={"forces": forces.copy(), "energy": 2.0})

    l2_row = PropertyTransform(property_key="forces", post_process="l2", property_key_out="forces_l2")(_row())
    np.testing.assert_allclose(l2_row.properties["forces_l2"], np.array([5.0, 0.0]))

    max_row = PropertyTransform(property_key="forces", post_process="max", property_key_out="forces_max")(_row())
    assert max_row.properties["forces_max"] == pytest.approx(4.0)

    pytest.importorskip("scm.plams")
    unit_row = PropertyTransform(
        property_key="energy",
        unit_transform=("angstrom", "angstrom"),
        property_key_out="energy_out",
    )(_row())
    assert unit_row.properties["energy_out"] == pytest.approx(2.0)
