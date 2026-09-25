import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.partition_criteria.criteria_1d import (
    ChemFormulaCriterion,
    FmaxCriterion,
    GyrationRCriterion,
    MaxDistanceCriterion,
    MetadataCriterion,
    MinDistanceCriterion,
    NumberOfAtomsCriterion,
    PropertyCriterion,
)
from scm.moliterate.partition_criteria.mol_connectivity_1d import MolConnectivityCriterion


def _atoms(symbols_with_coords):
    symbols, coords = zip(*symbols_with_coords)
    return Atoms(symbols=symbols, positions=list(coords))


def test_chem_formula_criterion_and_group_by_preserve_order():
    h2 = _atoms([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))])
    h2o = _atoms([("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0)), ("H", (0.0, 1.0, 0.0))])
    db = InMemoryMolData.create()
    db.add_system(ChemDataEntry(system=h2))
    db.add_system(ChemDataEntry(system=h2o))
    db.add_system(ChemDataEntry(system=h2))

    criterion = ChemFormulaCriterion()
    values = criterion(db)
    assert values.ravel().tolist() == ["H2", "H2O", "H2"]

    groups = ChemFormulaCriterion.group_by(values)
    assert set(groups) == {"H2", "H2O"}
    np.testing.assert_array_equal(groups["H2"], np.array([0, 2]))
    np.testing.assert_array_equal(groups["H2O"], np.array([1]))


def test_number_of_atoms_counts_each_molecule():
    db = InMemoryMolData.create()
    db.add_system(ChemDataEntry(system=_atoms([("H", (0.0, 0.0, 0.0))])))
    db.add_system(ChemDataEntry(system=_atoms([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))])))
    db.add_system(
        ChemDataEntry(system=_atoms([("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0)), ("H", (0.0, 1.0, 0.0))]))
    )

    result = NumberOfAtomsCriterion()(db)

    assert result.tolist() == [1, 2, 3]


def test_min_and_max_distance_match_known_geometries():
    h2 = _atoms([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))])
    bent = _atoms([("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0)), ("H", (0.0, 1.0, 0.0))])
    db = InMemoryMolData.create()
    db.add_system(ChemDataEntry(system=h2))
    db.add_system(ChemDataEntry(system=bent))

    min_result = MinDistanceCriterion()(db)
    max_result = MaxDistanceCriterion()(db)

    assert min_result.shape == (2,)
    assert max_result.shape == (2,)
    assert min_result[0] == pytest.approx(0.74)
    assert max_result[0] == pytest.approx(0.74)
    assert min_result[1] == pytest.approx(1.0)
    assert max_result[1] == pytest.approx(np.sqrt(2.0))


def test_gyration_radius_criterion_uses_molecule_values():
    mols = [
        _atoms([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]),
        _atoms([("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0)), ("H", (0.0, 1.0, 0.0))]),
    ]
    expected = [float(ChemDataEntry(system=m).get_gyration_radius()) for m in mols]
    db = InMemoryMolData.create()
    for m in mols:
        db.add_system(ChemDataEntry(system=m))

    result = GyrationRCriterion()(db)

    np.testing.assert_allclose(result.ravel(), expected)


def test_property_criterion_supports_metadata_and_unit_conversion():
    pytest.importorskip("scm.plams")
    props_info = [PropertyInfo(name="energy")]
    db = InMemoryMolData.create(available_properties=props_info)
    db.add_system(ChemDataEntry(system=Atoms(), properties={"energy": 1.0}))
    db.add_system(ChemDataEntry(system=Atoms(), properties={"energy": 2.0}))
    ratio_kcalMol_hartree = 0.0015936014383657205

    criterion = PropertyCriterion(property_key="energy", unit_transform=("kcal/mol", "hartree"))
    result = criterion(db)

    assert result.shape == (2,)
    assert result[0] == pytest.approx(1.0 * ratio_kcalMol_hartree)
    assert result[1] == pytest.approx(2.0 * ratio_kcalMol_hartree)


def test_property_criterion_l2_max_reduces_vector_property():
    props_info = [PropertyInfo(name="tensor")]
    db = InMemoryMolData.create(available_properties=props_info)
    db.add_system(
        ChemDataEntry(
            system=Atoms(),
            properties={"tensor": np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])},
        )
    )
    db.add_system(
        ChemDataEntry(
            system=Atoms(),
            properties={"tensor": np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]])},
        )
    )

    result = PropertyCriterion(property_key="tensor", post_process="l2_max")(db)

    assert result.tolist() == [2.0, 5.0]


def test_fmax_criterion_uses_force_like_keys_and_norms():
    props_info = [PropertyInfo(name="Gradients")]
    db = InMemoryMolData.create(available_properties=props_info)
    db.add_system(
        ChemDataEntry(
            system=_atoms([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))]),
            properties={"Gradients": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]},
        )
    )
    db.add_system(
        ChemDataEntry(
            system=_atoms([("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.0)), ("H", (0.0, 1.0, 0.0))]),
            properties={"Gradients": [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [1.0, 1.0, 1.0]]},
        )
    )

    result = FmaxCriterion()(db)

    assert result.tolist() == [2.0, 5.0]


def test_property_criterion_rejects_invalid_post_process():
    db = InMemoryMolData.create(available_properties=[PropertyInfo(name="energy")])
    db.add_system(ChemDataEntry(system=Atoms(), properties={"energy": 1.0}))

    with pytest.raises(ValueError):
        PropertyCriterion(property_key="energy", post_process="bad")(db)


def test_metadata_criterion_concatenates_requested_metadata():
    db = InMemoryMolData.create()
    db.add_system(ChemDataEntry(system=Atoms(), metadata={"a": 1, "b": "x"}))
    db.add_system(ChemDataEntry(system=Atoms(), metadata={"a": 2, "b": None}))

    result = MetadataCriterion(metadata_keys=["a", "b"])(db)

    assert result.tolist() == ["1%x", "2%None"]


def test_mol_connectivity_normalizes_integer_levels():
    criterion = MolConnectivityCriterion(hash_connectivity_level=2)
    assert criterion.hash_connectivity_level == "bond_orders"
    assert criterion.hash_connectivity_level_value == 2


def test_mol_connectivity_rejects_unsupported_level():
    with pytest.raises(ValueError, match="Unsupported hash_connectivity_level"):
        MolConnectivityCriterion(hash_connectivity_level=99)


def test_mol_connectivity_calls_label_with_flags_and_level():
    class _StubMolecule:
        def __init__(self, name, calls):
            self._name = name
            self._calls = calls

        def label(self, *, level, flags, keep_labels):
            self._calls.append((self._name, level, flags, keep_labels))
            return f"{self._name}:{level}:{flags}:{keep_labels}"

    class _StubAtoms:
        def get_pbc(self):
            return None

    class _Row:
        def __init__(self, molecule):
            self.molecule = molecule
            self.atoms = _StubAtoms()

    class _DB:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    calls = []
    mols = [_StubMolecule("a", calls), _StubMolecule("b", calls)]
    db = _DB([_Row(m) for m in mols])
    criterion = MolConnectivityCriterion(
        hash_connectivity_level="stereochemistry",
        flags={"include_rings": True},
    )

    result = criterion(db)

    assert result.tolist() == [
        "a:3:{'include_rings': True}:True",
        "b:3:{'include_rings': True}:True",
    ]
    assert calls == [
        ("a", 3, {"include_rings": True}, True),
        ("b", 3, {"include_rings": True}, True),
    ]


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("formula", "MolFormula"),
        ("single_bond_connectivity", "MolSingleBondConnectivity"),
        ("bond_orders", "MolBondOrders"),
    ],
)
def test_mol_connectivity_name(level, expected):
    assert MolConnectivityCriterion(hash_connectivity_level=level).name == expected


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        (ChemFormulaCriterion(), "formula"),
        (NumberOfAtomsCriterion(), "NAtoms"),
        (MinDistanceCriterion(), "min(distance)"),
        (MaxDistanceCriterion(), "max(distance)"),
        (GyrationRCriterion(), "GyrationRad"),
        (PropertyCriterion(property_key="energy"), "energy"),
        (PropertyCriterion(property_key="forces", post_process="l2_max"), "max(||forces||)"),
        (PropertyCriterion(property_key="forces", post_process="l2"), "||forces||"),
        (PropertyCriterion(property_key="forces", post_process="max"), "max(forces)"),
        (FmaxCriterion(), "fmax"),
        (MetadataCriterion(metadata_keys=["a", "b"]), "md[a%b]"),
        (MetadataCriterion(metadata_keys=["a", "b"], concat_symbol="|"), "md[a|b]"),
    ],
)
def test_criteria_name_properties(criterion, expected):
    assert criterion.name == expected
