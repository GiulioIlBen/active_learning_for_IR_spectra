from pathlib import Path

import numpy as np
import pytest

from scm.moliterate.filters.conformers_ams import (
    AMSConformersFilter,
    UniqueConformersAMSSettings,
    UniqueConformersCrestSettings,
    UniqueConformersRMSDSettings,
    UniqueConformersTFDSettings,
)
from scm.moliterate.interfaces.rkf_files import RKFMolData

TUTORIAL_DATA = Path(__file__).resolve().parents[1] / "tutorials" / "data"


def test_unique_conformers_ams_settings_build_sets_all_fields(monkeypatch):
    import scm.moliterate.filters.conformers_ams as conformers_ams

    class DummySettings:
        def __init__(self):
            self.energy_threshold = None
            self.dihedral_threshold = None
            self.distance_threshold = None
            self.use_torsions = None
            self.irrelevant_atoms = None
            self.max_dist = None
            self.compare_across_molecules = None

    class DummyConf:
        def __init__(self, energy_threshold, dihedral_threshold, distance_threshold):
            self.init_args = (energy_threshold, dihedral_threshold, distance_threshold)
            self.settings = DummySettings()

    monkeypatch.setattr(conformers_ams, "_get_impls", lambda: {"ams": DummyConf})

    settings = conformers_ams.UniqueConformersAMSSettings(
        energy_threshold=1.1,
        dihedral_threshold=22.0,
        distance_threshold=0.2,
        use_torsions=False,
        irrelevant_atoms=[1, 2, 3],
        max_dist=4.5,
        compare_across_molecules=False,
    )

    conf = settings.build()

    assert conf.init_args == (1.1, 22.0, 0.2)
    assert conf.settings.energy_threshold == 1.1
    assert conf.settings.dihedral_threshold == 22.0
    assert conf.settings.distance_threshold == 0.2
    assert conf.settings.use_torsions is False
    assert conf.settings.irrelevant_atoms == [1, 2, 3]
    assert conf.settings.max_dist == 4.5
    assert conf.settings.compare_across_molecules is False


@pytest.mark.parametrize(
    "settings_cls",
    [
        UniqueConformersAMSSettings,
        UniqueConformersCrestSettings,
        UniqueConformersRMSDSettings,
        UniqueConformersTFDSettings,
    ],
)
def test_conformers_filter_implementations_on_tutorial_molecule_rkf(settings_cls):
    pytest.importorskip("scm.conformers")
    pytest.importorskip("scm.plams")

    data_path = TUTORIAL_DATA / "molecule.rkf"
    if not data_path.exists():
        pytest.skip("Tutorial molecule.rkf not available")

    db = RKFMolData(data_source=str(data_path))
    flt = AMSConformersFilter(
        settings=settings_cls(),
        grouping_criterion=None,
        reorder_on_hash_connectivity=False,
    )

    selected = flt.apply_filter(db)

    assert selected is not None
    assert isinstance(selected, np.ndarray)
    assert selected.dtype == np.int64
    assert selected.size >= 1
    assert selected.min() >= 0
    assert selected.max() < len(db)
    assert np.array_equal(selected, np.unique(selected))
    assert np.array_equal(selected, np.sort(selected))


def test_conformers_filter_rejects_periodic_rkf():
    pytest.importorskip("scm.plams")

    data_path = TUTORIAL_DATA / "periodic.rkf"
    if not data_path.exists():
        pytest.skip("Tutorial periodic.rkf not available")

    db = RKFMolData(data_source=str(data_path))
    first_row = next(iter(db))
    if not np.any(first_row.system.pbc):
        pytest.skip("Tutorial periodic.rkf does not appear to be periodic")

    flt = AMSConformersFilter(
        settings=UniqueConformersAMSSettings(),
        grouping_criterion=None,
        reorder_on_hash_connectivity=False,
    )

    with pytest.raises(ValueError, match="Periodic systems"):
        flt.apply_filter(db)
