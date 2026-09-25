from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest
from scm.moliterate import PropertyInfo
from scm.plams import AMSJob

from scm.active_learning.engines import to_ams_engine
from scm.active_learning.engines._shared import apply_ams_external_capabilities, extract_system
from scm.active_learning.engines.ase_engine import ASEEngine, ASEPythonSettings
from scm.active_learning.task_parallelization import SerialStrategy


class DummyCalculator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyAtoms:
    def __init__(self):
        self.calc = None
        self.energy_calls = 0

    def get_potential_energy(self, force_consistent: bool = False):
        self.energy_calls += 1
        return 2.0 if force_consistent else 1.0


class ASEOnlyRow:
    """Expose ASE atoms while simulating an unavailable optional ChemicalSystem conversion used by ASE database rows."""

    def __init__(self, atoms):
        self.atoms = atoms

    @property
    def chemical_system(self):
        raise ImportError("libbase is not installed")


def test_apply_ams_external_capabilities_warns_when_amspipe_is_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scm.amspipe":
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    calculator = SimpleNamespace()
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.warns(UserWarning, match="AMSExternalCapabilities are unavailable"):
        apply_ams_external_capabilities(calculator, ["energy", "forces"])

    assert calculator.ams_capabilities is None


def test_validate_properties_and_alias_mapping():
    engine = ASEEngine(engine_id="ase", calculator=f"{__name__}.DummyCalculator")
    props = [PropertyInfo(name="Energy"), PropertyInfo(name="potential_energy"), PropertyInfo(name="dipolemoment")]

    canonical = engine._validate_properties(props)

    assert canonical == ["energy", "energy", "dipole_moment"]


def test_build_constructor_collects_calculator_kwargs():
    engine = ASEEngine.build(
        engine_id="ase",
        calculator="ase.calculators.lj.LennardJones",
        epsilon=0.5,
        rc=8.0,
    )

    assert engine.engine_id == "ase"
    assert engine.calculator == "ase.calculators.lj.LennardJones"
    assert engine.calculator_kwargs == {"epsilon": 0.5, "rc": 8.0}


def test_build_constructor_accepts_explicit_python_settings_without_mixing_kwargs():
    engine = ASEEngine.build(
        engine_id="ase",
        calculator="ase.calculators.lj.LennardJones",
        python=ASEPythonSettings(type="Current"),
        epsilon=0.5,
    )

    assert engine.python.type == "Current"
    assert engine.calculator_kwargs == {"epsilon": 0.5}


def test_build_tblite_uses_tblite_calculator_defaults():
    engine = ASEEngine.build_tblite(engine_id="xtb")

    assert engine.engine_id == "xtb"
    assert engine.calculator == "tblite.ase.TBLite"
    assert engine.calculator_kwargs == {"method": "GFN2-xTB"}


def test_build_tblite_allows_overrides():
    engine = ASEEngine.build_tblite(
        engine_id="xtb",
        method="GFN1-xTB",
        charge=1,
        multiplicity=2,
    )

    assert engine.calculator_kwargs == {
        "method": "GFN1-xTB",
        "charge": 1,
        "multiplicity": 2,
    }


def test_ase_engine_finetunable_flag_defaults_to_false_and_can_be_enabled():
    default_engine = ASEEngine(engine_id="ase", calculator=f"{__name__}.DummyCalculator")
    finetunable_engine = ASEEngine(
        engine_id="mace",
        calculator=f"{__name__}.DummyCalculator",
        finetunable=True,
    )

    assert default_engine.is_finetunable() is False
    assert finetunable_engine.is_finetunable() is True


def test_ase_engine_exposes_model_paths_committee_members():
    engine = ASEEngine(
        engine_id="mace",
        calculator=f"{__name__}.DummyCalculator",
        calculator_kwargs={
            "model_paths": ["member-0.model", "member-1.model"],
            "device": "cpu",
        },
        finetunable=True,
    )

    members = engine.committee_members

    assert engine.has_committee is True
    assert [member.engine_id for member in members] == ["mace_m000", "mace_m001"]
    assert [member.calculator_kwargs["model_paths"] for member in members] == ["member-0.model", "member-1.model"]
    assert all(member.calculator_kwargs["device"] == "cpu" for member in members)
    assert all(member.finetunable is True for member in members)


def test_ase_engine_single_model_path_has_no_committee_members():
    engine = ASEEngine(
        engine_id="mace",
        calculator=f"{__name__}.DummyCalculator",
        calculator_kwargs={"model_paths": "member-0.model"},
    )

    assert engine.has_committee is False
    assert engine.committee_members == []


def test_to_ams_engine_exports_model_paths_committee_as_hybrid_engine():
    engine = ASEEngine(
        engine_id="mace",
        calculator=f"{__name__}.DummyCalculator",
        calculator_kwargs={
            "model_paths": ["member-0.model", "member-1.model", "member-2.model"],
            "device": "cpu",
            "default_dtype": "float64",
            "model_type": "EnergyDipoleMACE",
        },
        finetunable=True,
    )

    ams_engine = to_ams_engine(engine)
    settings = ams_engine.settings

    assert ams_engine.engine_id == "mace"
    assert ams_engine.finetunable is True
    assert settings.input.Hybrid.Committee.Enabled == "Yes"
    assert settings.input.Hybrid.TweakRequestForSubEngines == "No"
    assert settings.input.Hybrid.Energy.Term == [
        "Factor=0.3333333333333333 Region=* UseCappingAtoms=No EngineID=Engine1",
        "Factor=0.3333333333333333 Region=* UseCappingAtoms=No EngineID=Engine2",
        "Factor=0.3333333333333333 Region=* UseCappingAtoms=No EngineID=Engine3",
    ]
    assert settings.runscript.preamble_lines == ["export OMP_NUM_THREADS=1"]

    subengines = settings.input.Hybrid.Engine
    assert [subengine._h for subengine in subengines] == ["ASE Engine1", "ASE Engine2", "ASE Engine3"]
    assert [subengine.Type for subengine in subengines] == ["Import", "Import", "Import"]
    assert [subengine.Import for subengine in subengines] == [f"{__name__}.DummyCalculator"] * 3
    assert [subengine.Python.Type for subengine in subengines] == ["Current", "Current", "Current"]
    assert all(
        f"model_paths='member-{index}.model'" in subengine.Arguments for index, subengine in enumerate(subengines)
    )
    assert all("device='cpu'" in subengine.Arguments for subengine in subengines)
    assert all("default_dtype='float64'" in subengine.Arguments for subengine in subengines)
    assert all("model_type='EnergyDipoleMACE'" in subengine.Arguments for subengine in subengines)

    ams_input = AMSJob(settings=settings).get_input()
    assert "Engine Hybrid" in ams_input
    assert "Engine ASE Engine1" in ams_input
    assert "Engine ASE Engine2" in ams_input
    assert "Engine ASE Engine3" in ams_input
    assert "Committee\n    Enabled Yes" in ams_input


def test_ase_engine_settings_include_python_current_by_default():
    engine = ASEEngine(engine_id="ase", calculator=f"{__name__}.DummyCalculator", calculator_kwargs={"alpha": 1})

    settings = engine.settings

    assert settings.input.ase.Type == "Import"
    assert settings.input.ase.Python.Type == "Current"
    assert settings.input.ase.Import == f"{__name__}.DummyCalculator"
    assert "alpha=1" in settings.input.ase.Arguments


def test_ase_engine_python_settings_round_trip_through_model_and_settings():
    engine = ASEEngine(
        engine_id="ase",
        calculator=f"{__name__}.DummyCalculator",
        python={"type": "Current"},
    )

    assert engine.python == ASEPythonSettings(type="Current")
    assert engine.model_dump(mode="json")["python"] == {"type": "Current", "info": None}
    assert engine.settings.input.ase.Python.Type == "Current"


def test_validate_properties_rejects_unknown():
    engine = ASEEngine(engine_id="ase", calculator=f"{__name__}.DummyCalculator")

    with pytest.raises(ValueError):
        engine._validate_properties([PropertyInfo(name="unknown_property")])


def test_resolve_calculator_class_requires_fqcn():
    engine = ASEEngine(engine_id="ase", calculator="NotAPath")

    with pytest.raises(ValueError, match="fully qualified class path"):
        engine._resolve_calculator_class()


def test_resolve_calculator_class_missing_class():
    engine = ASEEngine(engine_id="ase", calculator="types.NoSuchClass")

    with pytest.raises(ValueError, match="Cannot find calculator class"):
        engine._resolve_calculator_class()


def test_run_single_point_computes_results_and_caches_canonical_property(monkeypatch):
    engine = ASEEngine(engine_id="ase", calculator=f"{__name__}.DummyCalculator")
    atoms = DummyAtoms()
    seen_systems = []

    def fake_to_ase_atoms(system):
        seen_systems.append(system)
        return atoms

    monkeypatch.setattr(ASEEngine, "_to_ase_atoms", staticmethod(fake_to_ase_atoms))
    props = [PropertyInfo(name="energy"), PropertyInfo(name="potential_energy"), PropertyInfo(name="free_energy")]
    dataset = [SimpleNamespace(chemical_system="system-1")]

    results = list(engine.run_single_point(props, dataset, SerialStrategy()))

    assert results == [{"energy": 1.0, "potential_energy": 1.0, "free_energy": 2.0}]
    assert atoms.energy_calls == 2
    assert seen_systems == ["system-1"]
    assert isinstance(atoms.calc, DummyCalculator)


def test_run_single_point_reports_failure(monkeypatch):
    engine = ASEEngine(engine_id="ase", calculator=f"{__name__}.DummyCalculator")

    def boom(_system):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(ASEEngine, "_to_ase_atoms", staticmethod(boom))
    props = [PropertyInfo(name="energy")]

    results = list(engine.run_single_point(props, [SimpleNamespace(chemical_system="x")], SerialStrategy()))

    assert results == [{"FAILURE": "conversion failed"}]


def test_extract_system_prefers_ase_atoms_before_unavailable_chemical_system():
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms("Ar")

    extracted = extract_system(ASEOnlyRow(atoms))

    assert extracted is atoms


def test_run_single_point_with_ase_lennard_jones():
    ase = pytest.importorskip("ase")
    lj = pytest.importorskip("ase.calculators.lj")

    atoms = ase.Atoms("Ar2", positions=[(0.0, 0.0, 0.0), (3.5, 0.0, 0.0)])
    expected_atoms = atoms.copy()
    expected_atoms.calc = lj.LennardJones()
    expected_energy = expected_atoms.get_potential_energy()

    engine = ASEEngine(engine_id="ase", calculator="ase.calculators.lj.LennardJones")
    props = [PropertyInfo(name="energy")]
    dataset = [SimpleNamespace(chemical_system=atoms)]

    results = list(engine.run_single_point(props, dataset, SerialStrategy()))

    assert results == [{"energy": pytest.approx(expected_energy)}]
