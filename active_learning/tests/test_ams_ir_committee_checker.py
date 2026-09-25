from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scm.active_learning.checker_getters.checkers import AMSIRCommitteeAgreementChecker
from scm.active_learning.checker_getters.checkers.ams_ir_committee_checker import HarmonicIRSpectrum


def _spectrum(freqs, ints, engine="m") -> HarmonicIRSpectrum:
    return HarmonicIRSpectrum(engine=engine, frequencies=np.asarray(freqs, float), intensities=np.asarray(ints, float))


def test_identical_spectra_have_cos_ir_one():
    spectrum = _spectrum([1000.0, 1700.0, 3000.0], [10.0, 50.0, 5.0])

    assert spectrum.cos_ir(spectrum, np.arange(0.0, 4001.0), 30.0) == pytest.approx(1.0)


def test_disjoint_spectra_have_low_cos_ir():
    a = _spectrum([800.0], [10.0])
    b = _spectrum([3000.0], [10.0])

    assert a.cos_ir(b, np.arange(0.0, 4001.0), 30.0) < 0.01


def test_broadened_is_a_normalised_lorentzian_and_skips_imaginary_modes():
    grid = np.arange(-20000.0, 20001.0, 0.5)
    spectrum = _spectrum([-200.0, 1500.0], [99.0, 7.0])

    area = np.trapezoid(spectrum.broadened(grid, 30.0), grid)

    assert area == pytest.approx(7.0, rel=1e-2)


def test_committee_agreement_is_mean_member_cos_ir():
    checker = AMSIRCommitteeAgreementChecker()
    committee = _spectrum([1000.0, 1700.0], [10.0, 50.0], engine="hybrid")
    same = _spectrum([1000.0, 1700.0], [10.0, 50.0])
    shifted = _spectrum([1300.0, 2200.0], [10.0, 50.0])

    agreement = checker.committee_agreement(committee, [same, shifted])

    expected = np.mean([same.cos_ir(committee, checker.grid, 30.0), shifted.cos_ir(committee, checker.grid, 30.0)])
    assert agreement == pytest.approx(expected)
    assert agreement < checker.min_committee_agreement


def test_imaginary_indices_use_threshold():
    spectrum = _spectrum([-50.0, -5.0, 100.0], [1.0, 1.0, 1.0])

    assert list(spectrum.imaginary_indices(-10.0)) == [0]


def test_load_spectra_registers_hybrid_member_rkfs(tmp_path, monkeypatch):
    from scm.active_learning.checker_getters.checkers import ams_ir_committee_checker as checker_module

    for name in ("hybrid-term1-ASE.rkf", "hybrid-term2-ASE.rkf", "hybrid-term3-ASE.rkf"):
        (tmp_path / name).touch()

    class FakeResults:
        def __init__(self):
            self.job = SimpleNamespace(path=tmp_path)
            self.rkfs = {"hybrid": object()}

        def collect_rkfs(self):
            pass

        def get_main_engine_name(self):
            return "hybrid"

        def engine_names(self):
            return list(self.rkfs)

        def get_frequencies(self, engine):
            return [1000.0]

        def get_ir_intensities(self, engine):
            return [10.0]

    kf_calls = []
    monkeypatch.setattr(checker_module, "KFFile", lambda path: kf_calls.append(path) or object())

    committee, members = AMSIRCommitteeAgreementChecker.load_spectra(FakeResults())

    assert committee.engine == "hybrid"
    assert [member.engine for member in members] == [
        "hybrid-term1-ASE",
        "hybrid-term2-ASE",
        "hybrid-term3-ASE",
    ]
    assert kf_calls == [str(tmp_path / f"hybrid-term{i}-ASE.rkf") for i in range(1, 4)]


def test_missing_normal_modes_is_a_failed_check_not_a_crash(monkeypatch):
    checker = AMSIRCommitteeAgreementChecker()

    def _raise(_results):
        raise KeyError("Vibrations")

    monkeypatch.setattr(AMSIRCommitteeAgreementChecker, "load_spectra", staticmethod(_raise))
    monkeypatch.setattr(
        AMSIRCommitteeAgreementChecker,
        "collect_ids",
        lambda self, result: dict(engine_id="e", task_id="go", system_id="M0000", checker_id=self.checker_id),
    )

    checks = checker.run(SimpleNamespace(plams_results=None))

    assert len(checks) == 1
    assert checks[0].success == "IRSpect_missing"
