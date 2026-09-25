from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces import InMemoryMolData

from scm.active_learning.checker_getters.checkers import AMSFrequenciesChecker, AMSTrajChecker, NoneChecker
from scm.active_learning.checker_getters.getters import AMSTrajGetter, NMSGetters
from scm.active_learning.checker_getters.getters.nms_getter import (
    NMSData,
    NMSFrequenciesNotFoundError,
)


def _result_with_frequencies(frequencies):
    plams_results = SimpleNamespace(get_frequencies=lambda engine=None: frequencies)
    return SimpleNamespace(
        plams_results=plams_results,
        from_task=SimpleNamespace(task_id="go", system_id="mol"),
        from_engine=SimpleNamespace(engine_id="model"),
    )


def test_frequencies_checker_rejects_results_without_modes_above_cutoff():
    checker = AMSFrequenciesChecker(min_wavenumber=500.0)

    result = checker.run(_result_with_frequencies([-20.0, 10.0, 115.0]))[0]

    assert result.value == 115.0
    assert result.target == 500.0
    assert result.success == "NO_USABLE_FREQUENCIES"


def test_frequencies_checker_accepts_result_with_mode_above_cutoff():
    checker = AMSFrequenciesChecker(min_wavenumber=500.0)

    result = checker.run(_result_with_frequencies([100.0, 501.0]))[0]

    assert result.success == "OK"


def test_nms_data_raises_when_frequency_filter_removes_every_mode():
    data = NMSData(
        atoms_equilibrium=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        force_constants=np.asarray([1.0]),
        normal_modes=np.asarray([[[1.0, 0.0, 0.0]]]),
        frequencies=np.asarray([499.0]),
    )

    with pytest.raises(NMSFrequenciesNotFoundError, match="above 500"):
        data.filter(500)


def test_fractional_trajectory_selector_uses_thirds_and_final_frame():
    dataset = InMemoryMolData.create(available_properties=[])
    for index in range(10):
        dataset.add_system(ChemDataEntry(system=Atoms("H", positions=[[float(index), 0.0, 0.0]])))
    selector = AMSTrajGetter.FractionalTrajectorySelector()

    selected = selector.run(dataset, checker_results=[])

    assert list(selected.absolute_idxs) == [3, 6, 9]


def test_fractional_trajectory_selector_rejects_trajectory_with_one_frame():
    dataset = InMemoryMolData.create(available_properties=[])
    dataset.add_system(ChemDataEntry(system=Atoms("H")))

    with pytest.raises(ValueError, match="at least 2 distinct frames"):
        AMSTrajGetter.FractionalTrajectorySelector().run(dataset, checker_results=[])


def test_checker_getter_chain_supports_trajectory_frequency_and_nms_stages():
    chain = (
        (AMSTrajChecker() + AMSTrajGetter())
        >> (
            AMSFrequenciesChecker()
            + AMSTrajGetter(
                condition_filters=[AMSTrajGetter.FractionalTrajectorySelector()],
                getter_id="fallback",
            )
        )
        >> (NoneChecker() + NMSGetters(max_n_structures=3))
    )

    assert len(chain.check_getters) == 3
