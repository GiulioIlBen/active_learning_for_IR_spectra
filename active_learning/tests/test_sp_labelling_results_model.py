from __future__ import annotations

from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.results.sp_labelling import SPLabellingResults


def test_sp_labelling_results_defaults():
    dataset = InMemoryMolData.create()

    results = SPLabellingResults(dataset=dataset)

    assert results.dataset is dataset
    assert results.failed_simulations == set()
