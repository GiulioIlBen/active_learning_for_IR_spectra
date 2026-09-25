from __future__ import annotations

import pytest

pytest.importorskip("scm.conformers", reason="requires optional dependency scm.conformers")

from scm.active_learning.results.task.ams_conformers import AMSConformersResults


def test_ams_conformers_results_type_default():
    assert AMSConformersResults.model_fields["type"].default == "AMSConformersResults"
