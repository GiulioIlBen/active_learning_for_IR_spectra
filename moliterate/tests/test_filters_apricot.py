import sys
import types
from typing import Literal

import numpy as np
import pytest
from pydantic import ConfigDict, ValidationError

from scm.moliterate.core.partition_criteria import BasePartitionCriteria
from scm.moliterate.filters import ApricotFilter
from scm.moliterate.partition_criteria import MinDistanceCriterion


class ConstantCriterion(BasePartitionCriteria):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["ConstantCriterion"] = "ConstantCriterion"
    output: np.ndarray

    def __call__(self, db):
        return self.output


@pytest.fixture
def fake_apricot(monkeypatch):
    class FacilityLocationSelection:
        def __init__(self, n_samples, metric=None, optimizer="lazy", n_jobs=1, random_state=None, verbose=False):
            self.n_samples = n_samples
            self.metric = metric
            self.optimizer = optimizer
            self.ranking = np.array([], dtype=int)

        def fit(self, features):
            self.ranking = np.arange(min(self.n_samples, len(features)))

    module = types.SimpleNamespace(FacilityLocationSelection=FacilityLocationSelection)
    monkeypatch.setitem(sys.modules, "apricot", module)
    return module


def test_invalid_strategy_raises_in_validate_cls_kwargs(fake_apricot):
    criterion = MinDistanceCriterion()

    with pytest.raises(ValidationError, match="Invalid apricot selection strategy"):
        ApricotFilter(
            n_samples=1,
            partition_criterion=criterion,
            selection_strategy="FeatureBasedSelection",
            cls_kwargs={"metric": "euclidean"},
        )


def test_invalid_strategy_raises_in_validate_apricot_kwargs(fake_apricot):
    criterion = MinDistanceCriterion()

    with pytest.raises(ValidationError, match="Invalid apricot selection strategy"):
        ApricotFilter(
            n_samples=1,
            partition_criterion=criterion,
            selection_strategy="FeatureBasedSelection",
        )


def test_call_returns_ranking(fake_apricot):
    criterion = ConstantCriterion(output=np.array([[0.1], [0.2], [0.3]]))
    flt = ApricotFilter.model_construct(n_samples=2, partition_criterion=criterion)

    ranking = flt.apply_filter(db=object())

    assert ranking.tolist() == [0, 1]


def test_invalid_selection_strategy_raises(fake_apricot):
    criterion = MinDistanceCriterion()

    with pytest.raises(ValidationError):
        ApricotFilter(
            n_samples=1,
            partition_criterion=criterion,
            selection_strategy="NotARealSelector",
        )


def test_invalid_cls_kwargs_raise_clear_error(fake_apricot):
    criterion = MinDistanceCriterion()

    with pytest.raises(ValidationError, match="Invalid cls_kwargs"):
        ApricotFilter(
            n_samples=1,
            partition_criterion=criterion,
            cls_kwargs={"bad_kw": 1},
        )


def test_numpy_kwargs_are_normalized(fake_apricot):
    criterion = MinDistanceCriterion()

    # TODO: I get an error here but do not know why...
    flt = ApricotFilter(n_samples=1, partition_criterion=criterion, cls_kwargs={"reservoir": np.array([1, 2, 3])})
    flt.model_dump_json()
