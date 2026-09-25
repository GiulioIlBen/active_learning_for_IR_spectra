import inspect
from typing import Any, Dict, Literal, Optional

import numpy as np
from pydantic import ConfigDict, field_serializer, model_validator

from scm.moliterate.core import BaseChemDataSet, BaseFilter
from scm.moliterate.partition_criteria import UnionPartitionCriterion


class ApricotFilter(BaseFilter):
    """
    interface of: https://github.com/jmschrei/apricot
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["ApricotFilter"] = "ApricotFilter"  # Field discriminator!
    n_samples: int
    partition_criterion: UnionPartitionCriterion
    selection_strategy: Literal[
        "FacilityLocationSelection",
        "FeatureBasedSelection",
        "MaxCoverageSelection",
        "SaturatedCoverageSelection",
        "SumRedundancySelection",
        "GraphCutSelection",
        "MixtureSelection",
    ] = "FacilityLocationSelection"
    # initial_subset:Optional[List[int]]=None
    # optimizer:str='lazy'
    # optimizer_kwds:dict={}
    # reservoir:Optional[np.ndarray]=None
    # max_reservoir_size:int=1000
    n_jobs: int = 1
    random_state: Optional[int] = None
    verbose: bool = False
    cls_kwargs: Dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _validate_cls_kwargs(cls, data):
        if not isinstance(data, dict):
            return data

        value = data.get("cls_kwargs")
        if value is None:
            data["cls_kwargs"] = {}
            return data
        if not isinstance(value, dict):
            raise TypeError("cls_kwargs must be a dictionary.")

        strategy = data.get("selection_strategy", cls.model_fields["selection_strategy"].default)
        cls.is_installed()
        import apricot

        if not hasattr(apricot, strategy):
            raise ValueError(f"Invalid apricot selection strategy: {strategy}")
        data["cls_kwargs"] = value
        return data

    @field_serializer("cls_kwargs", when_used="json")
    def _cls_kwargs_serializer(self, value):
        return {key: (val.tolist() if isinstance(val, np.ndarray) else val) for key, val in value.items()}

    @model_validator(mode="after")
    def _validate_apricot_kwargs(self):
        import apricot

        strategy = self.selection_strategy
        value = self.cls_kwargs
        if not hasattr(apricot, strategy):
            raise ValueError(f"Invalid apricot selection strategy: {strategy}")
        apricot_cls = getattr(apricot, strategy)
        signature = inspect.signature(apricot_cls.__init__)
        disallowed = {"self", "__class__", "n_samples", "n_jobs", "random_state", "verbose"}
        allowed_params = {name for name, param in signature.parameters.items() if name not in disallowed}
        has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        # Some apricot classes expose kwargs that are not reflected in the signature.
        # Keep a small allowlist to avoid false negatives while still catching typos.
        extra_allowed = {"reservoir", "max_reservoir_size"}
        if has_var_kwargs:
            invalid_kwargs = set()
        else:
            invalid_kwargs = set(value) - allowed_params - extra_allowed
        if invalid_kwargs:
            raise ValueError(f"Invalid cls_kwargs for {strategy}: {sorted(invalid_kwargs)} | {allowed_params=}")
        return self

    def get_selector(self):
        import apricot

        cls = getattr(apricot, self.selection_strategy)
        return cls(
            n_samples=self.n_samples,
            # initial_subset=self.initial_subset,
            # optimizer=self.optimizer,
            # optimizer_kwds=self.optimizer_kwds,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.cls_kwargs,
        )

    def apply_filter(self, db: BaseChemDataSet) -> np.ndarray:
        selector = self.get_selector()
        features = self.partition_criterion(db)
        selector.fit(features)
        return selector.ranking

    @staticmethod
    def is_installed():
        try:
            import apricot  # noqa F401
        except ImportError:
            raise ImportError("apricot is not installed: use `pip install apricot-select`")
