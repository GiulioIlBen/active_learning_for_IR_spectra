from __future__ import annotations

import math
from typing import Any, ClassVar, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, PrivateAttr, model_validator

from scm.moliterate.core import BaseChemDataSet, PropertyInfo
from scm.moliterate.partition_criteria import MetadataCriterion
from scm.moliterate.utils.unit_conversion import conversion_ratio

MetricName = Literal["mae", "rmse", "mse", "max_abs", "mean_error", "max_scaled_abs"]


class _PairwiseCommon(BaseModel):
    property: str
    metric: MetricName
    per_n_atoms: bool = False
    # ``atom_type`` is an atomic number to select; ``components`` is an array axis.
    target: float = 0.0
    lower_is_better: bool = True
    no_data_is_success: bool = False

    def _common_result_fields(self) -> Dict[str, Any]:
        return self.model_dump(
            include={
                "property",
                "metric",
                "per_n_atoms",
                "target",
                "lower_is_better",
                "no_data_is_success",
            }
        )


class PairwiseResult(_PairwiseCommon):
    units: str
    # default grouping: per system/atom type/component x,y,z
    atom_type: int = -1
    components: int = -1
    value: float
    success: Union[Literal["OK", "FAIL", "NoData"], str] = ""
    n_entries: int
    group_info: Dict[str, str] = {}

    @model_validator(mode="after")
    def _validate_success(self):
        if self.success != "":
            return self
        if self.n_entries == 0:
            self.success = "NoData"
            return self

        passes_target = self.value < self.target if self.lower_is_better else self.value > self.target
        if passes_target:
            self.success = "OK"
        else:
            self.success = "FAIL"
        return self

    def compact_dump(self, **kwargs):
        ret = self.model_dump(**kwargs)
        pop_list = ["target", "lower_is_better", "per_n_atoms", "no_data_is_success"]
        if self.atom_type == -1:
            pop_list.append("atom_type")
        if self.components == -1:
            pop_list.append("components")
        if self.per_n_atoms:
            ret["units"] += "/NAtoms"
        for p in pop_list:
            ret.pop(p, None)
        return ret


class _NPMetricAccumulator:
    """
    Responsible for accumulating diffs based on the chosen metric,
    but the actual diff calculation is done at PropMetricEvaluator level
    """

    def __init__(self) -> None:
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum = 0.0
        self.max_abs = 0.0
        self.count = 0

    def update(self, diff: np.ndarray) -> None:
        if diff.size == 0:
            return
        flat = diff.ravel()
        abs_flat = np.abs(flat)
        self.sum_abs += float(abs_flat.sum())
        self.sum_sq += float((flat * flat).sum())
        self.sum += float(flat.sum())
        self.max_abs = max(self.max_abs, float(abs_flat.max(initial=0.0)))
        self.count += int(flat.size)

    def metric_value(self, metric: MetricName) -> float:
        if self.count == 0:
            return math.nan
        if metric == "mae":
            return self.sum_abs / self.count
        if metric == "mse":
            return self.sum_sq / self.count
        if metric == "rmse":
            return math.sqrt(self.sum_sq / self.count)
        if metric in ["max_abs", "max_scaled_abs"]:
            return self.max_abs
        if metric == "mean_error":
            return self.sum / self.count
        raise ValueError(f"Unsupported metric: {metric}")


class PropMetricEvaluator(_PairwiseCommon):
    # TODO: Implement magnitude and cosine comparison for vectorial quantities:
    # for cosine we should change _np_property_diff method.
    # For Magnitude maybe I should not implement here but rely on property transform to provide the magnitude.
    # default grouping: per system/atom type/component x,y,z
    atom_type: int = -1
    components: int = -1  # TODO: I could add option for diagonal and off-diagonal terms inspection.
    target_units: Optional[str] = None
    transform_settings: Dict[str, Dict[str, Union[str, int, float]]] = {}
    _accumulators: Dict[Tuple[int, int], _NPMetricAccumulator] = PrivateAttr(default_factory=dict)
    _current_relative_idx: Optional[int] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_model(self):
        if self.metric == "max_scaled_abs":
            self.transform_settings["max_scaled_abs"] = {
                "L": 3.0,
                "x0": 7.0,
                "k": 0.5,
            }
        return self

    def reset_accumulators(self) -> None:
        self._accumulators.clear()
        self._current_relative_idx = None

    def accumulate(
        self,
        value_ref,
        value_cmp,
        n_atoms: Optional[int] = None,
        atomic_numbers: Optional[NDArray] = None,
    ) -> None:
        diff = self._np_property_diff(
            value_ref,
            value_cmp,
            n_atoms,
        )
        for atom_type, component, group_diff in self._group_diffs(diff, atomic_numbers):
            key = (atom_type, component)
            if key not in self._accumulators:
                acc = _NPMetricAccumulator()
                self._accumulators[key] = acc
            self._accumulators[key].update(group_diff)

    def build_results(
        self,
        units_map: Mapping[str, str],
    ) -> List[PairwiseResult]:
        units, conversion = self.get_unit_conversion(units_map)

        common_fields = self._common_result_fields()

        if not self._accumulators:
            return [
                PairwiseResult(
                    **common_fields,
                    units=units,
                    value=math.nan,
                    n_entries=0,
                    atom_type=-1,
                    components=-1,
                )
            ]

        results: List[PairwiseResult] = []
        for atom_type, component in sorted(self._accumulators, key=lambda item: (item[0], item[1])):
            acc = self._accumulators[(atom_type, component)]
            results.append(
                PairwiseResult(
                    **common_fields,
                    units=units,
                    value=acc.metric_value(self.metric) * conversion,
                    n_entries=acc.count,
                    atom_type=atom_type,
                    components=component,
                )
            )
        return results

    def get_unit_conversion(self, units_map: Mapping[str, str]):
        source_units = units_map.get(self.property, "")
        target_units = self.target_units
        units = source_units
        conversion = 1.0
        if target_units is not None and source_units is not None and target_units != source_units:
            conversion = conversion_ratio(source_units, target_units)
            units = target_units
        if self.metric == "mse":
            conversion *= conversion
        return units, conversion

    def _group_diffs(
        self,
        diff: NDArray,
        atomic_numbers: Optional[NDArray],
    ) -> Iterable[Tuple[int, int, np.ndarray]]:
        # Sanity check
        if self.components >= diff.ndim:
            raise ValueError(
                f"components axis {self.components} out of bounds for diff with ndim {diff.ndim} "
                f"{self._idx_context_error_msg()}"
            )
        if self.atom_type < 0 and self.components >= 0 and atomic_numbers is not None:
            if diff.shape[self.components] == atomic_numbers.shape[0]:
                raise ValueError(
                    f"components axis {self.components} size {diff.shape[self.components]} "
                    f"matches number of atoms {atomic_numbers.shape[0]} {self._idx_context_error_msg()}"
                )

        # Atom types handling. ``atom_type`` is an atomic number, not an array
        # axis. Atom-resolved properties use the first axis whose size matches
        # the number of atoms (for example, forces have shape ``(n_atoms, 3)``).
        if self.atom_type < 0 or atomic_numbers is None:
            atom_groups = [(-1, diff)]
        else:
            atom_axes = [axis for axis, size in enumerate(diff.shape) if size == atomic_numbers.shape[0]]
            if not atom_axes:
                raise ValueError(
                    f"no axis in diff shape {diff.shape} matches number of atoms {atomic_numbers.shape[0]} "
                    f"for atom_type {self.atom_type} {self._idx_context_error_msg()}"
                )
            atom_axis = atom_axes[0]
            indices = np.nonzero(atomic_numbers == self.atom_type)[0]
            atom_groups = [(self.atom_type, np.take(diff, indices, axis=atom_axis))]

        # Components types handling
        if self.components < 0:
            for atom_type, group_diff in atom_groups:
                yield atom_type, -1, group_diff
            return

        for atom_type, group_diff in atom_groups:
            for component in range(group_diff.shape[self.components]):
                comp_diff = np.take(group_diff, [component], axis=self.components)
                yield atom_type, int(component), comp_diff

    def _np_property_diff(
        self,
        value_ref,
        value_cmp,
        n_atoms: Optional[int],
    ) -> np.ndarray:
        arr_ref = self._to_np_array(value_ref)
        arr_cmp = self._to_np_array(value_cmp)
        diff = arr_cmp - arr_ref
        if self.metric == "max_scaled_abs":
            diff = np.abs(diff) - self._scale_reference(arr_ref)
        if n_atoms is not None:
            diff = diff / max(n_atoms, 1)
        return diff

    def _to_np_array(self, value) -> NDArray:
        try:
            return np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Value must be numeric, got {type(value)!r} {self._idx_context_error_msg()}") from exc

    def _idx_context_error_msg(self):
        return f"| Property '{self.property}' | At relative index: {self._current_relative_idx}"

    def _scale_reference(self, reference: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """
        Transforms/scale the reference using a Logistic/sigmoid function:
        it starts from 0 with max in x0.

        .. code-block:: python

            # Defaults:
            L = 3.0
            x0 = 7.0
            k = 0.5
            return (
                L / (1 + np.exp(-k * (np.abs(reference) - x0)))
                - L / (1 + np.exp(-k * (-x0)))
            )

        :param reference: values to be scaled
        :type reference: Union[float, NDArray[np.float64]]
        :return: Array with same shape as ``reference`` but scaled down for large components
        :rtype: NDArray[np.float64]
        """
        L = float(self.transform_settings["max_scaled_abs"]["L"])
        x0 = float(self.transform_settings["max_scaled_abs"]["x0"])
        k = float(self.transform_settings["max_scaled_abs"]["k"])
        return L / (1 + np.exp(-k * (np.abs(reference) - x0))) - L / (1 + np.exp(-k * (-x0)))


class PairwiseDatasetMetrics(BaseModel):
    """TBN: units are handled considering uniformity in the dataset!"""

    settings: List[PropMetricEvaluator] = []

    PropMetricEv: ClassVar[Type[PropMetricEvaluator]] = PropMetricEvaluator

    @classmethod
    def from_dataset(
        cls,
        dataset: BaseChemDataSet,
        metrics: Sequence[MetricName] = ("mae",),
        per_n_atoms: Optional[Sequence[str]] = ("energy",),
    ):
        per_n_atoms_list = per_n_atoms or []
        settings: List[PropMetricEvaluator] = []
        for prop in dataset.out_properties:
            for metric in metrics:
                settings.append(
                    PropMetricEvaluator(
                        property=prop.name,
                        metric=metric,
                        per_n_atoms=prop.name in per_n_atoms_list,
                    )
                )
        return cls(settings=settings)

    def compare_two_grouped(
        self, dataset_ref: BaseChemDataSet, dataset_cmp: BaseChemDataSet, metadata_grouping: List[str]
    ) -> List[PairwiseResult]:
        md_criterion = MetadataCriterion(metadata_keys=metadata_grouping, concat_symbol="^%^")
        ref_groups = md_criterion.group_by(md_criterion(dataset_ref))
        cmp_groups = md_criterion.group_by(md_criterion(dataset_cmp))
        ret = []
        for k in ref_groups:
            dataset_ref_k = dataset_ref.subset(ref_groups[k])
            dataset_cmp_k = dataset_cmp.subset(cmp_groups[k])
            group_info = dict(zip(metadata_grouping, k.split(md_criterion.concat_symbol)))
            res = self.compare_two(dataset_ref_k, dataset_cmp_k)
            for r in res:
                r.group_info = group_info
            ret.extend(res)
        return ret

    def compare_two(
        self,
        dataset_ref: BaseChemDataSet,
        dataset_cmp: BaseChemDataSet,
    ) -> List[PairwiseResult]:
        if not self.settings:
            return []
        if len(dataset_ref) != len(dataset_cmp):
            raise ValueError("Datasets must have the same length to compare properties.")

        for setting in self.settings:
            setting.reset_accumulators()

        for i, (row_ref, row_cmp) in enumerate(zip(dataset_ref, dataset_cmp)):
            for setting in self.settings:
                setting._current_relative_idx = i

                prop = setting.property
                if prop not in row_ref.properties or prop not in row_cmp.properties:
                    continue
                prop_ref = row_ref.properties[prop]
                prop_cmp = row_cmp.properties[prop]
                n_atoms = None
                if setting.per_n_atoms:
                    n_atoms = len(row_ref.atoms)
                atomic_numbers = None
                if setting.atom_type >= 0 or (setting.atom_type < 0 and setting.components >= 0):
                    atomic_numbers = row_ref.atoms.get_atomic_numbers()
                setting.accumulate(
                    prop_ref,
                    prop_cmp,
                    n_atoms,
                    atomic_numbers,
                )

        props = [x.property for x in self.settings]
        units_map: Dict[str, str] = {p.name: p.unit or "" for p in dataset_ref.out_properties if p.name in props}
        results: List[PairwiseResult] = []
        for setting in self.settings:
            results.extend(setting.build_results(units_map))
        return results

    def _property_info_maps(
        self, props: Iterable[PropertyInfo]
    ) -> Tuple[Dict[str, str], Dict[str, Optional[Union[Tuple[Union[int, str], ...], str]]]]:
        units_map: Dict[str, str] = {}
        shapes_map: Dict[str, Optional[Union[Tuple[Union[int, str], ...], str]]] = {}
        for prop in props:
            units_map[prop.name] = "" if prop.unit is None else str(prop.unit)
            shapes_map[prop.name] = prop.shape
        return units_map, shapes_map
