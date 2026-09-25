import operator
from typing import Iterable, List, Literal, Tuple, Union

import numpy as np

from scm.moliterate.core import BaseChemDataSet, BaseFilter, ChemDataEntry
from scm.moliterate.transforms import UnionRowTransform
from scm.moliterate.transforms.atoms_stats import (
    MinMaxAtomsDistance,
    PropertyTransform,
)


class ConditionsFilter(BaseFilter):
    """
    conditions are list of strings:
        with a string like 'key=value', where '=' can also be one of
                    '<=', '<', '>', '>=' or '!='.
        special key values are in self._map_prop_trn: @min_distance, @max_distance or @fmax
    """

    type: Literal["ConditionsFilter"] = "ConditionsFilter"  # Field discriminator!
    conditions: Union[List[Union[str, Tuple[UnionRowTransform, str]]], str]
    combine_conditions_fun: Literal["all", "any"] = "all"
    collect_from: Literal["metadata", "properties", "properties_first"] = "properties"
    default: bool = False

    _operator_symbols = ("<=", ">=", "!=", "<", ">", "=")
    _operators = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "=": operator.eq,
        "!=": operator.ne,
    }

    @property
    def _map_prop_trn(self):
        return {
            "@min_distance": MinMaxAtomsDistance(min_out_key="min_distance", max_out_key=None),
            "@max_distance": MinMaxAtomsDistance(min_out_key=None, max_out_key="max_distance"),
            "@fmax": PropertyTransform(property_key="forces", post_process="l2_max", property_key_out="fmax"),
        }

    @property
    def _combine_fn(self):
        return all if self.combine_conditions_fun == "all" else any

    def parsed_conditions(self):
        if isinstance(self.conditions, list):
            return self.conditions
        return self.conditions.split(",")

    def apply_filter(self, db: BaseChemDataSet) -> Iterable[int]:
        """return RELATIVE indices of the dataset in input"""
        combine = self._combine_fn
        conditions = self.parsed_conditions()

        matched_indices: List[int] = []
        for idx, row in enumerate(db):
            if combine(self.eval_condition(condition, row) for condition in conditions):
                matched_indices.append(idx)
        return np.array(matched_indices)

    def eval_condition(self, condition: Union[str, Tuple[UnionRowTransform, str]], row: ChemDataEntry):
        trn, selection = self._extract_trn_selection(condition)
        op_symbol, key, value = self._parse_selection(selection)
        row, key = self._trn_with_special(row, trn, key)

        if self.collect_from == "properties_first":
            val_left = row.properties.get(key, row.metadata.get(key))
        elif self.collect_from == "metadata":
            val_left = row.metadata.get(key)
        elif self.collect_from == "properties":
            val_left = row.properties.get(key)
        else:
            raise ValueError(f"{self.collect_from=} not valid. ['properties_first','metadata','properties']")

        if val_left is None:
            return self.default

        if isinstance(val_left, np.ndarray):
            if val_left.size != 1:
                return self.default
            val_left = val_left.item()

        if isinstance(val_left, (np.bool_, bool)):
            val_right = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(val_left, (int, float, np.integer, np.floating)):
            try:
                val_right = float(value)
            except ValueError:
                return self.default
            val_left = float(val_left)
        else:
            val_right = value
        return bool(self._operators[op_symbol](val_left, val_right))

    def _trn_with_special(self, row, trn, key: str):
        if "@" in key and trn is None:
            trn = self._map_prop_trn[key]
            key = key.replace("@", "")
        if trn is not None:
            row = trn(row)
        return row, key

    def _parse_selection(self, selection: str):
        op_symbol = None
        key = None
        value = None
        for symbol in self._operator_symbols:
            if symbol in selection:
                key, value = selection.split(symbol, 1)
                op_symbol = symbol
                break
        if op_symbol is None or key is None or value is None:
            raise ValueError(f"Invalid selection string: {selection!r}")
        key = key.strip()
        value = value.strip()
        return op_symbol, key, value

    def _extract_trn_selection(self, condition: Union[str, Tuple[UnionRowTransform, str]]):
        if isinstance(condition, str):
            trn = None
            selection = condition
        else:
            trn, selection = condition
        selection = selection.strip()
        return trn, selection

    def __str__(self) -> str:
        return f"{self.__class__}(conditions={self.conditions}, combine_conditions_fun={self.combine_conditions_fun})"

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()
