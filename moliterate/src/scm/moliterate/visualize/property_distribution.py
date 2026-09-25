from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from scm.moliterate.core.base_chem_dataset import BaseChemDataSet, IterDataKeyError
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.transforms import PropertyTransform

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


PropertySelector = Union[str, PropertyTransform]


def _import_matplotlib_pyplot():
    try:
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is not installed. Install the visualization extras with `pip install 'moliterate[viz]'`."
        ) from exc
    return plt


def _property_info(dataset: BaseChemDataSet, property_name: str) -> Optional[PropertyInfo]:
    for prop in dataset.out_properties:
        if prop.name == property_name:
            return prop
    return None


def _property_label(property_name: str, property_info: Optional[PropertyInfo]) -> str:
    if property_info is None or property_info.unit in (None, "NotUniform"):
        return property_name
    return f"{property_name} [{property_info.unit}]"


def _flatten_numeric_values(
    values: Union[np.ndarray, Sequence[float], Iterable[float]], *, drop_non_finite: bool = True
) -> np.ndarray:
    if isinstance(values, np.ndarray):
        numeric_values = np.asarray(values, dtype=float).reshape(-1)
    else:
        numeric_values = np.asarray(list(values), dtype=float).reshape(-1)
    if drop_non_finite:
        numeric_values = numeric_values[np.isfinite(numeric_values)]
    return numeric_values


def collect_property_values(
    dataset: BaseChemDataSet,
    property_name: str,
    *,
    transform: Optional[PropertyTransform] = None,
    drop_non_finite: bool = True,
) -> np.ndarray:
    """Collect a property across the dataset as a 1D float array."""
    collected = []
    for idx, row in enumerate(dataset):
        row_to_read = copy.deepcopy(row)
        if transform is not None:
            row_to_read = transform(row_to_read)
            property_key = transform.property_key_out
        else:
            property_key = property_name

        if property_key not in row_to_read.properties:
            raise IterDataKeyError(f"Property '{property_key}' not found in row {idx}.")
        try:
            collected.append(np.asarray(row_to_read.properties[property_key], dtype=float).reshape(-1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Property '{property_key}' in row {idx} cannot be converted to float.") from exc
    if not collected:
        return np.asarray([], dtype=float)
    values = np.concatenate(collected)
    if drop_non_finite:
        values = values[np.isfinite(values)]
    return values


class PropertyDistributionPlot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    properties: Optional[List[PropertySelector]] = None
    bins: Union[int, str, Sequence[float]] = 30
    density: bool = False
    drop_non_finite: bool = True
    max_cols: int = 3
    figsize_per_plot: Tuple[float, float] = (4.5, 3.5)
    sharey: bool = False
    title: Optional[str] = None
    hist_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("properties", mode="before")
    @classmethod
    def _normalize_properties(cls, value):
        if value is None:
            return None
        if isinstance(value, (str, PropertyTransform)):
            return [value]
        return list(value)

    @field_validator("max_cols")
    @classmethod
    def _validate_max_cols(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_cols must be >= 1.")
        return value

    def _resolve_selectors(self, dataset: BaseChemDataSet) -> List[PropertySelector]:
        if self.properties is not None:
            return list(self.properties)
        return [prop.name for prop in dataset.out_properties]

    def _selector_name(self, selector: PropertySelector) -> str:
        if isinstance(selector, str):
            return selector
        return selector.property_key_out

    def _selector_label(self, dataset: BaseChemDataSet, selector: PropertySelector) -> str:
        if isinstance(selector, str):
            return _property_label(selector, _property_info(dataset, selector))
        unit = None
        if selector.unit_transform is not None:
            unit = selector.unit_transform[1]
        info = PropertyInfo(name=selector.property_key_out, unit=unit)
        return _property_label(selector.property_key_out, info)

    def _selector_values(self, dataset: BaseChemDataSet, selector: PropertySelector) -> np.ndarray:
        if isinstance(selector, str):
            return collect_property_values(dataset, selector, drop_non_finite=self.drop_non_finite)
        return collect_property_values(
            dataset,
            selector.property_key,
            transform=selector.model_copy(deep=True),
            drop_non_finite=self.drop_non_finite,
        )

    def run(self, dataset: BaseChemDataSet) -> Tuple["Figure", np.ndarray]:
        selectors = self._resolve_selectors(dataset)
        if not selectors:
            raise ValueError("No properties available to plot.")

        values_by_selector = [self._selector_values(dataset, selector) for selector in selectors]
        for selector, values in zip(selectors, values_by_selector):
            if values.size == 0:
                raise ValueError(f"No values available to plot for '{self._selector_name(selector)}'.")

        plt = _import_matplotlib_pyplot()
        n_plots = len(selectors)
        n_cols = min(self.max_cols, n_plots)
        n_rows = math.ceil(n_plots / n_cols)
        figsize = (self.figsize_per_plot[0] * n_cols, self.figsize_per_plot[1] * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharey=self.sharey)

        hist_options = dict(self.hist_kwargs)
        hist_options.setdefault("color", "#4C78A8")
        hist_options.setdefault("edgecolor", "white")
        hist_options.setdefault("linewidth", 0.8)

        axes_flat = axes.reshape(-1)
        for axis, selector, values in zip(axes_flat, selectors, values_by_selector):
            label = self._selector_label(dataset, selector)
            axis.hist(values, bins=self.bins, density=self.density, **hist_options)
            axis.set_title(label)
            axis.set_xlabel(label)
            axis.set_ylabel("Density" if self.density else "Count")
            axis.grid(axis="y", alpha=0.2)

        for axis in axes_flat[n_plots:]:
            axis.remove()

        if self.title is not None:
            fig.suptitle(self.title)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        else:
            fig.tight_layout()
        return fig, axes

    def __call__(self, dataset: BaseChemDataSet) -> Tuple["Figure", np.ndarray]:
        return self.run(dataset)
