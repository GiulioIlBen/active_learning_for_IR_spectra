import os
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from scm.moliterate.core.base_chem_dataset import IterDataCoreError
from scm.moliterate.core.base_chem_dataset_writer import BaseChemDataSetWriter
from scm.moliterate.core.concat_chem_dataset import ConcatChemDataSet
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.interfaces import UnionInterfaces, UnionInterfaceWriters
from scm.moliterate.interfaces.ase_database import ASEMolData
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.interfaces.params_data import ParAMSData, ParAMSDataError
from scm.moliterate.interfaces.rkf_files import RKFMolData

PathLike = Union[Path, str]


class ChemDataSetFormat(Enum):
    """Enumeration of data formats"""

    ASE = ASEMolData
    RKF = RKFMolData
    ParAMS = ParAMSData
    IN_MEMORY = InMemoryMolData
    # CONCAT = ConcatChemDataSet


@overload
def create_dataset(
    data_source: PathLike = "",
    fmt: Literal[ChemDataSetFormat.ASE] = ChemDataSetFormat.ASE,
    distance_unit="Ang",
    available_properties: Optional[Sequence[Union[PropertyInfo, Dict]]] = None,
    **kwargs,
) -> ASEMolData: ...


@overload
def create_dataset(
    data_source: PathLike = "",
    fmt: Literal[ChemDataSetFormat.IN_MEMORY] = ChemDataSetFormat.IN_MEMORY,
    distance_unit="Ang",
    available_properties: Optional[Sequence[Union[PropertyInfo, Dict]]] = None,
    **kwargs,
) -> InMemoryMolData: ...


def create_dataset(
    data_source: PathLike = "",
    fmt: Literal[ChemDataSetFormat.ASE, ChemDataSetFormat.IN_MEMORY] = ChemDataSetFormat.ASE,
    distance_unit="Ang",
    available_properties: Optional[Sequence[Union[PropertyInfo, Dict]]] = None,
    **kwargs,
) -> UnionInterfaceWriters:
    """
    Create a new atoms dataset.

    Args:
        data_source: file path for the dataset (ignored for in-memory datasets)
        format: atoms data format to create
        available_properties: optional explicit list of `PropertyInfo` entries
        **kwargs: forwarded to the specific backend constructor
    """
    if available_properties is not None:
        properties = [x if isinstance(x, PropertyInfo) else PropertyInfo(**x) for x in available_properties]
    else:
        properties = None
    if not isinstance(fmt, ChemDataSetFormat):
        raise IterDataCoreError(f"Cannot create dataset for format: {fmt}")
    return fmt.value.create(
        data_source=str(data_source), distance_unit=distance_unit, available_properties=properties, **kwargs
    )


def load_dataset(
    data_source: Union[PathLike, List[PathLike], List[Dict[str, Any]]],
    fmt: Optional[Union[ChemDataSetFormat, List[ChemDataSetFormat]]] = None,
    **kwargs,
) -> Union[UnionInterfaces, ConcatChemDataSet]:
    """
    Load one or more chemistry datasets and return the appropriate interface.

    Args:
        data_source: dataset path, or a list of paths/dicts for concatenation
        fmt: optional data format or list of formats matching data_source
        **kwargs: forwarded to the specific backend constructor if data_source is PathLike

    Returns:
        A dataset interface instance, or a concatenated dataset when multiple
        paths are provided.
    """
    if isinstance(data_source, (list, tuple)):
        if isinstance(fmt, (list, tuple)):
            if len(fmt) != len(data_source):
                raise IterDataCoreError("When passing multiple data_sources the format list must match in length.")
            formats = list(fmt)
        else:
            formats = [fmt for _ in data_source]
        iter_coll = [
            load_dataset(fmt=f, **path) if isinstance(path, dict) else load_dataset(path, fmt=f, **kwargs)
            for path, f in zip(data_source, formats)
        ]
        return ConcatChemDataSet(data_source=iter_coll)

    if isinstance(fmt, (list, tuple)):
        raise IterDataCoreError(f"{fmt=} not valid for one data_source {data_source=}")
    data_source, resolved_format = resolve_format(str(data_source), fmt=fmt)
    return resolved_format.value(data_source=data_source, **kwargs)


def resolve_format(data_source: str, fmt: Optional[ChemDataSetFormat] = None) -> Tuple[str, ChemDataSetFormat]:
    """
    Extract data format from file suffix, check for consistency with (optional) given
    format, or append suffix to file path.

    Args:
        data_source: path to atoms data
        format: atoms data format

    """
    collection_errors = []
    if fmt is ChemDataSetFormat.IN_MEMORY:
        return data_source, fmt
    if fmt is None or fmt is ChemDataSetFormat.ParAMS:
        try:
            ParAMSData._check_is_params(data_source)
            return str(data_source), ChemDataSetFormat.ParAMS
        except ParAMSDataError as e:
            collection_errors.append((e, ParAMSDataError))
            pass
    extension_map = {ChemDataSetFormat.ASE: ".db", ChemDataSetFormat.RKF: ".rkf"}
    _, suffix = os.path.splitext(data_source)
    if suffix in extension_map.values():
        # found a format valid
        if fmt is None:
            inverse = {v: k for k, v in extension_map.items()}
            fmt = inverse[suffix]
        assert fmt in extension_map, f"File extension {suffix} is not compatible with chosen format {fmt}"
    elif os.path.isdir(data_source):
        fmt = ChemDataSetFormat.ParAMS
    elif len(suffix) == 0 and fmt in extension_map:
        data_source = data_source + extension_map[fmt]
    elif len(suffix) == 0 and fmt is None:
        raise IterDataCoreError(
            f"If format is not given, `data_source` needs a supported file extension! | {data_source=}"
        )
    else:
        raise IterDataCoreError(f"Unsupported file extension: {suffix}, \n{collection_errors=}")
    assert fmt is not None
    return data_source, fmt
