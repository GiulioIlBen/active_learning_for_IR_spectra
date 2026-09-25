from __future__ import annotations

import copy
import os
from collections import defaultdict
from itertools import islice
from os.path import join
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, Iterator, List, Literal, Optional, Set, Tuple, Union

from pydantic import PrivateAttr, model_validator

from scm.moliterate.core.base_chem_dataset import BaseChemDataSet
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.utils.unit_conversion import conversion_ratio

if TYPE_CHECKING:
    from scm.params import ParAMSJob, ParAMSResults, ResultsImporter
    from scm.plams import Settings

__all__ = [
    "ParAMSData",
]

_PARAMS_IMPORTS: Optional[Tuple[Any, Any, Any, Any, Any, Any]] = None


def _import_required_params():
    global _PARAMS_IMPORTS
    if _PARAMS_IMPORTS is None:
        try:
            from scm.params import DataSet, ParAMSJob, ParAMSResults, ResultsImporter, JCEntry  # noqa F401
            from scm.plams import Settings  # noqa F401

            _PARAMS_IMPORTS = (Settings, DataSet, ParAMSJob, ParAMSResults, ResultsImporter, JCEntry)
        except ModuleNotFoundError:
            raise ParAMSDataError("Not found scm.params, to install it look at documentation")
    return _PARAMS_IMPORTS


class ParAMSDataError(Exception):
    pass


class ParAMSDataWarning(Warning):
    pass


class ParAMSData(BaseChemDataSet):
    """ParAMS database interface for atomistic data."""

    type: Literal["ParAMSData"] = "ParAMSData"
    data_source: str
    add_dataset_metadata: bool = False

    dataset_metadata_key: ClassVar[Literal["dataset"]] = "dataset"

    _results_importer: "ResultsImporter" = PrivateAttr()
    _keys: List[Any] = PrivateAttr()
    _metadata_cache: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _properties_cache: Optional[List[PropertyInfo]] = PrivateAttr(default=None)

    # TODO: "ResultsImporter", "ParAMSResults", "ParAMSJob", does BaseModel raise an error if data_source: ParAMSJob?
    # TODO: I have to fix this pattern: only at initialization
    # I allow user to insert in data_source all the union options
    # but then after init data_source should be a string and in private attr should be _results_importer
    @model_validator(mode="after")
    def _validate_data_source(self):
        if self.data_source and not os.path.exists(self.data_source):
            raise ParAMSDataError(f"ParAMS file does not exists at {self.data_source}")
        importer, data_source_str = self._importer_from_any(self.data_source)
        self.data_source = data_source_str
        self._results_importer = importer
        self._keys = sorted(list(self._results_importer.job_collection.keys()), key=str)
        if len(self._keys) == 0:
            raise ParAMSDataError(f"Not found any entry is it the right path: {self.data_source}")
        return self

    # --------------------------------------------------------------------- #
    # initialization helpers
    # --------------------------------------------------------------------- #
    @classmethod
    def _importer_from_any(cls, path: Union[str, Path, "ResultsImporter", "ParAMSResults", "ParAMSJob"]):
        _, DataSet, ParAMSJob, ParAMSResults, ResultsImporter, _ = _import_required_params()

        if isinstance(path, ResultsImporter):
            return path, str(Path(".").resolve())
        if isinstance(path, ParAMSResults):
            return ResultsImporter.from_params_results(path), path.job.path
        if isinstance(path, ParAMSJob):
            return ResultsImporter.from_params_results(path.results), path.path
        if not isinstance(path, (str, Path)):
            raise ParAMSDataError(f"ParAMSData import failed: {type(path)=} is not valid type. {path=} ")
        path = cls._check_is_params(path)
        if path.endswith(".yaml.gz"):
            ds_folder = Path(path).parent / "data_sets"
            paths = [x for x in ds_folder.glob("*.yaml.gz")]
            data_set = {str(x.name).split(".")[0]: DataSet(str(x)) for x in paths}
            return (
                ResultsImporter(
                    job_collection=path,
                    data_set=data_set,
                ),
                path,
            )
        elif path.endswith(".yaml"):
            path = str(Path(path).parent)
            return ResultsImporter.from_yaml(path), path
        elif path.endswith(".xyz"):
            path = str(Path(path).parent)
            return ResultsImporter.from_ase(path), path
        raise ParAMSDataError(f"You should not be here. BAD implementation of _check_is_params? {path=}")

    @classmethod
    def _check_is_params(cls, path: Union[Path, str]):
        path = str(path)
        contains_expected_file = [
            "job_collection.yaml.gz",
            "settings_and_initial_data/job_collection.yaml.gz",
            "results/settings_and_initial_data/job_collection.yaml.gz",
            "job_collection.yaml",
            "training_set.xyz",
        ]
        extensions = [".yaml.gz", ".xtz", ".yaml"]
        for pp in [path, *map(lambda x: join(path, x), contains_expected_file)]:
            if Path(pp).is_file() and any([pp.endswith(ext) for ext in extensions]):
                return pp
        raise ParAMSDataError(
            f"ParAMSData import failed: path has not the right extension "
            "or does not contains the required file:\n {path=} \n "
            "Available: {extensions=} \n Expected file options {contains_expected_file}"
        )

    # --------------------------------------------------------------------- #
    # BaseMolIterate interface
    # --------------------------------------------------------------------- #
    def total_len(self) -> int:
        return len(self._keys)

    @property
    def available_properties(self) -> List[PropertyInfo]:
        if self._properties_cache is not None:
            return list(self._properties_cache)

        prop_db: Dict[str, Set[str]] = defaultdict(set)
        prop_unit: Dict[str, Set[str]] = defaultdict(set)
        prop_shape: Dict[str, Set] = defaultdict(set)
        for dataset_name, dataset in self._results_importer.data_sets.items():
            for dataset_entry in dataset:
                for prop_name in getattr(dataset_entry, "extractors", []):
                    prop_db[prop_name].add(dataset_name)
                    prop_unit[prop_name].add(dataset_entry.unit[0])
                    try:
                        shape = dataset_entry.reference.shape
                    except Exception:
                        shape = "float"
                    prop_shape[prop_name].add(shape)
        properties: List[PropertyInfo] = []
        for prop, datasets in sorted(prop_db.items()):
            description = f"Units might not be uniform! Found in {sorted(set(datasets))}"
            shape = prop_shape[prop]
            if len(shape) == 1:
                shape_out = list(shape)[0]
            else:
                tuple_shapes = [s for s in shape if isinstance(s, tuple)]
                if not tuple_shapes:
                    shape_out = "float"
                else:
                    max_len = max(len(s) for s in tuple_shapes)
                    axis_values = []
                    for axis in range(max_len):
                        vals = {s[axis] if axis < len(s) else None for s in tuple_shapes}
                        if len(vals) == 1 and None not in vals:
                            axis_values.append(next(iter(vals)))
                        else:
                            axis_values.append(-1)
                    shape_out = tuple(axis_values)
            if prop == "forces":
                shape_out = tuple(val if val == 3 else -1 for val in shape_out)
            properties.append(
                PropertyInfo(name=prop, unit=list(prop_unit[prop])[0], shape=shape_out, description=description)
            )

        self._properties_cache = properties
        return list(properties)

    @property
    def metadata(self) -> Dict[str, Any]:
        Settings, _, _, _, _, _ = _import_required_params()

        if self._metadata_cache is not None:
            return dict(self._metadata_cache)

        settings = Settings()
        for job_entry in self._results_importer.job_collection.values():
            job_settings = getattr(job_entry, "settings", None)
            if job_settings is not None:
                settings += job_settings

        jc = self._results_importer.job_collection
        metadata = {
            "General": getattr(jc, "header", {}),
            "PropUnitsShape": {p.name: [p.unit, p.shape] for p in self.available_properties},
            "jc_nEngines": len(getattr(jc, "engines", [])),
            "all_settings": settings,
            "engine_collection": str(getattr(self._results_importer, "engine_collection", "")),
        }
        self._metadata_cache = metadata
        return dict(metadata)

    def __iter__(self) -> Iterator[ChemDataEntry]:
        transform = self.composed_transform
        for idx_absolute in self.select_indices():
            row = self._get_row_by_absolute_idx(int(idx_absolute))
            row = transform(row)
            yield row

    def get_row(self, idx: int) -> ChemDataEntry:
        idxs = [x for x in self.select_indices(indices=idx)]
        idx_absolute = int(idxs[0])
        row = self._get_row_by_absolute_idx(idx_absolute)
        return self.composed_transform(row)

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _get_row_by_absolute_idx(self, idx_absolute: int) -> ChemDataEntry:
        origin_key = self._keys[idx_absolute]
        job_entry = copy.deepcopy(self._results_importer.job_collection[origin_key])

        metadata: Dict[str, Any] = dict(getattr(job_entry, "metadata", {}) or {})
        reference_engine = getattr(job_entry, "reference_engine", None)
        extra_engine = getattr(job_entry, "extra_engine", None)
        if reference_engine is not None:
            metadata["reference_engine"] = reference_engine
        if extra_engine is not None:
            metadata["extra_engine"] = extra_engine

        properties: Dict[str, Any] = {}
        dataset_map: Dict[str, str] = {}
        extra_metadata: Dict[str, Any] = {}

        for prop in (p.name for p in self.available_properties):
            value, dataset_name, prop_metadata = self._load_property(origin_key, prop)
            properties[prop] = value
            if dataset_name is not None:
                dataset_map[prop] = dataset_name
            extra_metadata.update(prop_metadata)

        if dataset_map is not None:
            if len(set(dataset_map.values())) == 1:
                ds_name = list(dataset_map.values())[0]
            else:
                # ds_name = "-".join(list(dataset_map.values()))
                ds_name = dataset_map
            metadata[self.dataset_metadata_key] = ds_name
        if extra_metadata:
            metadata.update(extra_metadata)

        return ChemDataEntry(
            system=job_entry.molecule,
            properties=properties,
            metadata=metadata,
            idx_absolute=idx_absolute,
            idx_origin=origin_key,
        )

    def _load_property(self, origin_key: Any, prop: str):
        property_entry_key = f"{prop}('{origin_key}')"
        value = None
        dataset_name: Optional[str] = None
        prop_metadata: Dict[str, Any] = {}

        for db_name, dataset in self._results_importer.data_sets.items():
            if property_entry_key not in dataset:
                continue
            ds_entry = copy.deepcopy(dataset[property_entry_key])
            if value is not None:
                raise ParAMSDataError(
                    f"Found duplicate property '{prop}' for job '{origin_key}' in datasets {dataset_name} and {db_name}"
                )
            value = getattr(ds_entry, "reference", None)
            dataset_name = db_name
            if self.add_dataset_metadata:
                prop_metadata.update({f"{prop}_{k}": v for k, v in getattr(ds_entry, "metadata", {}).items()})

        return value, dataset_name, prop_metadata

    # --------------------------------------------------------------------- #
    # method helpers towards a BaseChemDataSetWriter
    # --------------------------------------------------------------------- #

    @classmethod
    def _importer_from_entries(
        cls,
        data: Iterable[ChemDataEntry],
        properties_to_add: List[PropertyInfo],
        suffix_origin: Optional[str] = "",
        ri: Optional["ResultsImporter"] = None,
        shared_settings: Optional["Settings"] = None,
        dataset_metadata_key: Optional[str] = None,
        engine_metadata_key: str = "reference_engine",
        default_dataset: str = "training_set",
    ) -> "ResultsImporter":
        Settings, DataSet, ParAMSJob, ParAMSResults, ResultsImporter, JCEntry = _import_required_params()
        if ri is None:
            settings = Settings()
            ri = ResultsImporter(settings=settings)

        # properties conversion
        supported_properties = ["energy", "gradients", "forces", "dipole_moment"]
        present = []
        for p in properties_to_add:
            if p.name not in supported_properties:
                raise ValueError(f"Property {p.name} cannot be added to this dataset | {supported_properties=}")
            if p.unit is None:
                raise ValueError(f"Units are required for this conversion but found {p=}")
            present.append(p.name)
        s = Settings()
        s.input.ams.task = "SinglePoint"
        if shared_settings is not None:
            s += shared_settings
        if any(x in present for x in ["gradients", "forces"]):
            s.input.ams.properties.gradients = "yes"
        if "dipole_moment" in present:
            s.input.ams.properties.dipolemoment = "yes"

        # add data
        start = len(ri.job_collection)
        ds_key = dataset_metadata_key or cls.dataset_metadata_key
        for i, row_data in enumerate(data, start=start):
            job_id = "{}{:04d}".format("ChemEntry", i)
            ### Metadata transfer
            metadata = {}
            metadata.update(row_data.metadata)
            if suffix_origin is not None:
                metadata[f"{suffix_origin}idx_origin"] = row_data.idx_origin
                metadata[f"{suffix_origin}idx_absolute"] = row_data.idx_absolute
            ### Add to job_collection
            # almost equivalent to ri.job_collection.load_single_ase_atoms
            jce = JCEntry(
                settings=s,
                molecule=row_data.molecule,
                reference_engine=metadata.pop(engine_metadata_key, ""),
                **metadata,
            )
            ri.job_collection.add_entry(job_id, jce)

            ### Add to data_set
            dataset_label = row_data.metadata.get(ds_key, default_dataset)

            def add_to_ds(data_set):
                if p.name not in row_data.properties:
                    return
                data_set.add_entry(
                    f"{p.name}('{job_id}')",
                    reference=row_data.properties[p.name],
                    unit=(p.unit, conversion_ratio("au", p.unit)),
                )

            if isinstance(dataset_label, str):
                data_set = ri.get_data_set(dataset_label)
                for p in properties_to_add:
                    add_to_ds(data_set)
            elif isinstance(dataset_label, dict):
                for p in properties_to_add:
                    data_set = ri.get_data_set(dataset_label[p.name])
                    add_to_ds(data_set)
            else:
                raise TypeError(
                    f"{type(dataset_label)=} is not str, or dict "
                    + "| {job_id=} | Value:{dataset_label} | {row_data.metadata.keys()=}"
                )
        return ri
