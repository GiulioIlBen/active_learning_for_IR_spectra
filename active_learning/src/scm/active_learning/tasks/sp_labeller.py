from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Tuple, Union

from pydantic import BaseModel, field_serializer, field_validator
from scm.moliterate import (
    ChemDataSetFormat,
    ConcreteInterfaces,
    PropertyInfo,
    create_dataset,
)

from scm.active_learning.engines.core import Engine
from scm.active_learning.task_parallelization import (
    ConcreteParallelizationStrategy,
    SerialStrategy,
)
from scm.active_learning.tasks.core import Task

if TYPE_CHECKING:
    from scm.active_learning.results import SPLabellingResults


class SPLabeller(BaseModel):
    properties: List[PropertyInfo]
    parallelization: ConcreteParallelizationStrategy = SerialStrategy()
    out_fmt: Literal[ChemDataSetFormat.ASE, ChemDataSetFormat.IN_MEMORY] = ChemDataSetFormat.ASE
    out_path: str = "labelling_000.db"
    if_out_path_exists: Literal["rename", "raise", "overwrite"] = "rename"
    batch_collection: int = 1000

    @field_validator("out_fmt", mode="before")
    @classmethod
    def _validate_out_fmt(cls, out_fmt: Union[str, ChemDataSetFormat]):
        mapped = {
            "ASE": ChemDataSetFormat.ASE,
            "IN_MEMORY": ChemDataSetFormat.IN_MEMORY,
        }
        if isinstance(out_fmt, str):
            return mapped[out_fmt]
        return out_fmt

    @field_serializer("out_fmt")
    def serialize_out_fmt(self, out_fmt: ChemDataSetFormat) -> str:
        # Enum values point to dataset classes; serialize to the enum name for JSON output
        return out_fmt.name

    def run(self, engine: Engine, dataset: ConcreteInterfaces, **kwargs) -> Tuple[ConcreteInterfaces, Dict[int, str]]:
        ds = create_dataset(
            data_source=self._get_valid_path(),
            fmt=self.out_fmt,
            available_properties=self.properties,
        )
        default_values = {prop.name: None for prop in self.properties}
        failed_simulations = {}
        sp_results = engine.run_single_point(
            properties=self.properties,
            dataset=dataset,
            parallel_settings=self.parallelization,
        )
        with ds.batch_collector(batch_size=self.batch_collection) as add:
            for i, (entry, res) in enumerate(zip(dataset, sp_results, strict=True)):
                if "FAILURE" in res:
                    failed_simulations[i] = res["FAILURE"]
                    entry.properties = default_values
                    entry.metadata["FAILURE"] = True
                    entry.metadata["error"] = res["FAILURE"]
                else:
                    entry.properties.update(res)
                    entry.metadata["FAILURE"] = False
                    entry.metadata["error"] = ""
                add(entry)

        return ds, failed_simulations

    def _get_valid_path(self) -> str:
        out_path = self.out_pathlib
        if not out_path.exists():
            return self.out_path
        if self.if_out_path_exists == "raise":
            raise FileExistsError(f"{out_path=}")
        if self.if_out_path_exists == "overwrite":
            out_path.unlink()
            return self.out_path
        if self.if_out_path_exists == "rename":
            candidate = out_path
            for _ in range(100):
                candidate = self._special_rename(candidate)
                if not candidate.exists():
                    return str(candidate)
            raise FileExistsError(f"Could not find a free path after {100} rename attempts starting from {out_path}")
        raise ValueError("You should not be here...")

    @property
    def out_pathlib(self):
        return Path(self.out_path)

    @staticmethod
    def _special_rename(out_pathlib: Path) -> Path:
        suffix = "".join(out_pathlib.suffixes)
        stem = out_pathlib.name[: -len(suffix)] if suffix else out_pathlib.name
        match = re.search(r"\d+$", stem)
        if match is None:
            new_stem = f"{stem}_000"
            return out_pathlib.with_name(f"{new_stem}{suffix}")
        digits = match.group(0)
        incremented = str(int(digits) + 1).zfill(len(digits))
        new_stem = f"{stem[: -len(digits)]}{incremented}"
        return out_pathlib.with_name(f"{new_stem}{suffix}")


class SPLabellerData(Task[Engine]):
    type: Literal["SPLabellerData"] = "SPLabellerData"  # It must have a type, and it is used as discriminator
    sp_labeller: SPLabeller
    dataset: ConcreteInterfaces

    @property
    def system_id(self) -> str:
        return f"Len{len(self.dataset)}"

    def run(self, engine: Engine, **kwargs) -> "SPLabellingResults":
        from scm.active_learning.results import SPLabellingResults

        ds, failures = self.sp_labeller.run(engine=engine, dataset=self.dataset)
        return SPLabellingResults(
            task=self,
            engine=engine,
            dataset=ds,
            failed_simulations=failures,
        )
