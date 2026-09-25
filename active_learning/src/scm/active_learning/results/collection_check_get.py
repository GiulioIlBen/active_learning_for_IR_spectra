from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

from pydantic import BaseModel
from scm.moliterate.core.concat_chem_dataset import ConcatChemDataSet
from tabulate import tabulate

from .check_get import CheckGetResults


class CollectionCheckGetResults(BaseModel):
    validity_get_results: List[CheckGetResults]

    def get_final_dataset(self):
        return ConcatChemDataSet(
            data_source=[x.getter_results.get_dataset_with_metadata() for x in self.validity_get_results]
        )

    def validity_results_table(self):
        return tabulate(
            [x.to_compact_dict() for r in self.validity_get_results for x in r.validity_results],
            headers="keys",
        )

    def getter_results_table(self):

        return tabulate(
            [
                {
                    **r.getter_results.model_dump(
                        exclude={
                            "dataset",
                        }
                    ),
                    "NData": len(r.getter_results.dataset),
                }
                for r in self.validity_get_results
            ],
            headers="keys",
        )

    def summary(self, groupby: Literal["task_id", "system_id"] | None = None):
        @dataclass
        class Data:
            n_valid: int = 0
            n_valid_ok: int = 0
            n_frames_i: int = 0
            tasks: set = field(default_factory=set)
            systems: set = field(default_factory=set)

            def ret(self):
                return {
                    "n_tasks": len(self.tasks),
                    "n_systems": len(self.systems),
                    "n_validity_checks": self.n_valid,
                    "n_validity_success": self.n_valid_ok,
                    "n_frames": self.n_frames_i,
                }

        total: dict[str, Data] = {}

        def get_data(c: CheckGetResults) -> Data:
            if groupby is None:
                if "" not in total:
                    total[""] = Data()
                return total[""]
            v = getattr(c.getter_results, groupby)
            if v not in total:
                total[v] = Data()
            return total[v]

        for check_get in self.validity_get_results:
            data = get_data(check_get)
            data.n_frames_i += len(check_get.getter_results.dataset)
            for check in check_get.validity_results:
                data.tasks.add(check.task_id)
                data.systems.add(check.system_id)
                data.n_valid += 1
                data.n_valid_ok += int(check.success == "OK")

        if groupby is None:
            yield total[""].ret()
            return
        for x, y in total.items():
            yield {
                groupby: x,
                **y.ret(),
            }
