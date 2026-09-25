from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, model_validator

# steps,last_checker_ids,last_checker_reasons,iterdata_consequences,len_data_to_add,iterdata_coll,iteration_al


class EngineCheckResult(BaseModel):
    engine_id: str
    task_id: str
    system_id: Union[int, str]
    checker_id: str
    property: str
    units: str = ""
    per_n_atoms: bool = False
    metric: str
    # default grouping: per system/atom type/component x,y,z
    atom_type: int = -1
    components: int = -1
    value_type: Literal["str", "float"] = "float"
    value: Union[float, str]
    target: Union[float, str] = 0.0
    lower_is_better: bool = True
    no_data_is_success: bool = False
    n_entries: int
    success: Union[Literal["OK"], str]

    @model_validator(mode="before")
    @classmethod
    def _validate_value_type(cls, data):
        if isinstance(data, cls) or not isinstance(data, dict):
            return data
        if data.get("value_type") == "str":
            if "target" not in data or data.get("target") is None:
                data["target"] = ""
            value = data.get("value")
            if value is not None and not isinstance(value, str):
                raise TypeError("value must be str when value_type='str'")
            target = data.get("target")
            if target is not None and not isinstance(target, str):
                raise TypeError("target must be str when value_type='str'")
        return data

    def is_success(self):
        return self.success == "OK" or (self.success == "NoData" and self.no_data_is_success)

    @staticmethod
    def summarize_by_atoms(dic: dict):
        if dic.pop("per_n_atoms"):
            dic["units"] += "/NAtoms"
        return dic

    def to_compact_dict(self) -> dict:
        return self.summarize_by_atoms(self.model_dump(exclude={"lower_is_better", "no_data_is_success", "value_type"}))
