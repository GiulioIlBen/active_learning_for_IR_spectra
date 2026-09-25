from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict

from scm.moliterate.utils.unit_conversion import conversion_ratio


class PropertyInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    unit: Optional[Union[str, Literal["NotUniform"]]] = None
    shape: Optional[Union[Tuple[Union[int, Literal["nAtoms"]], ...], Literal["float"]]] = None
    description: str = ""

    def __hash__(self) -> int:
        return hash((self.name, self.unit, self.shape))

    def unit_conversion_from(self, from_units: str) -> float:
        if self.unit is None or self.unit == "NotUniform":
            return 1
        return conversion_ratio(from_units, self.unit)
