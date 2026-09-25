from __future__ import annotations

from typing import Set

from pydantic import BaseModel
from scm.moliterate import ConcreteInterfaces


class SPLabellingResults(BaseModel):
    dataset: ConcreteInterfaces
    failed_simulations: Set[int] = set()
