from __future__ import annotations

from typing import Union

from pydantic import BaseModel
from scm.moliterate import ConcreteInterfaces
from scm.moliterate.interfaces import InMemoryMolData
from scm.moliterate.transforms import MetadataAddTransform


class FrameGetterResult(BaseModel):
    engine_id: str
    task_id: str
    system_id: Union[int, str]
    checker_id: str
    getter_id: str
    dataset: ConcreteInterfaces = InMemoryMolData()

    def get_dataset_with_metadata(self):
        ret = self.dataset.subset()
        ret.transforms.append(
            MetadataAddTransform(
                key_values=self.model_dump(
                    include={
                        "engine_id",
                        "task_id",
                        "system_id",
                        "checker_id",
                        "getter_id",
                    }
                ),
                overwrite=False,
            )
        )
        return ret
