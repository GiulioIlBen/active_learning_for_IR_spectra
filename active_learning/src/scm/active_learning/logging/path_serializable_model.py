from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, TypeVar, Union

from pydantic import BaseModel
from scm.plams import Settings

from scm.active_learning.logging.state_io import files_copy as copy_loop_files
from scm.active_learning.logging.state_paths import rewrite_payload_paths
from scm.active_learning.logging.state_serialization import load_state_payload, write_state_payload

TPathSerializableModel = TypeVar("TPathSerializableModel", bound="PathSerializableModel")


class PathSerializableModel(BaseModel):
    @staticmethod
    def _prepare_payload(payload: dict[str, Any], path: Union[str, Path]) -> tuple[Path, dict[str, Any]]:
        state_path = Path(path).expanduser()
        state_dir = state_path.parent.expanduser().resolve(strict=False)
        return state_path, rewrite_payload_paths(payload, state_dir, relative=True)

    @staticmethod
    # TODO: Remove after downstream callers migrate to _prepare_payload.
    def _prepare_json_payload(payload: dict[str, Any], path: Union[str, Path]) -> tuple[Path, dict[str, Any]]:
        return PathSerializableModel._prepare_payload(payload, path)

    def write_payload(self, payload: dict[str, Any], path: Union[str, Path], *, indent: int = 2) -> Path:
        state_path, rewritten_payload = self._prepare_payload(payload, path)
        return write_state_payload(rewritten_payload, state_path, indent=indent)

    def write_model(self, path: Union[str, Path], *, indent: int = 2) -> Path:
        payload = self.model_dump(mode="json", round_trip=True)
        return self.write_payload(payload, path, indent=indent)

    @classmethod
    def load_model(cls: type[TPathSerializableModel], path: Union[str, Path]) -> TPathSerializableModel:
        normalized = cls._load_normalized_payload(path)
        return cls.model_validate(normalized)

    @classmethod
    def _load_normalized_payload(cls, path: Union[str, Path]) -> dict[str, Any]:
        state_path = Path(path).expanduser().resolve(strict=False)
        payload = load_state_payload(state_path)
        return rewrite_payload_paths(payload, state_path.parent, relative=False)

    # TODO: Remove after downstream callers migrate to write_payload.
    def write_json_payload(self, payload: dict[str, Any], path: Union[str, Path], *, indent: int = 2) -> Path:
        return self.write_payload(payload, path, indent=indent)

    # TODO: Remove after downstream callers migrate to write_model.
    def write_model_json(self, path: Union[str, Path], *, indent: int = 2) -> Path:
        return self.write_model(path, indent=indent)

    @classmethod
    # TODO: Remove after downstream callers migrate to load_model.
    def load_model_json(cls: type[TPathSerializableModel], path: Union[str, Path]) -> TPathSerializableModel:
        return cls.load_model(path)

    @classmethod
    # TODO: Remove after downstream callers migrate to _load_normalized_payload.
    def _load_normalized_json(cls, path: Union[str, Path]) -> dict[str, Any]:
        return cls._load_normalized_payload(path)

    @classmethod
    def files_copy(
        cls,
        state_path: Union[str, Path],
        destination_folder: Union[str, Path],
        state_name: Optional[str] = None,
        *,
        # TODO: Remove json_name after downstream callers migrate to state_name.
        json_name: Optional[str] = None,
    ) -> Path:
        return copy_loop_files(
            cls,
            state_path,
            destination_folder,
            state_name=state_name,
            json_name=json_name,
        )

    @property
    def settings(self) -> Settings:
        return Settings(self.model_dump(mode="json"))
