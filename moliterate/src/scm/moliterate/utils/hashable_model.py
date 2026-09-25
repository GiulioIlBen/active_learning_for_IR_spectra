import hashlib
import json
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict


def stable_json_digest(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_set(values: Iterable[Any]) -> str:
    parts: list[str] = []
    for v in values:
        if isinstance(v, FrozenHashableModel):
            parts.append(v._content_digest())
        else:
            parts.append(stable_json_digest(v))
    parts.sort()
    return stable_json_digest(parts)


class FrozenHashableModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    def hash_payload(self) -> Any:
        """
        Override per-model: return a JSON-serializable structure that represents identity.
        Use hash_set(...) for set fields.
        """
        # By default, use model_dump(mode="python") and let subclasses override if needed.
        return self.model_dump(mode="python")

    def _content_digest(self) -> str:
        return stable_json_digest(self.hash_payload())

    def __hash__(self) -> int:
        # int required; stable across runs
        return int(self._content_digest(), 16)
