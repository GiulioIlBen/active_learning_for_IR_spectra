from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Union

import yaml

StateFormat = Literal["json", "yaml"]

_FORMAT_BY_SUFFIX: dict[str, StateFormat] = {
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


class _LiteralBlockString(str):
    pass


class _StateDumper(yaml.SafeDumper):
    pass


def _represent_literal_block_string(dumper: yaml.SafeDumper, value: _LiteralBlockString) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")


_StateDumper.add_representer(_LiteralBlockString, _represent_literal_block_string)


def _format_yaml_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_format_yaml_payload(value) for value in payload]
    if not isinstance(payload, dict):
        return payload

    formatted = {}
    if "type" in payload:
        formatted["type"] = _format_yaml_payload(payload["type"])

    for key, value in payload.items():
        if key == "type":
            continue
        if key == "chemical_systems" and isinstance(value, dict):
            formatted[key] = {
                name: _LiteralBlockString(system) if isinstance(system, str) else _format_yaml_payload(system)
                for name, system in value.items()
            }
        else:
            formatted[key] = _format_yaml_payload(value)
    return formatted


def state_format_from_path(path: Union[str, Path]) -> StateFormat:
    state_path = Path(path)
    try:
        return _FORMAT_BY_SUFFIX[state_path.suffix.lower()]
    except KeyError:
        supported = ", ".join(sorted(_FORMAT_BY_SUFFIX))
        message = f"Unsupported state file extension {state_path.suffix!r}; expected one of: {supported}"
        raise ValueError(message) from None


def load_state_payload(path: Union[str, Path]) -> dict[str, Any]:
    state_path = Path(path).expanduser().resolve(strict=False)
    state_format = state_format_from_path(state_path)
    with state_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) if state_format == "json" else yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"State file must contain a mapping at its root: {state_path}")
    return payload


def write_state_payload(
    payload: dict[str, Any],
    path: Union[str, Path],
    *,
    indent: int = 2,
) -> Path:
    state_path = Path(path).expanduser()
    state_format = state_format_from_path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_format == "json":
        text = json.dumps(payload, indent=indent)
    else:
        text = yaml.dump(
            _format_yaml_payload(payload),
            Dumper=_StateDumper,
            allow_unicode=True,
            default_flow_style=False,
            indent=indent,
            sort_keys=False,
        )
    state_path.write_text(text, encoding="utf-8")
    return state_path
