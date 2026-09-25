from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel
from scm.plams import Settings

from scm.active_learning.logging import log_level

from .state_paths import collect_paths
from .state_serialization import load_state_payload, write_state_payload

_IMPORTED_EXTERNAL_ROOT = Path("external") / "_imported"


@dataclass(frozen=True)
class _PathEntry:
    field_path: Tuple[Union[str, int], ...]
    resolved_path: Path


def _relative_to_or_none(path: Path, parent: Path) -> Optional[Path]:
    try:
        return path.relative_to(parent)
    except ValueError:
        return None


def _field_path_has_suffix(
    field_path: Tuple[Union[str, int], ...],
    suffix: Tuple[str, ...],
) -> bool:
    return len(field_path) >= len(suffix) and field_path[-len(suffix) :] == suffix


def _is_metadata_only_path(field_path: Tuple[Union[str, int], ...]) -> bool:
    return _field_path_has_suffix(field_path, ("root_dir", "run_root")) or _field_path_has_suffix(
        field_path, ("root_dir", "base_dir")
    )


def _sanitize_bucket_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "root"


def _external_anchor(path: Path) -> Path:
    if path.exists() and path.is_dir():
        return path
    return path.parent


def _external_bucket_name(path: Path) -> str:
    anchor = _external_anchor(path)
    digest = hashlib.sha1(str(anchor).encode("utf-8")).hexdigest()[:8]
    return f"{digest}_{_sanitize_bucket_label(anchor.name or 'root')}"


def _target_path_for_source(source_path: Path, source_run_root: Path, destination_folder: Path) -> Path:
    relative_run_path = _relative_to_or_none(source_path, source_run_root)
    if relative_run_path is not None:
        return destination_folder / relative_run_path

    anchor = _external_anchor(source_path)
    relative_external_path = _relative_to_or_none(source_path, anchor)
    if relative_external_path is None:
        relative_external_path = Path(source_path.name)
    return destination_folder / _IMPORTED_EXTERNAL_ROOT / _external_bucket_name(source_path) / relative_external_path


def _target_base_dir(source_base_dir: Path, source_run_root: Path, destination_folder: Path) -> Path:
    relative_run_root = _relative_to_or_none(source_run_root, source_base_dir)
    if relative_run_root is None:
        return destination_folder.parent

    target = destination_folder
    for _ in relative_run_root.parts:
        target = target.parent
    return target


def _resolve_source_path(value: Union[str, Path], source_state_dir: Path) -> Path:
    raw_path = Path(value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve(strict=False)
    return (source_state_dir / raw_path).resolve(strict=False)


def _collect_path_entries(flattened_payload: Settings, source_state_dir: Path) -> List[_PathEntry]:
    entries: List[_PathEntry] = []
    for field_path in collect_paths(flattened_payload):
        value = flattened_payload[field_path]
        if str(value) == "":
            continue
        entries.append(
            _PathEntry(
                field_path=field_path,
                resolved_path=_resolve_source_path(value, source_state_dir),
            )
        )
    return entries


def _rewrite_payload_paths(
    payload: Dict,
    replacements: Dict[Tuple[Union[str, int], ...], Path],
    state_dir: Path,
) -> Dict:
    flattened = Settings(payload).flatten()
    for field_path, target_path in replacements.items():
        flattened[field_path] = os.path.relpath(
            str(target_path.expanduser().resolve(strict=False)),
            start=str(state_dir),
        )
    return flattened.unflatten().as_dict()


def _source_run_root(path_entries: List[_PathEntry], source_state_dir: Path) -> Path:
    for entry in path_entries:
        if _field_path_has_suffix(entry.field_path, ("root_dir", "run_root")):
            return entry.resolved_path
    return source_state_dir


def _ensure_empty_destination(destination_folder: Path, destination_state: Path) -> None:
    if destination_folder.exists():
        if not destination_folder.is_dir():
            raise FileExistsError(f"Destination path exists and is not a directory: {destination_folder}")
        if any(destination_folder.iterdir()):
            raise FileExistsError(f"Destination folder is not empty: {destination_folder}")
    if destination_state.exists():
        raise FileExistsError(f"Destination state file already exists: {destination_state}")


def _validate_target_collisions(path_map: Dict[Path, Path]) -> None:
    seen_targets: Dict[Path, Path] = {}
    for source_path, target_path in path_map.items():
        previous_source = seen_targets.get(target_path)
        if previous_source is None:
            seen_targets[target_path] = source_path
            continue
        if previous_source != source_path:
            raise ValueError(
                "Two distinct source paths would be copied to the same destination: "
                f"{previous_source} and {source_path} -> {target_path}"
            )


def _reduce_copy_plan(path_map: Dict[Path, Path]) -> List[Tuple[Path, Path]]:
    reduced: List[Tuple[Path, Path]] = []
    for source_path, target_path in sorted(path_map.items(), key=lambda item: (len(item[0].parts), str(item[0]))):
        covered = False
        for parent_source, parent_target in reduced:
            if not parent_source.exists() or not parent_source.is_dir():
                continue
            relative_child = _relative_to_or_none(source_path, parent_source)
            if relative_child is not None and parent_target / relative_child == target_path:
                covered = True
                break
        if not covered:
            reduced.append((source_path, target_path))
    return reduced


def _warn_missing_sources(path_map: Dict[Path, Path]) -> None:
    for source_path in path_map:
        if source_path.exists():
            continue
        log_level(
            "Skipping missing path during ActiveLearningLoop.files_copy: {path}",
            level="WARNING",
            path=source_path,
        )


def _execute_copy_plan(copy_plan: List[Tuple[Path, Path]]) -> None:
    for source_path, target_path in copy_plan:
        if not source_path.exists():
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def _destination_state_path(destination_folder: Path, state_name: str) -> Path:
    state_name_path = Path(state_name)
    if state_name_path.is_absolute() or ".." in state_name_path.parts or state_name_path.name == "":
        raise ValueError(f"state_name must be a relative filename inside the destination folder: {state_name}")
    return destination_folder / state_name_path


def files_copy(
    model_type: Type[BaseModel],
    state_path: Union[str, Path],
    destination_folder: Union[str, Path],
    state_name: Optional[str] = None,
    *,
    # TODO: Remove json_name after downstream callers migrate to state_name.
    json_name: Optional[str] = None,
) -> Path:
    del model_type  # Kept in the public signature for compatibility with existing callers.
    if state_name is not None and json_name is not None:
        raise ValueError("Pass only one of state_name or the deprecated json_name")

    source_state_path = Path(state_path).expanduser().resolve(strict=False)
    source_state_dir = source_state_path.parent
    destination_dir = Path(destination_folder).expanduser().resolve(strict=False)
    destination_name = state_name or json_name or f"copy{source_state_path.suffix.lower()}"
    destination_state = _destination_state_path(destination_dir, destination_name)

    payload = load_state_payload(source_state_path)

    path_entries = _collect_path_entries(Settings(payload).flatten(), source_state_dir)
    source_run_root = _source_run_root(path_entries, source_state_dir)
    _ensure_empty_destination(destination_dir, destination_state)

    field_rewrites: Dict[Tuple[Union[str, int], ...], Path] = {}
    source_to_target: Dict[Path, Path] = {}

    for entry in path_entries:
        if _field_path_has_suffix(entry.field_path, ("root_dir", "run_root")):
            target_path = destination_dir
        elif _field_path_has_suffix(entry.field_path, ("root_dir", "base_dir")):
            target_path = _target_base_dir(entry.resolved_path, source_run_root, destination_dir)
        else:
            target_path = _target_path_for_source(entry.resolved_path, source_run_root, destination_dir)

        field_rewrites[entry.field_path] = target_path
        if _is_metadata_only_path(entry.field_path):
            continue

        previous_target = source_to_target.get(entry.resolved_path)
        if previous_target is not None and previous_target != target_path:
            raise ValueError(
                f"Source path {entry.resolved_path} would be rewritten to two destinations: "
                f"{previous_target} and {target_path}"
            )
        source_to_target[entry.resolved_path] = target_path

    _validate_target_collisions(source_to_target)
    _warn_missing_sources(source_to_target)
    copy_plan = _reduce_copy_plan(source_to_target)

    destination_dir.mkdir(parents=True, exist_ok=True)
    _execute_copy_plan(copy_plan)

    rewritten_payload = _rewrite_payload_paths(payload, field_rewrites, destination_state.parent)
    return write_state_payload(rewritten_payload, destination_state)
