from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
import yaml
from scm.plams import Settings

from scm.active_learning.logging.path_serializable_model import PathSerializableModel
from scm.active_learning.logging.state_paths import (
    collect_paths,
    collect_runtime_paths,
    fix_path,
    rewrite_payload_paths,
)


class DummyLabeller(PathSerializableModel):
    out_path: Optional[Path] = None


class DummyLoopStateModel(PathSerializableModel):
    data_source: Optional[Path] = None
    labeller: DummyLabeller = DummyLabeller()
    callbacks: List[str] = []

    def payload_for_path(self, path: Union[str, Path]) -> Dict[str, Any]:
        payload = self.model_dump(mode="json", round_trip=True)
        _, rewritten_payload = self._prepare_payload(payload, path)
        return rewritten_payload


def test_legacy_json_modules_reexport_state_helpers():
    from scm.active_learning.logging.json_io import files_copy as legacy_files_copy
    from scm.active_learning.logging.json_paths import rewrite_payload_paths as legacy_rewrite_payload_paths
    from scm.active_learning.logging.state_io import files_copy

    assert legacy_files_copy is files_copy
    assert legacy_rewrite_payload_paths is rewrite_payload_paths


def test_collect_paths_returns_only_registered_path_fields():
    flattened = Settings(
        {
            "data_source": "data.db",
            "nested": {"data_source": "nested.db", "ignore": "value"},
            "callbacks": [{"root_dir": {"run_root": "run", "logging_file": "log.txt"}}],
            "labeller": {"out_path": "labelled.db"},
        }
    ).flatten()

    paths = set(collect_paths(flattened))

    assert ("data_source",) in paths
    assert ("nested", "data_source") in paths
    assert ("callbacks", 0, "root_dir", "run_root") in paths
    assert ("labeller", "out_path") in paths
    assert ("nested", "ignore") not in paths
    assert ("callbacks", 0, "root_dir", "logging_file") not in paths


def test_collect_paths_matches_registered_model_path_lists():
    flattened = Settings(
        {
            "result": {
                "final_engine": {
                    "model_files": ["foundation-0.model", "foundation-1.model"],
                    "calculator_kwargs": {
                        "model_paths": ["committee-0.model", "committee-1.model"],
                        "model_file": "single.model",
                    },
                }
            }
        }
    ).flatten()

    paths = set(collect_paths(flattened))

    assert ("result", "final_engine", "model_files", 0) in paths
    assert ("result", "final_engine", "model_files", 1) in paths
    assert ("result", "final_engine", "calculator_kwargs", "model_paths", 0) in paths
    assert ("result", "final_engine", "calculator_kwargs", "model_paths", 1) in paths
    assert ("result", "final_engine", "calculator_kwargs", "model_file") in paths


def test_fix_path_relativizes_existing_runtime_relative_paths(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source_file = workspace / "data.db"
    source_file.write_text("content", encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    monkeypatch.chdir(workspace)

    rewritten = fix_path("data.db", state_dir, relative=True)

    assert rewritten == os.path.relpath(str(source_file.resolve()), start=str(state_dir))


def test_fix_path_keeps_missing_relative_paths_when_relativizing(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    assert fix_path("missing.db", state_dir, relative=True) == "missing.db"
    assert fix_path("", state_dir, relative=True) == ""


def test_rewrite_payload_paths_absolutizes_only_registered_fields(tmp_path):
    state_dir = tmp_path / "configs"
    state_dir.mkdir()
    payload = {
        "data_source": "inputs/data.db",
        "labeller": {"out_path": "outputs/labelled.db"},
        "plain_path": "leave/me/alone",
    }

    rewritten = rewrite_payload_paths(payload, state_dir, relative=False)

    assert rewritten["data_source"] == str((state_dir / "inputs/data.db").resolve(strict=False))
    assert rewritten["labeller"]["out_path"] == str((state_dir / "outputs/labelled.db").resolve(strict=False))
    assert rewritten["plain_path"] == "leave/me/alone"


def test_collect_runtime_paths_resolves_registered_journey_and_engine_paths(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    previous_job = workspace / "run" / "iter_00" / "previous_job"
    start_engine = workspace / "models" / "start_engine"
    trained_engine = workspace / "run" / "iter_01" / "ParAMSTrainer"
    for path in (previous_job, start_engine, trained_engine):
        path.mkdir(parents=True)

    payload = {
        "journey": {"previous_job_": "run/iter_00/previous_job"},
        "current_state": {"start_engine": {"job_path": "models/start_engine"}},
        "result": {"iterations": [{"train_results": {"engine": {"job_path": "run/iter_01/ParAMSTrainer"}}}]},
        "plain_path": "ignore/me",
    }

    runtime_paths = set(collect_runtime_paths(payload))

    assert previous_job.resolve(strict=False) in runtime_paths
    assert start_engine.resolve(strict=False) in runtime_paths
    assert trained_engine.resolve(strict=False) in runtime_paths
    assert (workspace / "ignore" / "me").resolve(strict=False) not in runtime_paths


def test_write_model_json_round_trips_registered_paths(tmp_path):
    source_file = tmp_path / "inputs" / "source.db"
    source_file.parent.mkdir()
    source_file.write_text("content", encoding="utf-8")

    labeller_out = tmp_path / "outputs" / "labelled.db"
    labeller_out.parent.mkdir()
    labeller_out.write_text("content", encoding="utf-8")

    model = DummyLoopStateModel(
        data_source=source_file,
        labeller=DummyLabeller(out_path=labeller_out),
    )
    json_path = tmp_path / "saved" / "loop.json"

    payload = model.payload_for_path(json_path)

    assert payload["data_source"] == os.path.relpath(str(source_file.resolve()), start=str(json_path.parent))
    assert payload["labeller"]["out_path"] == os.path.relpath(str(labeller_out.resolve()), start=str(json_path.parent))

    written_path = model.write_model_json(json_path)
    reloaded = DummyLoopStateModel.load_model_json(written_path)

    assert reloaded.data_source == source_file.resolve()
    assert reloaded.labeller.out_path == labeller_out.resolve()


def test_write_model_yaml_round_trips_registered_paths(tmp_path):
    source_file = tmp_path / "inputs" / "source.db"
    source_file.parent.mkdir()
    source_file.write_text("content", encoding="utf-8")
    model = DummyLoopStateModel(data_source=source_file)

    yaml_path = model.write_model(tmp_path / "saved" / "loop.yaml")
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    reloaded = DummyLoopStateModel.load_model(yaml_path)

    assert payload["data_source"] == os.path.relpath(str(source_file.resolve()), start=str(yaml_path.parent))
    assert reloaded.data_source == source_file.resolve()


def test_state_serialization_rejects_unsupported_extension(tmp_path):
    model = DummyLoopStateModel()

    with pytest.raises(ValueError, match="Unsupported state file extension"):
        model.write_model(tmp_path / "loop.toml")


def test_state_serialization_rejects_non_mapping_yaml(tmp_path):
    yaml_path = tmp_path / "loop.yaml"
    yaml_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a mapping"):
        DummyLoopStateModel.load_model(yaml_path)


def test_files_copy_round_trips_through_path_serializable_model_subclass(tmp_path):
    source_file = tmp_path / "inputs" / "source.db"
    source_file.parent.mkdir()
    source_file.write_text("source", encoding="utf-8")

    labeller_out = tmp_path / "outputs" / "labelled.db"
    labeller_out.parent.mkdir()
    labeller_out.write_text("labelled", encoding="utf-8")

    source_json = DummyLoopStateModel(
        data_source=source_file,
        labeller=DummyLabeller(out_path=labeller_out),
    ).write_model_json(tmp_path / "saved" / "loop.json")

    copied_json = DummyLoopStateModel.files_copy(source_json, tmp_path / "copied")
    reloaded = DummyLoopStateModel.load_model_json(copied_json)

    assert isinstance(reloaded, DummyLoopStateModel)
    assert reloaded.data_source is not None and reloaded.data_source.is_file()
    assert reloaded.labeller.out_path is not None and reloaded.labeller.out_path.is_file()
    assert (tmp_path / "copied") in reloaded.data_source.parents
    assert (tmp_path / "copied") in reloaded.labeller.out_path.parents
    assert reloaded.data_source.read_text(encoding="utf-8") == "source"
    assert reloaded.labeller.out_path.read_text(encoding="utf-8") == "labelled"


def test_files_copy_preserves_yaml_format_by_default(tmp_path):
    source_file = tmp_path / "inputs" / "source.db"
    source_file.parent.mkdir()
    source_file.write_text("source", encoding="utf-8")
    source_yaml = DummyLoopStateModel(data_source=source_file).write_model(tmp_path / "saved" / "loop.yaml")

    copied_yaml = DummyLoopStateModel.files_copy(source_yaml, tmp_path / "copied")
    reloaded = DummyLoopStateModel.load_model(copied_yaml)

    assert copied_yaml.name == "copy.yaml"
    assert reloaded.data_source is not None and reloaded.data_source.read_text(encoding="utf-8") == "source"


def test_rewrite_payload_paths_preserves_long_lists(tmp_path):
    payload = {
        "mlip_trainer": {"data": {"folder": "runs"}},
        "accuracy_checker": {"settings": [{"property": f"p{i}", "atom_type": i} for i in range(12)]},
    }

    rewritten = rewrite_payload_paths(payload, tmp_path, relative=False)

    assert rewritten["accuracy_checker"]["settings"] == payload["accuracy_checker"]["settings"]
    assert rewritten["mlip_trainer"]["data"]["folder"] == str((tmp_path / "runs").resolve())
