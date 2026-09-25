import re

import yaml

from scm.active_learning.logging.state_serialization import load_state_payload, write_state_payload


def _assert_type_is_first(payload):
    if isinstance(payload, list):
        for value in payload:
            _assert_type_is_first(value)
        return
    if not isinstance(payload, dict):
        return

    if "type" in payload:
        assert next(iter(payload)) == "type"
    for value in payload.values():
        _assert_type_is_first(value)


def test_yaml_formats_only_chemical_systems_as_literal_blocks(tmp_path):
    payload = {
        "description": "first line\nsecond line",
        "task": {
            "atomistic_system": {
                "chemical_systems": {
                    "main": "System\n   Atoms\n      H 0.0 0.0 0.0\n   End\nEnd",
                }
            }
        },
    }
    state_path = tmp_path / "state.yaml"

    write_state_payload(payload, state_path)

    text = state_path.read_text(encoding="utf-8")
    assert re.search(r"(?m)^      main: \|-$", text)
    assert not re.search(r"(?m)^description: \|", text)
    assert load_state_payload(state_path) == payload
    assert type(payload["task"]["atomistic_system"]["chemical_systems"]["main"]) is str


def test_yaml_places_every_type_discriminator_first(tmp_path):
    payload = {
        "callbacks": [
            {
                "root_dir": {"base_dir": "ALruns"},
                "state_format": "yaml",
                "type": "FolderManagerCallback",
            },
            {"type": "ALStateLogger"},
            {
                "log_history_table": True,
                "type": "ALMinimalLogger",
            },
        ],
        "journey": {
            "task": {
                "settings": {},
                "type": "AMSTask",
            },
            "type": "MoleculesJourney",
        },
    }
    state_path = tmp_path / "state.yaml"

    write_state_payload(payload, state_path)

    dumped_payload = yaml.safe_load(state_path.read_text(encoding="utf-8"))
    _assert_type_is_first(dumped_payload)
