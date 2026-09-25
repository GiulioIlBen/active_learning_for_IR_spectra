import importlib.util
import runpy
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COPY_PATH = ROOT / "src/scm/active_learning/cli/copy.py"
PLOT_PATH = ROOT / "src/scm/active_learning/cli/plot.py"
RUN_PATH = ROOT / "src/scm/active_learning/cli/run.py"
VALIDATE_PATH = ROOT / "src/scm/active_learning/cli/validate.py"
AL_PATH = ROOT / "src/scm/active_learning/__main__.py"


def _raise_missing_file(path):
    raise FileNotFoundError(2, "No such file or directory", str(path))


def _stub_active_learning(monkeypatch, calls, load_model_json=None):
    scm = types.ModuleType("scm")
    active_learning = types.ModuleType("scm.active_learning")
    cli = types.ModuleType("scm.active_learning.cli")

    class ActiveLearningLoop:
        @classmethod
        def load_model(cls, path):
            if load_model_json is not None:
                return load_model_json(path)
            calls["path"] = Path(path)

            def save_to_pdf(name, folder):
                calls["pdf"] = (name, folder)

            def run():
                action = "run:resume" if run_control.mode == "resume" else "run"
                calls.setdefault("actions", []).append(action)

            run_control = types.SimpleNamespace(mode="fresh")
            return types.SimpleNamespace(
                analysis=types.SimpleNamespace(plot=types.SimpleNamespace(save_to_pdf=save_to_pdf)),
                run=run,
                run_control=run_control,
            )

        @classmethod
        def load_model_json(cls, path):
            return cls.load_model(path)

        @classmethod
        def files_copy(cls, state_path, destination_folder, state_name=None, *, json_name=None):
            if load_model_json is not None:
                return load_model_json(state_path)
            output_name = state_name or json_name or f"copy{Path(state_path).suffix.lower()}"
            calls["copy"] = (Path(state_path), Path(destination_folder), output_name)

    monkeypatch.setitem(sys.modules, "scm", scm)
    monkeypatch.setitem(sys.modules, "scm.active_learning", active_learning)
    monkeypatch.setitem(sys.modules, "scm.active_learning.cli", cli)
    active_learning.ActiveLearningLoop = ActiveLearningLoop
    active_learning.__path__ = []
    cli.__path__ = []

    for module_name, module_path in (
        ("scm.active_learning.cli.copy", COPY_PATH),
        ("scm.active_learning.cli.plot", PLOT_PATH),
        ("scm.active_learning.cli.run", RUN_PATH),
        ("scm.active_learning.cli.validate", VALIDATE_PATH),
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        assert spec and spec.loader
        spec.loader.exec_module(module)


def test_plot_cli_loads_loop_and_saves_default_pdf(monkeypatch, tmp_path):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["python", str(tmp_path / "al.json")])

    runpy.run_path(str(PLOT_PATH), run_name="__main__")

    assert calls == {"path": tmp_path / "al.json", "pdf": ("analysis.pdf", "<ALFolder>")}


def test_plot_cli_loads_yaml_state(monkeypatch, tmp_path):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["python", str(tmp_path / "al.yaml")])

    runpy.run_path(str(PLOT_PATH), run_name="__main__")

    assert calls == {"path": tmp_path / "al.yaml", "pdf": ("analysis.pdf", "<ALFolder>")}


def test_copy_cli_copies_loop_to_default_json_name(monkeypatch, tmp_path):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["python", str(tmp_path / "al.json"), str(tmp_path / "copied")])

    runpy.run_path(str(COPY_PATH), run_name="__main__")

    assert calls == {"copy": (tmp_path / "al.json", tmp_path / "copied", "copy.json")}


def test_copy_cli_preserves_yaml_format_by_default(monkeypatch, tmp_path):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["python", str(tmp_path / "al.yaml"), str(tmp_path / "copied")])

    runpy.run_path(str(COPY_PATH), run_name="__main__")

    assert calls == {"copy": (tmp_path / "al.yaml", tmp_path / "copied", "copy.yaml")}


def test_copy_cli_without_args_shows_help(monkeypatch, capsys):
    _stub_active_learning(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["python"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(COPY_PATH), run_name="__main__")

    assert excinfo.value.code == 0
    assert "STATE-PATH" in capsys.readouterr().out


def test_plot_cli_without_args_shows_help(monkeypatch, capsys):
    _stub_active_learning(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["python"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(PLOT_PATH), run_name="__main__")

    assert excinfo.value.code == 0
    assert "STATE-PATH" in capsys.readouterr().out


def test_al_plot_dispatches_to_plot_cli(monkeypatch, tmp_path):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["python", "plot", str(tmp_path / "al.json")])

    runpy.run_path(str(AL_PATH), run_name="__main__")

    assert calls == {"path": tmp_path / "al.json", "pdf": ("analysis.pdf", "<ALFolder>")}


def test_al_copy_dispatches_to_copy_cli(monkeypatch, tmp_path):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["al", "copy", str(tmp_path / "al.json"), str(tmp_path / "copied")])

    runpy.run_path(str(AL_PATH), run_name="__main__")

    assert calls == {"copy": (tmp_path / "al.json", tmp_path / "copied", "copy.json")}


def test_al_copy_without_destination_shows_help_before_error(monkeypatch, capsys):
    _stub_active_learning(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["al", "copy", "state.json"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(AL_PATH), run_name="__main__")

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "usage: al copy" in stderr
    assert "DESTINATION-FOLDER" in stderr


def test_al_plot_without_state_json_shows_help_before_error(monkeypatch, capsys):
    _stub_active_learning(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["al", "plot"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(AL_PATH), run_name="__main__")

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "usage: al plot" in stderr
    assert "the following arguments are required: STATE-PATH" in stderr


def test_al_plot_runtime_error_is_short_by_default(monkeypatch):
    _stub_active_learning(
        monkeypatch,
        {},
        load_model_json=_raise_missing_file,
    )
    monkeypatch.setattr(sys, "argv", ["al", "plot", "123"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(AL_PATH), run_name="__main__")

    assert excinfo.value.code == "FileNotFoundError: [Errno 2] No such file or directory: '123'"


def test_al_plot_runtime_error_reraises_with_long(monkeypatch):
    _stub_active_learning(
        monkeypatch,
        {},
        load_model_json=_raise_missing_file,
    )
    monkeypatch.setattr(sys, "argv", ["al", "plot", "123", "--long"])

    with pytest.raises(FileNotFoundError, match=r"No such file or directory: '123'"):
        runpy.run_path(str(AL_PATH), run_name="__main__")


def test_al_help_shows_plot_command(monkeypatch, capsys):
    _stub_active_learning(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["python"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(AL_PATH), run_name="__main__")

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "copy" in output
    assert "plot" in output
    assert "run" in output
    assert "validate" in output


def test_al_validate_loads_yaml_without_running_loop(monkeypatch, tmp_path, capsys):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    state_path = tmp_path / "al.yaml"
    monkeypatch.setattr(sys, "argv", ["al", "validate", str(state_path)])

    runpy.run_path(str(AL_PATH), run_name="__main__")

    assert calls == {"path": state_path}
    assert f"Valid YAML state: {state_path}" in capsys.readouterr().out


def test_al_validate_rejects_non_yaml_state(monkeypatch, tmp_path):
    _stub_active_learning(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["al", "validate", str(tmp_path / "al.json")])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(AL_PATH), run_name="__main__")

    assert excinfo.value.code == "ValueError: state path must be a YAML file ending in .yaml or .yml"


def test_al_run_dispatches_to_run_cli(monkeypatch, tmp_path):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["al", "run", str(tmp_path / "al.json")])

    runpy.run_path(str(AL_PATH), run_name="__main__")

    assert calls == {"path": tmp_path / "al.json", "actions": ["run"]}


def test_al_run_resume_sets_resume_mode(monkeypatch, tmp_path):
    calls = {}
    _stub_active_learning(monkeypatch, calls)
    monkeypatch.setattr(sys, "argv", ["al", "run", str(tmp_path / "al.json"), "--resume"])

    runpy.run_path(str(AL_PATH), run_name="__main__")

    assert calls == {"path": tmp_path / "al.json", "actions": ["run:resume"]}


def test_al_run_runtime_error_reraises_by_default(monkeypatch):
    _stub_active_learning(
        monkeypatch,
        {},
        load_model_json=_raise_missing_file,
    )
    monkeypatch.setattr(sys, "argv", ["al", "run", "123"])

    with pytest.raises(FileNotFoundError, match=r"No such file or directory: '123'"):
        runpy.run_path(str(AL_PATH), run_name="__main__")


def test_al_run_rejects_long_flag(monkeypatch):
    _stub_active_learning(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["al", "run", "state.yaml", "--long"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(AL_PATH), run_name="__main__")

    assert excinfo.value.code == 2


def test_al_run_without_path_json_shows_help_before_error(monkeypatch, capsys):
    _stub_active_learning(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["al", "run"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(AL_PATH), run_name="__main__")

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "usage: al run" in stderr
    assert "the following arguments are required: MODEL-PATH" in stderr
