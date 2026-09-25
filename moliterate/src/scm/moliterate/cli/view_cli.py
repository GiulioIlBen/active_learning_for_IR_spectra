import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


def _viewer_app_path() -> Path:
    return Path(__file__).resolve().parents[1] / "apps" / "view_app.py"


def _ensure_marimo_installed() -> None:
    if importlib.util.find_spec("marimo") is None:
        raise ModuleNotFoundError(
            "marimo is not installed. Install the visualization extras with `pip install 'moliterate[viz]'`."
        )


def launch_structure_viewer(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_marimo_installed()
    command = [sys.executable, "-m", "marimo", "run", str(_viewer_app_path())]
    print("Running: " + " ".join(command))
    if argv:
        command.extend(argv)
    return subprocess.run(command, check=False).returncode


def main(cli_args: Optional[Sequence[str]] = None) -> int:
    return launch_structure_viewer(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
