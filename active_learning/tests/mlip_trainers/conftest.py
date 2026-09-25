from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("mlip-trainers")
    group.addoption(
        "--mlip-trainer-results-dir",
        action="store",
        default=None,
        metavar="PATH",
        help="Directory used for generated MLIP trainer test folders.",
    )
    group.addoption(
        "--keep-mlip-trainer-results",
        action="store_true",
        default=False,
        help="Keep generated MLIP trainer test folders after the test session.",
    )


@pytest.fixture(scope="session")
def mlip_trainer_results_root(pytestconfig: pytest.Config) -> Iterator[Path]:
    base_dir = _results_base_dir(pytestconfig)
    base_dir.mkdir(parents=True, exist_ok=True)
    _ensure_gitignore(base_dir)

    session_dir = base_dir / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    session_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield session_dir
    finally:
        if not pytestconfig.getoption("--keep-mlip-trainer-results"):
            shutil.rmtree(session_dir, ignore_errors=True)


@pytest.fixture
def mlip_trainer_case_dir(request: pytest.FixtureRequest, mlip_trainer_results_root: Path) -> Path:
    case_dir = mlip_trainer_results_root / _safe_node_id(request.node.nodeid)
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir


def _results_base_dir(pytestconfig: pytest.Config) -> Path:
    configured_dir = pytestconfig.getoption("--mlip-trainer-results-dir")
    if configured_dir is None:
        return Path(__file__).parent / "_results"

    path = Path(configured_dir).expanduser()
    if path.is_absolute():
        return path
    return Path(pytestconfig.rootpath) / path


def _ensure_gitignore(base_dir: Path) -> None:
    (base_dir / ".gitignore").write_text("*\n", encoding="utf-8")


def _safe_node_id(node_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", node_id).strip("._")
    # safe_id = safe_id.replace("tests_mlip_trainers", "").replace(".py", "")
    if len(safe_id) <= 160:
        return safe_id
    return safe_id[-160]
