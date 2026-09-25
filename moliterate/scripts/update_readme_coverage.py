#!/usr/bin/env python3
"""Update README coverage block from coverage.py JSON output."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

LEGACY_TEST_NAME_PREFIX = "Test Name:"


def test_name_prefix(test_index: int) -> str:
    return f"Test{test_index} | Name:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        dest="json_path",
        default="coverage.json",
        help="Path to coverage.json (default: coverage.json)",
    )
    parser.add_argument(
        "--pytest-log",
        dest="pytest_log_path",
        default="pytest.log",
        help="Path to pytest log (default: pytest.log)",
    )
    parser.add_argument(
        "--readme",
        dest="readme_path",
        default="README.md",
        help="Path to README.md (default: README.md)",
    )
    parser.add_argument(
        "--test",
        default="NotFound",
        help="Key for the last test done",
    )
    parser.add_argument(
        "--test-index",
        dest="test_index",
        type=int,
        default=0,
        help="Coverage line index (default: 0). Example: 0 -> 'Test0 | Name:'",
    )
    return parser.parse_args()


def load_coverage_percent(json_path: Path) -> float:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return float(data["totals"]["percent_covered"])


def load_pytest_summary(pytest_log_path: Path) -> str:
    lines = pytest_log_path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        summary = line.strip()
        if summary:
            return summary.replace("=", "")
    return "No pytest output found"


def update_readme(
    readme_path: Path,
    percent: float,
    pytest_summary: str,
    last_test: str = "",
    test_index: int = 0,
) -> None:
    readme_text = readme_path.read_text(encoding="utf-8")
    lines = readme_text.splitlines(keepends=True)

    target_prefix = test_name_prefix(test_index)
    test_info = f"{target_prefix} {last_test}, on {datetime.date.today()}"
    coverage_line = f"Coverage: {percent:.1f}%"
    tests_line = f"Tests: {pytest_summary}"
    updated_line = f"{test_info} | {coverage_line} | {tests_line}"

    # Support migration from the old format "Test Name:" when test_index=0.
    accepted_prefixes = [target_prefix]
    if test_index == 0:
        accepted_prefixes.append(LEGACY_TEST_NAME_PREFIX)

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if any(stripped.startswith(prefix) for prefix in accepted_prefixes):
            line_ending = "\n" if line.endswith("\n") else ""
            lines[idx] = f"{updated_line}{line_ending}"
            readme_path.write_text("".join(lines), encoding="utf-8")
            return

    insert_at = 0
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("Test") and "| Name:" in line:
            insert_at = idx + 1

    line_to_insert = f"{updated_line}\n"
    if insert_at == 0 and len(lines) >= 1 and lines[0].startswith("#"):
        if len(lines) >= 2 and lines[1].strip() == "":
            insert_at = 2
            line_to_insert = f"{updated_line}\n\n"
        else:
            insert_at = 1
            line_to_insert = f"\n{updated_line}\n"

    lines.insert(insert_at, line_to_insert)
    readme_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    readme_path = Path(args.readme_path)
    pytest_log_path = Path(args.pytest_log_path)
    last_test = str(args.test)
    test_index = int(args.test_index)

    if not json_path.exists():
        raise SystemExit(
            f"{json_path} not found. Run: python -m coverage json -o {json_path.name}"
        )
    if not pytest_log_path.exists():
        raise SystemExit(f"{pytest_log_path} not found. Run: pytest | tee {pytest_log_path.name}")

    percent = load_coverage_percent(json_path)
    pytest_summary = load_pytest_summary(pytest_log_path)
    update_readme(readme_path, percent, pytest_summary, last_test, test_index=test_index)


if __name__ == "__main__":
    main()
