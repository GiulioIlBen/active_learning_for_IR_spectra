from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


class MACETrainingMetrics:
    """Parse and plot metrics emitted by MACE training runs."""

    @classmethod
    def plot_training_metrics(
        cls,
        path: Path | str,
        metrics: Optional[Sequence[str]] = None,
        output_path: Path | str | None = None,
        figsize: Optional[Tuple[float, float]] = None,
        max_columns: int = 3,
    ) -> Any:
        if max_columns < 1:
            raise ValueError("max_columns must be at least 1.")

        import matplotlib.pyplot as plt

        train_log_path = cls.resolve_training_metrics_train_log_path(path)
        records = list(cls.iter_training_metric_records_from_train_log(train_log_path))
        if len(records) == 0:
            raise ValueError(f"No epoch-indexed numeric MACE metrics found in {train_log_path}.")

        metric_names = list(metrics) if metrics is not None else cls.training_metric_names(records)
        if len(metric_names) == 0:
            raise ValueError(f"No plottable MACE metrics found in {train_log_path}.")

        n_columns = min(max_columns, len(metric_names))
        n_rows = (len(metric_names) + n_columns - 1) // n_columns
        resolved_figsize = figsize or (5.2 * n_columns, 3.4 * n_rows)
        fig, axes_grid = plt.subplots(n_rows, n_columns, figsize=resolved_figsize, sharex=True, squeeze=False)
        axes = [axis for row in axes_grid for axis in row]

        for axis, metric_name in zip(axes, metric_names, strict=False):
            series = cls.training_metric_series(records, metric_name)
            plotted = False
            for label, points in series.items():
                if len(points) == 0:
                    continue
                points = sorted(points)
                epochs = [point[0] for point in points]
                values = [point[1] for point in points]
                axis.plot(epochs, values, marker="o", markersize=3.5, linewidth=1.8, label=label)
                plotted = True
            axis.set_title(metric_name)
            axis.set_yscale("log")
            axis.set_xlabel("epoch")
            axis.set_ylabel(metric_name)
            axis.grid(which="both", ls="--", linewidth=0.8)
            axis.set_axisbelow(True)
            if plotted and len(series) > 1:
                axis.legend()
            if not plotted:
                axis.text(0.5, 0.5, "no data", transform=axis.transAxes, ha="center", va="center")

        for axis in axes[len(metric_names) :]:
            axis.set_visible(False)

        fig.suptitle(f"MACE training metrics: {train_log_path.name}")
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(Path(output_path).expanduser().resolve(strict=False))
        return fig

    @staticmethod
    def resolve_training_metrics_train_log_path(path: Path | str) -> Path:
        candidate = Path(path).expanduser().resolve(strict=False)
        if candidate.name.endswith("_train.txt"):
            if not candidate.exists():
                raise FileNotFoundError(f"MACE training log file not found: {candidate}")
            return candidate
        if not candidate.exists():
            raise FileNotFoundError(f"MACE training metrics path not found: {candidate}")
        if not candidate.is_dir():
            raise ValueError(f"MACE training metrics path must be a directory or '*_train.txt' file: {candidate}")

        search_dirs = [candidate]
        results_dir = candidate / "results"
        if results_dir.is_dir():
            search_dirs.insert(0, results_dir)

        for search_dir in search_dirs:
            train_logs = sorted(search_dir.glob("*_train.txt"))
            if len(train_logs) > 0:
                return train_logs[-1]

        recursive_train_logs = sorted(candidate.glob("**/*_train.txt"))
        if len(recursive_train_logs) > 0:
            return recursive_train_logs[-1]
        raise FileNotFoundError(f"MACETrainer could not find a training log matching '*_train.txt' below {candidate}.")

    @classmethod
    def iter_training_metric_records_from_train_log(cls, train_log_path: Path) -> Iterator[Dict[str, Any]]:
        for line_number, log_record in cls._iter_json_log_records(train_log_path):
            epoch = cls.to_int(log_record.get("epoch"))
            if epoch is None:
                continue
            record: Dict[str, Any] = {
                "epoch": epoch,
                "mode": log_record.get("mode"),
                "head": log_record.get("head"),
                "line_number": line_number,
            }
            for key, value in log_record.items():
                numeric_value = cls.to_float(value)
                if numeric_value is None or key in {"epoch", "step", "time"}:
                    continue
                record[key] = numeric_value
            if len(record) > 4:
                yield record

    @staticmethod
    def training_metric_names(records: Sequence[Dict[str, Any]]) -> List[str]:
        metadata_keys = {"epoch", "mode", "head", "line_number"}
        metric_names: list[str] = []
        for record in records:
            for key in record:
                if key not in metadata_keys and key not in metric_names:
                    metric_names.append(key)
        return metric_names

    @classmethod
    def training_metric_series(
        cls,
        records: Sequence[Dict[str, Any]],
        metric_name: str,
    ) -> Dict[str, list[tuple[int, float]]]:
        series: Dict[str, list[tuple[int, float]]] = {}
        for record in records:
            value = cls.to_float(record.get(metric_name))
            epoch = cls.to_int(record.get("epoch"))
            if value is None or epoch is None:
                continue
            mode = record.get("mode") or "unknown"
            head = record.get("head")
            label = str(mode) if head in {None, ""} else f"{mode} / {head}"
            series.setdefault(label, []).append((epoch, value))
        return series

    @classmethod
    def iter_tidy_error_metric_rows_from_train_log(cls, train_log_path: Path) -> Iterator[Dict[str, Any]]:
        for line_number, log_record in cls._iter_json_log_records(train_log_path):
            if log_record.get("mode") == "opt":
                continue

            row_info = {
                "mode": log_record.get("mode"),
                "head": log_record.get("head"),
                "time": log_record.get("time"),
                "raw_path": str(train_log_path),
                "line_number": line_number,
            }
            for metric_key, metric_value in log_record.items():
                numeric_value = cls.to_float(metric_value)
                if numeric_value is None or metric_key in {"epoch", "step", "time"}:
                    continue
                property_name, metric_name = cls.parse_tidy_metric_key(metric_key)
                yield {
                    "dataset": cls.resolve_tidy_metric_dataset(log_record),
                    "epoch": cls.to_int(log_record.get("epoch")),
                    "step": cls.to_int(log_record.get("step")),
                    "property": property_name,
                    "metrics": metric_name,
                    "unit": None,
                    "value": numeric_value,
                    "info": {**row_info, "raw_metric_key": metric_key},
                }

    @staticmethod
    def resolve_tidy_metric_dataset(log_record: Dict[str, Any]) -> Optional[str]:
        mode = log_record.get("mode")
        if mode == "eval":
            return "validation_set"
        if mode == "train":
            return "training_set"
        if mode == "test":
            return "test_set"
        return None

    @staticmethod
    def parse_tidy_metric_key(metric_key: str) -> Tuple[Optional[str], str]:
        if metric_key == "loss":
            return None, "loss"

        match = re.match(r"^(?P<metric>(?:rel_)?(?:mae|rmse|q95))_(?P<property>.+)$", metric_key)
        if match is None:
            return None, metric_key

        metric_name = match.group("metric")
        property_token = match.group("property")
        if property_token.endswith("_per_atom"):
            metric_name = f"{metric_name}_per_atom"
            property_token = property_token[: -len("_per_atom")]

        property_name = {
            "e": "energy",
            "energy": "energy",
            "f": "forces",
            "force": "forces",
            "forces": "forces",
            "mu": "dipole",
            "dipole": "dipole",
            "dipole_moment": "dipole",
        }.get(property_token)
        if property_name is None and ("stress" in property_token or "virial" in property_token):
            property_name = "stress"
        return property_name, metric_name

    @classmethod
    def collect_validation_log_metrics(cls, model_dir: Path) -> Dict[str, List[Any]]:
        train_logs = sorted(model_dir.parent.glob("logs/*.log"))
        if len(train_logs) == 0:
            return {}
        return cls.parse_validation_log_metrics(train_logs[-1].read_text(encoding="utf-8", errors="ignore"))

    @classmethod
    def parse_validation_log_metrics(cls, train_log_text: str) -> Dict[str, List[Any]]:
        table_metrics = cls.parse_validation_table_metrics(train_log_text)
        if len(table_metrics) > 0:
            return table_metrics
        return cls.parse_validation_key_value_metrics(train_log_text)

    @classmethod
    def parse_validation_table_metrics(cls, train_log_text: str) -> Dict[str, List[Any]]:
        log_metrics: Dict[str, List[Any]] = {
            "dataset": [],
            "property": [],
            "metric": [],
            "value": [],
            "source": [],
        }
        last_model_label = "final"
        lines = train_log_text.splitlines()
        line_idx = 0

        while line_idx < len(lines):
            stripped_line = lines[line_idx].strip()
            model_match = re.search(r"Loaded (?P<label>.+?) model\b", stripped_line, flags=re.IGNORECASE)
            if model_match is not None:
                last_model_label = re.sub(r"[^a-z0-9]+", "_", model_match.group("label").strip().lower()).strip("_")

            if "Error-table on TRAIN and VALID" not in stripped_line:
                line_idx += 1
                continue

            table_lines: list[str] = []
            next_idx = line_idx + 1
            while next_idx < len(lines):
                table_line = lines[next_idx].strip()
                if not table_line.startswith(("+", "|")):
                    break
                table_lines.append(table_line)
                next_idx += 1

            if len(table_lines) >= 4:
                cls._append_validation_table_rows(log_metrics, table_lines, last_model_label)
            line_idx = next_idx

        if len(log_metrics["dataset"]) == 0:
            return {}
        return log_metrics

    @classmethod
    def _append_validation_table_rows(
        cls,
        log_metrics: Dict[str, List[Any]],
        table_lines: Sequence[str],
        last_model_label: str,
    ) -> None:
        table_rows = [line for line in table_lines if line.startswith("|")]
        if len(table_rows) < 2:
            return
        headers = cls.split_ascii_table_row(table_rows[0])
        source = f"train_log_{last_model_label}"
        for row_line in table_rows[1:]:
            row_values = cls.split_ascii_table_row(row_line)
            if len(row_values) != len(headers):
                continue
            dataset = cls.dataset_from_table_config_type(row_values[0])
            if dataset is None:
                continue
            for header, raw_value in zip(headers[1:], row_values[1:], strict=True):
                parsed_header = cls.parse_validation_table_header(header)
                if parsed_header is None:
                    continue
                property_name, metric_name = parsed_header
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                log_metrics["dataset"].append(dataset)
                log_metrics["property"].append(property_name)
                log_metrics["metric"].append(metric_name)
                log_metrics["value"].append(value)
                log_metrics["source"].append(source)

    @classmethod
    def parse_validation_key_value_metrics(cls, train_log_text: str) -> Dict[str, List[Any]]:
        metrics: Dict[str, float] = {}
        key_value_pattern = re.compile(
            r"(?P<key>[A-Za-z][A-Za-z0-9_./ -]*?)\s*[:=]\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )

        for line in train_log_text.splitlines():
            lowered = line.lower()
            if "valid" not in lowered and "validation" not in lowered:
                continue
            for match in key_value_pattern.finditer(line):
                raw_key = match.group("key").strip().lower()
                key_parts = re.split(r"\s+", raw_key)
                key = next((part for part in reversed(key_parts) if "valid" in part or "validation" in part), raw_key)
                key = re.sub(r"\s+", "_", key)
                if "valid" not in key and "validation" not in key:
                    key = f"validation_{key}"
                metrics[key] = float(match.group("value"))

        if len(metrics) == 0:
            return {}

        log_metrics: Dict[str, List[Any]] = {
            "dataset": [],
            "property": [],
            "metric": [],
            "value": [],
            "source": [],
        }
        property_aliases = (
            ("energy", ("energy", "_e", "energies")),
            ("forces", ("forces", "_f", "force")),
            ("stress", ("stress", "virials", "virial")),
            ("dipole", ("dipole", "_mu", "mu")),
        )
        metric_aliases = (
            ("rmse", ("rmse",)),
            ("mae", ("mae",)),
            ("loss", ("loss",)),
        )

        for key, value in metrics.items():
            property_name = next(
                (name for name, aliases in property_aliases if any(alias in key for alias in aliases)),
                "validation",
            )
            metric_name = next(
                (name for name, aliases in metric_aliases if any(alias in key for alias in aliases)),
                key.removeprefix("validation_").removeprefix("valid_"),
            )
            log_metrics["dataset"].append("validation_set")
            log_metrics["property"].append(property_name)
            log_metrics["metric"].append(metric_name)
            log_metrics["value"].append(value)
            log_metrics["source"].append("train_log_final")
        return log_metrics

    @staticmethod
    def split_ascii_table_row(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    @staticmethod
    def dataset_from_table_config_type(config_type: str) -> Optional[str]:
        lowered = config_type.strip().lower()
        if lowered.startswith("train"):
            return "training_set"
        if lowered.startswith("valid"):
            return "validation_set"
        if lowered.startswith("test"):
            return "test_set"
        return None

    @classmethod
    def parse_validation_table_header(cls, header: str) -> Optional[Tuple[str, str]]:
        normalized = " ".join(header.strip().lower().split())
        if normalized == "config_type":
            return None

        match = re.match(r"^(?P<metric>rmse|mae|q95)\s+(?P<property>\w+)(?:\s*/\s*(?P<unit>.+))?$", normalized)
        if match is not None:
            property_name = cls.map_validation_table_property(match.group("property"))
            if property_name is None:
                return None
            metric_name = match.group("metric")
            unit = match.group("unit") or ""
            if re.search(r"(^|/)\s*atom\b", unit):
                metric_name = f"{metric_name}_per_atom"
            return property_name, metric_name

        match = re.match(r"^rel\s+(?P<property>\w+)\s+(?P<metric>rmse|mae|q95)\s*%?$", normalized)
        if match is not None:
            property_name = cls.map_validation_table_property(match.group("property"))
            if property_name is None:
                return None
            return property_name, f"rel_{match.group('metric')}"
        return None

    @staticmethod
    def map_validation_table_property(property_token: str) -> Optional[str]:
        property_name = {
            "e": "energy",
            "energy": "energy",
            "f": "forces",
            "force": "forces",
            "forces": "forces",
            "mu": "dipole",
            "dipole": "dipole",
            "dipole_moment": "dipole",
        }.get(property_token)
        if property_name is None and ("stress" in property_token or "virial" in property_token):
            property_name = "stress"
        return property_name

    @classmethod
    def _iter_json_log_records(cls, train_log_path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
        train_log_text = train_log_path.read_text(encoding="utf-8", errors="ignore")
        for line_number, raw_line in enumerate(train_log_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                log_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse MACE training log JSON at {train_log_path}:{line_number}.") from exc
            if not isinstance(log_record, dict):
                raise ValueError(
                    f"Expected a JSON object in MACE training log at {train_log_path}:{line_number}, "
                    f"got {type(log_record).__name__}."
                )
            yield line_number, log_record

    @staticmethod
    def to_float(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def to_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return None
