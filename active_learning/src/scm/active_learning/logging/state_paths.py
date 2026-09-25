from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable

from scm.plams import Settings

PATH_POLICY_REGISTRY: list[str] = [
    r".*settings_json_path$",
    r".*save_folder$",
    r".*trajectory_path$",
    r".*data_source$",
    r".*run_directory$",
    r".*.data_source.data_source$",
    r".*labelled.task.out_path$",
    r".*.source_settings..*.input.MLPotential.ParameterDir$",
    r"journey.history.*.previous_job$",
    r".*.journey_advance_results.info.previous_job$",
    r"journey.previous_job_$",
    r"journey.data_source$",
    r"journey.molecules$",
    r"splitting.dataset.data_source$",
    r"labeller.out_path$",
    r"labeller_engine.job_path$",
    r"labeller_engine.calculator_kwargs.model_file$",
    r"labeller_engine.calculator_kwargs.model_paths$",
    r"labeller_engine.calculator_kwargs.base_calc$",
    r"labeller_engine.model_files$",
    r"start_engine.job_path$",
    r"start_engine.calculator_kwargs.model_file$",
    r"start_engine.calculator_kwargs.model_paths$",
    r"start_engine.calculator_kwargs.base_calc$",
    r"current_state.start_engine.job_path$",
    r"current_state.start_engine.calculator_kwargs.model_file$",
    r"current_state.start_engine.calculator_kwargs.model_paths$",
    r"current_state.start_engine.calculator_kwargs.base_calc$",
    r".*.start_engine.model_files$",
    r"result.final_engine.job_path$",
    r"result.iterations.*.start_engine.job_path$",
    r".*train_results.engine.job_path$",
    r"result.iterations.*.train_results.engine.job_path$",
    r".*.final_engine.calculator_kwargs.model_file$",
    r".*.final_engine.calculator_kwargs.model_paths$",
    r".*.final_engine.calculator_kwargs.base_calc$",
    r".*.start_engine.calculator_kwargs.model_file$",
    r".*.start_engine.calculator_kwargs.model_paths$",
    r".*.start_engine.calculator_kwargs.base_calc$",
    r".*.train_results.engine.calculator_kwargs.model_file$",
    r".*.train_results.engine.calculator_kwargs.model_paths$",
    r".*.train_results.engine.calculator_kwargs.base_calc$",
    r"(^|.*\.)model_paths\.\d+$",
    r"result.final_engine.model_files$",
    r"result.iterations.*.start_engine.model_files$",
    r"result.iterations.*.train_results.engine.model_files$",
    r"(^|.*\.)model_files\.\d+$",
    r"callbacks.*.root_dir.run_root$",
    r"callbacks.*.root_dir.base_dir$",
    r".*sp_labeller.out_path$",
    r".*training_metrics_csv_path$",
    # "callbacks.*.logging_file",
    # "mlip_trainer.data.dataset_subfolder",
    # "mlip_trainer.data.training_file_name",
    # "mlip_trainer.data.validation_file_name",
    # "mlip_trainer.data.config_file_name",
    # "mlip_trainer.data.model_subfolder",
    r".*data.folder$",
    # r"mlip_trainer.data.folder$",
    # "mlip_trainer.run.work_dir_name",
    # "mlip_trainer.model_filename",
    r"mlip_trainer.datapath$",
    # "mlip_trainer.model_dir",
    # "mlip_trainer.data.folder",
    # "mlip_trainer.run.folder",
]


def rewrite_payload_paths(payload: Dict, state_dir: Path, relative: bool = False) -> Dict:
    s = Settings(payload)
    flatten = s.flatten()

    paths = list(collect_paths(flatten))

    for p in paths:
        v = flatten[p]
        if relative and not Path(v).exists():
            continue
        # Assign on the flat mapping directly: merging with `Settings += flatten` corrupts lists with >= 11 items
        # after unflatten (e.g. [0..10] -> [0, 1, 9, 9, ...]).
        dict.__setitem__(flatten, p, fix_path(v, state_dir, relative))
    return flatten.unflatten().as_dict()


def collect_paths(flatten_settings: Settings):
    for k, v in flatten_settings.items():
        if not isinstance(v, (str, Path)):
            continue
        k_str = ".".join(map(str, k))
        for option in PATH_POLICY_REGISTRY:
            if re.match(option, k_str):
                yield k


def collect_runtime_paths(payload: Dict) -> Iterable[Path]:
    flatten = Settings(payload).flatten()
    for key in collect_paths(flatten):
        value = flatten[key]
        if str(value) == "":
            continue
        raw_path = Path(value).expanduser()
        if raw_path.is_absolute():
            yield raw_path.resolve(strict=False)
            continue
        yield _resolve_runtime_path(raw_path)


def fix_path(value: Path | str, state_dir: Path, relative: bool):
    if str(value) == "":
        return value

    raw_path = Path(value).expanduser()
    if relative:
        return _relativize_path(raw_path, state_dir)
    return _absolutize_path(raw_path, state_dir)


def _resolve_runtime_path(path: Path) -> Path:
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def _relativize_path(value: Path, state_dir: Path) -> Any:
    runtime_path = value if value.is_absolute() else _resolve_runtime_path(value)
    if not runtime_path.exists():
        return str(value)
    return os.path.relpath(str(runtime_path), start=str(state_dir))


def _absolutize_path(value: Path, state_dir: Path) -> Any:
    if value.is_absolute():
        return str(value)
    return str((state_dir / value).expanduser().resolve(strict=False))
