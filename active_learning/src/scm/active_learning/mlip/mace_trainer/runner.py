from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from scm.active_learning.mlip.mace_trainer._optional_dependencies import (
    _load_mace_finetuning_utils_module,
    _load_mace_model_script_utils_module,
    _load_mace_train_main,
    _load_mace_train_module,
    _load_torch_module,
)


def _clone_parameter(torch_module: Any, tensor: Any) -> Any:
    return torch_module.nn.Parameter(tensor.clone())


def _copy_optional_linear_weights(torch_module: Any, target: Any, source: Any) -> None:
    for attr_name in ("linear", "linear_1", "linear_2"):
        target_attr = getattr(target, attr_name, None)
        source_attr = getattr(source, attr_name, None)
        if target_attr is None or source_attr is None:
            continue
        if hasattr(target_attr, "weight") and hasattr(source_attr, "weight"):
            target_attr.weight = _clone_parameter(torch_module, source_attr.weight)


def _patched_mace_load_foundations(
    model: Any,
    model_foundations: Any,
    table: Any,
    load_readout: bool = False,
    use_shift: bool = False,
    use_scale: bool = True,
    max_L: int = 2,
    default_dtype: Any = None,
) -> Any:
    original_loader = getattr(_load_mace_finetuning_utils_module(), "load_foundations", None)
    if not callable(original_loader):
        raise AttributeError("mace.tools.finetuning_utils.load_foundations was not found or is not callable.")

    if float(model_foundations.r_max) != float(model.r_max):
        raise AssertionError("Foundation model r_max must match the fine-tuning model.")

    foundation_species = [int(z) for z in model_foundations.atomic_numbers]
    target_species = [int(z) for z in table.zs]
    if foundation_species != target_species:
        return original_loader(
            model,
            model_foundations,
            table,
            load_readout=load_readout,
            use_shift=use_shift,
            use_scale=use_scale,
            max_L=max_L,
        )

    torch_module = _load_torch_module()
    model.node_embedding.linear.weight = _clone_parameter(
        torch_module,
        model_foundations.node_embedding.linear.weight,
    )

    if model.radial_embedding.bessel_fn.__class__.__name__ == "BesselBasis":
        model.radial_embedding.bessel_fn.bessel_weights = _clone_parameter(
            torch_module,
            model_foundations.radial_embedding.bessel_fn.bessel_weights,
        )

    for interaction, foundation_interaction in zip(model.interactions, model_foundations.interactions, strict=True):
        interaction.linear_up.weight = _clone_parameter(torch_module, foundation_interaction.linear_up.weight)
        interaction.avg_num_neighbors = foundation_interaction.avg_num_neighbors

        layer_idx = 0
        while True:
            layer_name = f"layer{layer_idx}"
            target_layer = getattr(interaction.conv_tp_weights, layer_name, None)
            source_layer = getattr(foundation_interaction.conv_tp_weights, layer_name, None)
            if target_layer is None or source_layer is None:
                break
            if layer_idx == 0:
                num_radial = model.radial_embedding.out_dim
                target_layer.weight = _clone_parameter(torch_module, source_layer.weight[:num_radial, :])
            else:
                target_layer.weight = _clone_parameter(torch_module, source_layer.weight)
            layer_idx += 1

        interaction.linear.weight = _clone_parameter(torch_module, foundation_interaction.linear.weight)
        interaction.skip_tp.weight = _clone_parameter(torch_module, foundation_interaction.skip_tp.weight)

    for product, foundation_product in zip(model.products, model_foundations.products, strict=True):
        for contraction, foundation_contraction in zip(
            product.symmetric_contractions.contractions,
            foundation_product.symmetric_contractions.contractions,
            strict=True,
        ):
            contraction.weights_max = _clone_parameter(torch_module, foundation_contraction.weights_max)
            for idx in range(len(contraction.weights)):
                contraction.weights[idx] = _clone_parameter(torch_module, foundation_contraction.weights[idx])
        product.linear.weight = _clone_parameter(torch_module, foundation_product.linear.weight)

    if load_readout:
        for readout, foundation_readout in zip(model.readouts, model_foundations.readouts, strict=True):
            _copy_optional_linear_weights(torch_module, readout, foundation_readout)

    foundation_scale_shift = getattr(model_foundations, "scale_shift", None)
    model_scale_shift = getattr(model, "scale_shift", None)
    if foundation_scale_shift is not None and model_scale_shift is not None:
        if use_scale:
            model.scale_shift.scale = model_foundations.scale_shift.scale.clone()
        if use_shift:
            model.scale_shift.shift = model_foundations.scale_shift.shift.clone()

    return model


@contextmanager
def _working_directory(path: Path | str | None):
    if path is None:
        yield
        return
    old_cwd = Path.cwd()
    new_cwd = Path(path).expanduser().resolve(strict=False)
    new_cwd.mkdir(parents=True, exist_ok=True)
    os.chdir(new_cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextmanager
def _patched_run_train_load_foundations(
    patched_load_foundations: Optional[Callable[..., Any]] = None,
):
    if patched_load_foundations is None:
        yield
        return

    patch_targets: list[tuple[Any, str, Callable[..., Any]]] = []

    run_train_module = _load_mace_train_module()
    original_run_train_loader = getattr(run_train_module, "load_foundations", None)
    if callable(original_run_train_loader):
        patch_targets.append((run_train_module, "load_foundations", original_run_train_loader))

    model_script_utils_module = _load_mace_model_script_utils_module()
    original_model_script_utils_loader = getattr(model_script_utils_module, "load_foundations_elements", None)
    if callable(original_model_script_utils_loader):
        patch_targets.append(
            (
                model_script_utils_module,
                "load_foundations_elements",
                original_model_script_utils_loader,
            )
        )

    if len(patch_targets) == 0:
        raise AttributeError(
            "No supported MACE load foundations hook was found. "
            "Expected mace.cli.run_train.load_foundations or "
            "mace.tools.model_script_utils.load_foundations_elements."
        )

    for module, attr_name, _ in patch_targets:
        setattr(module, attr_name, patched_load_foundations)
    try:
        yield
    finally:
        for module, attr_name, original_loader in reversed(patch_targets):
            setattr(module, attr_name, original_loader)


def train_mace(
    config_file_path: Path | str,
    workdir: Path | str | None = None,
    patched_load_foundations: Optional[Callable[..., Any]] = None,
):
    mace_run_train_main = _load_mace_train_main()
    config_path = str(Path(config_file_path).expanduser().resolve(strict=False))
    logging.getLogger().handlers.clear()
    old_argv = list(sys.argv)
    try:
        sys.argv = ["program", "--config", config_path]
        with _patched_run_train_load_foundations(patched_load_foundations), _working_directory(workdir):
            mace_run_train_main()
    finally:
        sys.argv = old_argv


def run_train_mace_subprocess(
    config_file_path: Path | str,
    workdir: Path | str,
    log_path: Path | str,
    patch_load_foundations: bool = False,
    gpu_id: Optional[int] = None,
    suppress_warnings: bool = True,
) -> None:
    """Run one MACE training in a separate Python process, so several can run at the same time."""
    command = [
        sys.executable,
        "-m",
        "scm.active_learning.mlip.mace_trainer.runner",
        "--config",
        str(config_file_path),
        "--workdir",
        str(workdir),
    ]
    if patch_load_foundations:
        command.append("--patch-load-foundations")
    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if suppress_warnings:
        env["PYTHONWARNINGS"] = "ignore"
    log_path = Path(log_path)
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, env=env, check=False)
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        raise RuntimeError(f"MACE training failed (exit code {completed.returncode}), see {log_path}:\n{tail}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train one MACE model from a config file (used for parallel members).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--patch-load-foundations", action="store_true")
    args = parser.parse_args(argv)
    train_mace(
        config_file_path=args.config,
        workdir=args.workdir,
        patched_load_foundations=_patched_mace_load_foundations if args.patch_load_foundations else None,
    )


if __name__ == "__main__":
    main()
