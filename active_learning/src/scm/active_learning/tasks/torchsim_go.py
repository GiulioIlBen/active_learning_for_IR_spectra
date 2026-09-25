from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from scm.active_learning.engines import ConcreteSystems, Engine, TorchEngine
from scm.active_learning.tasks.core import Task

from ._torchsim import (
    build_reporter_kwargs,
    build_trajectory_path,
    ensure_force_support,
    load_single_atoms,
    model_supports_stress,
    require_torchsim,
    resolve_registry_member,
    state_final_energy,
)

if TYPE_CHECKING:
    from scm.active_learning.results.task.torchsim_go import TorchSimGOResult


class TorchSimGOTask(Task[TorchEngine]):
    model_config = ConfigDict(extra="forbid")
    type: Literal["TorchSimGOTask"] = "TorchSimGOTask"
    task_id: str = "TorchSimGOTask"
    atomistic_system: ConcreteSystems

    save_folder: str = "."
    results_subdir: str = "results_out"
    trajectory_filename: str = "trajectory.h5"

    optimizer: str = "fire"
    convergence_kind: Literal["energy", "force"] = "force"
    energy_tol: Optional[float] = Field(default=None, gt=0.0)
    force_tol: Optional[float] = Field(default=0.05, gt=0.0)
    cell_filter: Optional[str] = None
    autobatcher: bool = False
    runner_kwargs: Dict[str, Any] = Field(default_factory=dict)
    trajectory_reporter_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_convergence(self) -> "TorchSimGOTask":
        if self.convergence_kind == "energy" and self.energy_tol is None:
            raise ValueError("energy_tol must be provided when convergence_kind='energy'.")
        if self.convergence_kind == "force" and self.force_tol is None:
            raise ValueError("force_tol must be provided when convergence_kind='force'.")
        return self

    @property
    def system_id(self) -> str:
        return self.atomistic_system.system_id

    def run(self, engine: Engine, *args, **kwargs) -> "TorchSimGOResult":
        del args
        from scm.active_learning.results.task.torchsim_go import TorchSimGOResult

        if not isinstance(engine, TorchEngine):
            raise TypeError(f"{type(engine)=} is not TorchEngine")

        save_folder = Path(kwargs.pop("save_folder", self.save_folder))
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        ts = require_torchsim()
        atoms = load_single_atoms(self.atomistic_system)
        save_folder.mkdir(parents=True, exist_ok=True)
        trajectory_path = build_trajectory_path(save_folder, self.results_subdir, self.trajectory_filename)
        model = engine.build_model()
        ensure_force_support(model)

        if self.cell_filter is not None and not model_supports_stress(model):
            raise ValueError("TorchSim cell-filter optimization requires a model that implements stress.")

        optimizer = resolve_registry_member(ts.Optimizer, self.optimizer, "optimizer")
        init_kwargs = dict(self.runner_kwargs.get("init_kwargs", {}))
        if self.cell_filter is not None:
            init_kwargs["cell_filter"] = resolve_registry_member(ts.CellFilter, self.cell_filter, "cell filter")

        optimize_kwargs: Dict[str, Any] = {
            key: value for key, value in self.runner_kwargs.items() if key != "init_kwargs"
        }
        optimize_kwargs.update(
            {
                "system": [atoms],
                "model": model,
                "optimizer": optimizer,
                "convergence_fn": self._build_convergence_fn(ts),
                "autobatcher": self.autobatcher,
                "trajectory_reporter": build_reporter_kwargs(trajectory_path, self.trajectory_reporter_kwargs),
            }
        )
        if init_kwargs:
            optimize_kwargs["init_kwargs"] = init_kwargs

        final_state = ts.optimize(**optimize_kwargs)

        return TorchSimGOResult(
            task=self,
            engine=engine,
            save_folder=str(save_folder),
            trajectory_path=str(trajectory_path),
            optimizer=self.optimizer,
            convergence_kind=self.convergence_kind,
            final_energy=state_final_energy(final_state),
        )

    def _build_convergence_fn(self, ts: Any):
        if self.convergence_kind == "energy":
            assert self.energy_tol is not None
            return ts.generate_energy_convergence_fn(self.energy_tol)
        assert self.force_tol is not None
        return ts.generate_force_convergence_fn(self.force_tol)
