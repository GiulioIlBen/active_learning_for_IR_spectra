from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from pydantic import ConfigDict, Field

from scm.active_learning.engines import ConcreteSystems, Engine, TorchEngine
from scm.active_learning.tasks.core import Task

from ._torchsim import (
    build_reporter_kwargs,
    build_trajectory_path,
    load_single_atoms,
    require_torchsim,
    resolve_registry_member,
    state_final_energy,
    state_step_count,
)

if TYPE_CHECKING:
    from scm.active_learning.results.task.torchsim_md import TorchSimMDResult


class TorchSimMDTask(Task[TorchEngine]):
    model_config = ConfigDict(extra="forbid")
    type: Literal["TorchSimMDTask"] = "TorchSimMDTask"
    task_id: str = "TorchSimMDTask"
    atomistic_system: ConcreteSystems

    save_folder: str = "."
    results_subdir: str = "results_out"
    trajectory_filename: str = "trajectory.h5"

    integrator: str = "nvt_langevin"
    nsteps: int = Field(default=1000, ge=0)
    timestep: float = Field(default=0.001, gt=0.0)
    temperature: Optional[float] = None
    autobatcher: bool = False
    runner_kwargs: Dict[str, Any] = Field(default_factory=dict)
    trajectory_reporter_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def system_id(self) -> str:
        return self.atomistic_system.system_id

    def run(self, engine: Engine, *args, **kwargs) -> "TorchSimMDResult":
        del args
        from scm.active_learning.results.task.torchsim_md import TorchSimMDResult

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
        integrator = resolve_registry_member(ts.Integrator, self.integrator, "integrator")

        integrate_kwargs: Dict[str, Any] = dict(self.runner_kwargs)
        integrate_kwargs.update(
            {
                "system": [atoms],
                "model": model,
                "integrator": integrator,
                "n_steps": self.nsteps,
                "temperature": 0.0 if self.temperature is None else self.temperature,
                "timestep": self.timestep,
                "autobatcher": self.autobatcher,
                "trajectory_reporter": build_reporter_kwargs(trajectory_path, self.trajectory_reporter_kwargs),
            }
        )

        final_state = ts.integrate(**integrate_kwargs)

        return TorchSimMDResult(
            task=self,
            engine=engine,
            save_folder=str(save_folder),
            trajectory_path=str(trajectory_path),
            requested_steps=self.nsteps,
            executed_steps=state_step_count(final_state, default=self.nsteps),
            integrator=self.integrator,
            final_energy=state_final_energy(final_state),
        )
