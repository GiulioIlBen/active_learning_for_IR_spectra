from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Annotated

from scm.active_learning.engines import ASEEngine, ConcreteSystems, Engine
from scm.active_learning.tasks.core import Task

if TYPE_CHECKING:
    from scm.active_learning.results.task.ase_md import ASEMolecularDynamicsResult


def _require_ase_md_dependencies():
    try:
        from ase import constraints as ase_constraints
        from ase import units
        from ase.db import connect as ase_db_connect

        # from ase.md.nose_hoover_chain import NoseHooverChainNVT
        from ase.md.langevin import Langevin
        from ase.md.nptberendsen import NPTBerendsen
        from ase.md.nvtberendsen import NVTBerendsen
        from ase.md.verlet import VelocityVerlet
    except ImportError as exc:
        raise ImportError("ASEMolecularDynamics requires the optional dependency `ase` (`pip install ase`).") from exc

    dynamics_classes = {
        "VelocityVerlet": VelocityVerlet,
        "Langevin": Langevin,
        "NVTBerendsen": NVTBerendsen,
        "NPTBerendsen": NPTBerendsen,
    }
    return ase_constraints, units, ase_db_connect, dynamics_classes


@dataclass
class ASEMDContext:
    task: "ASEMolecularDynamics"
    engine: ASEEngine
    trajectory_path: Path
    dynamics: Any
    units: Any
    db: Any
    stop_triggered: bool = False
    stop_reasons: List[str] = dataclass_field(default_factory=list)


class ASEMDAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interval: int = Field(default=1, gt=0)

    @abstractmethod
    def bind(self, context: ASEMDContext): ...

    def attach(self, context: ASEMDContext) -> None:
        context.dynamics.attach(self.bind(context), interval=self.interval)


class ASEMDDBWriter(ASEMDAttachment):
    type: Literal["ASEMDDBWriter"] = "ASEMDDBWriter"

    def bind(self, context: ASEMDContext):
        def write_frame() -> None:
            atoms = context.dynamics.atoms
            atoms.info["eerr"] = context.task._energy_error(atoms)
            atoms.info["ferr"] = context.task._max_force_error(atoms)
            if context.task.save_dipole_moment:
                context.task._cache_optional_frame_properties(atoms)
            context.db.write(
                atoms,
                md_step=int(getattr(context.dynamics, "nsteps", 0)),
                md_time_fs=float(context.dynamics.get_time() / context.units.fs),
                engine_id=context.engine.engine_id,
                task_id=context.task.task_id,
                system_id=context.task.system_id,
            )

        return write_frame


class ASEMDStopOnError(ASEMDAttachment):
    type: Literal["ASEMDStopOnError"] = "ASEMDStopOnError"
    threshold: float = Field(default=0.1, gt=0.0)
    regularization: float = Field(default=0.2, gt=0.0)

    def bind(self, context: ASEMDContext):
        def stop_on_error() -> None:
            atomsi = context.dynamics.atoms
            ferr_rel = context.task._relative_force_error(atomsi, regularization=self.regularization)
            if np.max(ferr_rel) <= self.threshold:
                return
            context.stop_triggered = True
            reason = f"relative_force_error>{self.threshold}"
            if reason not in context.stop_reasons:
                context.stop_reasons.append(reason)
            context.dynamics.max_steps = 0

        return stop_on_error


class ASEMDStopOnDistance(ASEMDAttachment):
    type: Literal["ASEMDStopOnDistance"] = "ASEMDStopOnDistance"
    minimum_distance: float = Field(default=0.7, gt=0.0)

    def bind(self, context: ASEMDContext):
        def stop_on_distance() -> None:
            if context.task._minimum_interatomic_distance(context.dynamics.atoms) >= self.minimum_distance:
                return
            context.stop_triggered = True
            reason = f"minimum_distance<{self.minimum_distance}"
            if reason not in context.stop_reasons:
                context.stop_reasons.append(reason)
            context.dynamics.max_steps = 0

        return stop_on_distance


ConcreteASEMDAttachment = Annotated[
    Union[ASEMDDBWriter, ASEMDStopOnError, ASEMDStopOnDistance],
    Field(discriminator="type"),
]


class ASEMolecularDynamics(Task[ASEEngine]):
    model_config = ConfigDict(extra="forbid")
    type: Literal["ASEMolecularDynamics"] = "ASEMolecularDynamics"
    task_id: str = "ASEMolecularDynamics"
    atomistic_system: ConcreteSystems

    save_folder: str = "."
    results_subdir: str = "ase_md_results"
    trajectory_filename: str = "trajectory.db"
    append_trajectory: bool = True

    nsteps: int = Field(default=1000, ge=0)
    timestep: float = Field(default=1.0, gt=0.0)
    samplingfreq: int = Field(default=100, gt=0)
    save_dipole_moment: bool = False

    thermostat: Optional[Literal["Langevin", "Berendsen"]] = "Langevin"
    temperature: Optional[Union[float, List[float]]] = 300.0
    friction: float = Field(default=0.01, ge=0.0)

    barostat: Optional[Literal["Berendsen"]] = None
    pressure_au: Optional[float] = None
    compressibility_au: Optional[float] = None
    tau: Optional[int] = Field(default=None, gt=0)
    fixcm: bool = True

    exit_condition_freq: Optional[int] = 1
    minimum_distance: Optional[float] = 0.7
    error_threshold: Optional[float] = 0.1
    relative_force_regularization: float = Field(default=0.2, gt=0.0)

    constraints: List[str] = Field(default_factory=list)
    use_cell: bool = False
    cell: Optional[Union[List[float], List[List[float]]]] = None
    pbc: Optional[Union[bool, List[bool], Tuple[bool, bool, bool]]] = None

    attachments: List[ConcreteASEMDAttachment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_md_options(self) -> "ASEMolecularDynamics":
        if self.thermostat is not None and self.temperature is None:
            raise ValueError("temperature must be provided when thermostat is enabled.")
        if self.barostat is not None and self.thermostat != "Berendsen":
            raise ValueError("ASE Berendsen barostat currently requires thermostat='Berendsen'.")
        if self.barostat is not None and self.pressure_au is None:
            raise ValueError("pressure_au must be provided when barostat is enabled.")
        return self

    @property
    def system_id(self) -> str:
        return self.atomistic_system.system_id

    def run(self, engine: Engine, *args, **kwargs) -> "ASEMolecularDynamicsResult":
        del args
        from scm.active_learning.results.task.ase_md import ASEMolecularDynamicsResult

        if not isinstance(engine, ASEEngine):
            raise TypeError(f"{type(engine)=} is not ASEEngine")

        save_folder = Path(kwargs.pop("save_folder", self.save_folder))
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        ase_constraints, units, ase_db_connect, dynamics_classes = _require_ase_md_dependencies()

        save_folder.mkdir(parents=True, exist_ok=True)
        trajectory_path = save_folder / self.results_subdir / self.trajectory_filename
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)

        if trajectory_path.exists() and not self.append_trajectory:
            trajectory_path.unlink()

        db = ase_db_connect(str(trajectory_path), append=True)
        initial_rows = self._db_count(db)
        restart_used = initial_rows > 0
        if restart_used:
            atoms = self._last_db_row(db).toatoms(add_additional_information=True)
        else:
            atoms = self._load_initial_atoms(engine)

        self._apply_cell_and_pbc(atoms)
        self._apply_constraints(atoms, ase_constraints)
        atoms.calc = engine._build_calculator()

        dynamics = self._build_dynamics(atoms=atoms, units=units, dynamics_classes=dynamics_classes)
        context = ASEMDContext(
            task=self,
            engine=engine,
            trajectory_path=trajectory_path,
            dynamics=dynamics,
            units=units,
            db=db,
        )
        for attachment in self._resolved_attachments():
            attachment.attach(context)

        remaining_steps = max(0, self.nsteps - initial_rows * self.samplingfreq)
        if remaining_steps > 0:
            dynamics.run(remaining_steps)

        return ASEMolecularDynamicsResult(
            task=self,
            engine=engine,
            save_folder=str(save_folder),
            trajectory_path=str(trajectory_path),
            restart_used=restart_used,
            stop_triggered=context.stop_triggered,
            stop_reasons=context.stop_reasons,
            requested_steps=self.nsteps,
            executed_steps=int(getattr(dynamics, "nsteps", remaining_steps)),
            written_frames=max(0, self._db_count(db) - initial_rows),
            dynamics_type=type(dynamics).__name__,
        )

    def _resolved_attachments(self) -> List[ConcreteASEMDAttachment]:
        if self.attachments:
            return self.attachments

        attachments: List[ConcreteASEMDAttachment] = [ASEMDDBWriter(interval=self.samplingfreq)]
        if self.exit_condition_freq is None:
            return attachments
        if self.minimum_distance is not None:
            attachments.append(
                ASEMDStopOnDistance(
                    interval=self.exit_condition_freq,
                    minimum_distance=self.minimum_distance,
                )
            )
        if self.error_threshold is not None:
            attachments.append(
                ASEMDStopOnError(
                    interval=self.exit_condition_freq,
                    threshold=self.error_threshold,
                    regularization=self.relative_force_regularization,
                )
            )
        return attachments

    def _build_dynamics(self, atoms, units, dynamics_classes):
        timestep = self.timestep * units.fs
        tau_time = None if self.tau is None else self.tau * timestep

        if self.barostat == "Berendsen":
            kwargs: Dict[str, Any] = {
                "temperature_K": self._temperature_value(),
                "pressure_au": self.pressure_au,
                "compressibility_au": self.compressibility_au,
                "fixcm": self.fixcm,
            }
            if tau_time is not None:
                kwargs["taut"] = tau_time
                kwargs["taup"] = tau_time
            return dynamics_classes["NPTBerendsen"](atoms=atoms, timestep=timestep, **kwargs)

        if self.thermostat == "Berendsen":
            kwargs = {
                "temperature_K": self._temperature_value(),
                "fixcm": self.fixcm,
            }
            if tau_time is not None:
                kwargs["taut"] = tau_time
            return dynamics_classes["NVTBerendsen"](atoms=atoms, timestep=timestep, **kwargs)

        if self.thermostat == "Langevin":
            return dynamics_classes["Langevin"](
                atoms=atoms,
                timestep=timestep,
                temperature_K=self._temperature_value(),
                friction=self.friction / units.fs,
                fixcm=self.fixcm,
            )

        return dynamics_classes["VelocityVerlet"](atoms=atoms, timestep=timestep)

    def _load_initial_atoms(self, engine: ASEEngine):
        systems = self.atomistic_system.collect_molecules()
        system = systems
        if isinstance(systems, dict):
            if len(systems) != 1:
                raise ValueError("ASEMolecularDynamics requires exactly one structure in atomistic_system.")
            system = next(iter(systems.values()))
        return engine._to_ase_atoms(system)

    def _apply_cell_and_pbc(self, atoms) -> None:
        if not self.use_cell:
            return
        if self.cell is not None:
            atoms.set_cell(self.cell)
        if self.pbc is not None:
            atoms.set_pbc(self.pbc)

    def _apply_constraints(self, atoms, ase_constraints) -> None:
        if not self.constraints:
            return
        namespace = {name: getattr(ase_constraints, name) for name in dir(ase_constraints) if not name.startswith("_")}
        constraint_objects = []
        for spec in self.constraints:
            try:
                resolved = eval(spec, {"__builtins__": {}}, namespace)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Could not evaluate ASE constraint {spec!r}: {exc}") from exc
            if resolved is None:
                continue
            if isinstance(resolved, (list, tuple)):
                constraint_objects.extend(item for item in resolved if item is not None)
            else:
                constraint_objects.append(resolved)
        if constraint_objects:
            atoms.set_constraint(constraint_objects)

    def _temperature_value(self) -> float:
        if isinstance(self.temperature, list):
            if not self.temperature:
                raise ValueError("temperature list must contain at least one value.")
            return float(self.temperature[0])
        if self.temperature is None:
            raise ValueError("temperature must not be None for thermostatted dynamics.")
        return float(self.temperature)

    @staticmethod
    def _db_count(db) -> int:
        count = getattr(db, "count", None)
        if callable(count):
            return int(count())
        rows = getattr(db, "rows", None)
        if rows is not None:
            return len(rows)
        return 0

    @staticmethod
    def _last_db_row(db):
        rows = list(db.select())
        if not rows:
            raise ValueError("Could not restart ASE MD: trajectory database is empty.")
        return rows[-1]

    @staticmethod
    def _energy_error(atoms) -> float:
        """Return an energy standard deviation, not a variance.

        Ensemble uncertainty follows Kellner and Ceriotti, MLST 5, 035006
        (2024), where thresholds have the units of the predicted observable.
        """

        variance = float(atoms.calc.results.get("energy_var", 0.0))
        return float(np.sqrt(max(variance, 0.0)))

    @staticmethod
    def _force_variance(atoms) -> np.ndarray:
        calibrated_variance = atoms.calc.results.get("forces_var")
        if calibrated_variance is not None:
            variance = np.asarray(calibrated_variance, dtype=float)
            if variance.shape == (len(atoms), 3):
                return np.maximum(variance, 0.0)

        # Compatibility fallback for calculators exposing members without the
        # calibrated sample variance of Kellner and Ceriotti (2024).
        forces_comm = atoms.calc.results.get("forces_comm")
        if forces_comm is None:
            return np.zeros((len(atoms), 3), dtype=float)
        force_samples = np.asarray(forces_comm, dtype=float)
        if force_samples.ndim < 3:
            return np.zeros((len(atoms), 3), dtype=float)
        if force_samples.shape[0] < 2:
            return np.zeros((len(atoms), 3), dtype=float)
        return np.var(force_samples, axis=0, ddof=1)

    @classmethod
    def _max_force_error(cls, atoms) -> float:
        force_var = cls._force_variance(atoms)
        if force_var.size == 0:
            return 0.0
        return float(np.max(np.sqrt(np.sum(force_var, axis=1))))

    @classmethod
    def _relative_force_error(cls, atoms, regularization: float) -> np.ndarray:
        force_var = cls._force_variance(atoms)
        force = np.asarray(atoms.get_forces(), dtype=float)
        ferr = np.sqrt(np.sum(force_var, axis=1))
        return ferr / (np.linalg.norm(force, axis=1) + regularization)

    @staticmethod
    def _minimum_interatomic_distance(atoms) -> float:
        if len(atoms) < 2:
            return float("inf")
        get_all_distances = getattr(atoms, "get_all_distances", None)
        if callable(get_all_distances):
            distances = np.asarray(get_all_distances(), dtype=float)
        else:
            positions = np.asarray(atoms.get_positions(), dtype=float)
            deltas = positions[:, None, :] - positions[None, :, :]
            distances = np.linalg.norm(deltas, axis=-1)
        upper = distances[np.triu_indices(len(atoms), k=1)]
        if upper.size == 0:
            return float("inf")
        return float(np.min(upper))

    @staticmethod
    def _cache_optional_frame_properties(atoms) -> None:
        calc = getattr(atoms, "calc", None)
        if calc is None or not hasattr(calc, "results"):
            return

        getter = getattr(atoms, "get_dipole_moment", None)
        if not callable(getter):
            return

        # Request dipoles once before persisting the frame so ASE stores them in the DB row
        # when the active calculator supports this property.
        try:
            getter()
        except Exception:  # noqa: BLE001
            return
