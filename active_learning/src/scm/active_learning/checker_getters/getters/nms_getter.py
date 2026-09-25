from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Literal, Optional, Tuple, Type

import numpy as np
import scm.plams as plams
from ase import Atoms
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces import InMemoryMolData

from scm.active_learning.checker_getters.getters.core import Getter
from scm.active_learning.results import EngineCheckResult, FrameGetterResult
from scm.active_learning.results.task import AMSTaskResult
from scm.active_learning.tasks import AMSTask, Task

__all__ = ["NMSGetters"]


class NMSFrequenciesNotFoundError(ValueError):
    pass


def _attach_input_metadata(jobresults: plams.AMSResults, atoms_list) -> None:
    try:
        mol_in = jobresults.get_input_molecule()
    except Exception:
        return
    if isinstance(mol_in, plams.Molecule):
        info = mol_in[1].properties.as_dict()
        for atoms in atoms_list:
            atoms.info.update(info)


@dataclass(frozen=True)
class NMSData:
    atoms_equilibrium: Atoms
    force_constants: np.ndarray
    normal_modes: np.ndarray
    frequencies: np.ndarray

    def filter(self, min_wavenumber: int) -> "NMSData":
        keep_indices = self.frequencies > min_wavenumber
        filtered = NMSData(
            atoms_equilibrium=self.atoms_equilibrium,
            force_constants=self.force_constants[keep_indices],
            normal_modes=self.normal_modes[keep_indices],
            frequencies=self.frequencies[keep_indices],
        )
        if filtered.frequencies.size == 0:
            max_frequency = float(np.max(self.frequencies)) if self.frequencies.size else None
            raise NMSFrequenciesNotFoundError(
                "NMS requires at least one frequency above "
                f"{min_wavenumber} cm^-1; found none (maximum={max_frequency} cm^-1)."
            )
        return filtered

    def generate(
        self,
        max_n_structures: int,
        temperature: float,
        return_equ_atoms: bool,
        thermal_en_theory: Literal["classical", "quantum"],
        seed: Optional[int],
    ) -> List[Atoms]:
        if seed is not None:
            np.random.seed(seed)

        n_normal_modes = self.normal_modes.shape[0]
        n_atoms = len(self.atoms_equilibrium)
        atoms_list = [self.atoms_equilibrium.copy()] if return_equ_atoms else []

        k_b = plams.Units.constants["k_B"]
        if thermal_en_theory == "classical":
            k_b_hartree = k_b * plams.Units.conversion_ratio(inp="J", out="hartree")
            thermal_energy = 3 * n_atoms * k_b_hartree * temperature
        else:
            hbar = 1.054571817e-34
            hbar_ha_s = hbar * 2.294e17
            freq_hz = self.frequencies * plams.Units.constants["speed_of_light"] / 100
            mean_number_modes_of_vib = 1 / (np.exp(hbar * freq_hz / (k_b * temperature)) - 1)
            thermal_energy = freq_hz * hbar_ha_s * (0.5 + mean_number_modes_of_vib) * n_atoms * 3

        bohr_to_ang = plams.Units.conversion_ratio(inp="Bohr", out="Angstrom")
        for _ in range(max_n_structures):
            rand_values = np.random.rand(n_normal_modes)
            desired_sum = np.random.uniform(0, 1)
            coefficients = rand_values / (np.sum(rand_values) / desired_sum)
            plus_minus = np.random.randint(2, size=n_normal_modes) * 2 - 1
            displacements = plus_minus * np.sqrt(coefficients * thermal_energy / self.force_constants)
            displacements = displacements.reshape(n_normal_modes, 1, 1) * bohr_to_ang
            total_displacement = np.sum(displacements * self.normal_modes, axis=0)

            new_atoms = self.atoms_equilibrium.copy()
            new_atoms.positions = new_atoms.positions + total_displacement
            new_atoms.set_velocities(-total_displacement)
            atoms_list.append(new_atoms)

        return atoms_list


class NMSGetters(Getter[AMSTaskResult]):
    """
    Simple normal mode sampling that keeps everything in memory.
    """

    type: Literal["NMSGetters"] = "NMSGetters"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSTask,)
    max_n_structures: int
    temperature: float = 300
    min_wavenumber: int = 500
    return_equ_atoms: bool = False
    seed: Optional[int] = None
    thermal_en_theory: Literal["classical", "quantum"] = "quantum"
    engine_file: Optional[str] = None
    getter_id: str = "NMS"

    def _extract_data(self, result: AMSTaskResult) -> NMSData:
        jobresults = result.plams_results
        engine_file = self.engine_file
        if engine_file == "auto":
            engine_file = jobresults.rkfpath(file="engine")
            if engine_file is not None:
                engine_file = plams.kftools.KFFile(engine_file).reader.main_section

        return NMSData(
            atoms_equilibrium=plams.toASE(jobresults.get_main_molecule()),
            force_constants=np.asarray(jobresults.get_force_constants(engine=engine_file)),
            normal_modes=np.asarray(
                jobresults.get_normal_modes(engine=engine_file, mass_weighted_hessian_eigenvectors=False)
            ),
            frequencies=np.asarray(jobresults.get_frequencies(engine=engine_file)),
        )

    def run(
        self,
        result: AMSTaskResult,
        checker_results: List[EngineCheckResult],
        **kwargs,
    ) -> FrameGetterResult:
        jobresults = result.plams_results
        if jobresults is None:
            dataset = InMemoryMolData.create(distance_unit="Ang", available_properties=[])
            return FrameGetterResult(
                engine_id=result.from_engine.engine_id,
                task_id=result.from_task.task_id,
                system_id=result.from_task.system_id,
                checker_id=(checker_results[-1].checker_id if len(checker_results) > 0 else "-empty-"),
                getter_id=self.getter_id,
                dataset=dataset,
            )

        task = jobresults.job.get_task()
        if task is not None and task.lower() != "GeometryOptimization".lower():
            raise ValueError(
                (
                    f"{jobresults.job.get_task()=} is not GeometryOptimization,"
                    "but NMSGetters can be applied only to GeometryOptimization tasks"
                )
            )
        atoms_list = (
            self._extract_data(result)
            .filter(self.min_wavenumber)
            .generate(
                max_n_structures=self.max_n_structures,
                temperature=self.temperature,
                return_equ_atoms=self.return_equ_atoms,
                thermal_en_theory=self.thermal_en_theory,
                seed=self.seed,
            )
        )

        _attach_input_metadata(jobresults, atoms_list)
        dataset = InMemoryMolData.create(distance_unit="Ang", available_properties=[])

        def make_s(atoms):
            return ChemDataEntry(system=atoms)

        dataset.add_systems(map(make_s, atoms_list))
        return FrameGetterResult(
            engine_id=result.from_engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id=(checker_results[-1].checker_id if len(checker_results) > 0 else "-empty-"),
            getter_id=self.getter_id,
            dataset=dataset,
        )
