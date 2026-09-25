from typing import List, Union

from scm.plams import Units

from scm.active_learning.engines._shared import apply_ams_external_capabilities
from scm.active_learning.mlip.mace_trainer._optional_dependencies import _load_mace_calculator_class

MACECalculator = _load_mace_calculator_class()


class AMSMACECalculator(MACECalculator):
    def __init__(
        self,
        model_paths: Union[List, str],
        device: str,
        energy_units_to_eV: float = 1,
        length_units_to_A: float = 1,
        default_dtype="",
        charges_key="Qs",
        model_type="MACE",
        compile_mode=None,
        fullgraph=True,
        dipole_units: str = "Debye",
        mace_model_architecture=None,
        **kwargs,
    ):
        """dipole_units are those that the model predicts, the output is converted to ase units: e*Ang"""
        del mace_model_architecture
        super().__init__(
            model_paths=model_paths,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key=charges_key,
            model_type=model_type,
            compile_mode=compile_mode,
            fullgraph=fullgraph,
            **kwargs,
        )
        self.dipole_units = dipole_units

        apply_ams_external_capabilities(self, self.implemented_properties)

    def calculate(self, atoms=None, properties=None, system_changes=...):
        super().calculate(atoms, properties, system_changes)
        if "dipole" in self.results:
            self.results["dipole"] *= Units.conversion_ratio(self.dipole_units, "e*Ang")
        if "dipole_var" in self.results:
            self.results["dipole_var"] *= Units.conversion_ratio(self.dipole_units, "e*Ang")
        if "atomic_dipoles" in self.results:
            self.results["atomic_dipoles"] *= Units.conversion_ratio(self.dipole_units, "e*Ang")
