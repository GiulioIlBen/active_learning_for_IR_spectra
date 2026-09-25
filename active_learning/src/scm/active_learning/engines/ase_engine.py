from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, Iterable, List, Literal, Optional, Sequence

from pydantic import BaseModel, JsonValue
from scm.moliterate import ConcreteInterfaces, PropertyInfo
from scm.plams import Settings
from scm.plams.interfaces.adfsuite.ams import hybrid_committee_engine_settings

from scm.active_learning.task_parallelization import ParallelStrategy

from ._shared import extract_system, resolve_dotted_object, to_ase_atoms
from .ams import AMSEngine
from .core import Engine


class ASEPythonSettings(BaseModel):
    type: Literal["Current", "Uv", "Conda"] = "Current"
    info: Optional[str] = None

    @property
    def settings(self):
        s = Settings()
        s.Type = self.type
        if self.info is not None:
            s[self.type] = self.info
        return s


class ASEEngine(Engine[ParallelStrategy]):
    type: Literal["ASEEngine"] = "ASEEngine"
    calculator: str
    python: ASEPythonSettings = ASEPythonSettings()
    calculator_kwargs: Dict[str, JsonValue] = {}
    finetunable: bool = False

    _PROPERTY_ALIASES: ClassVar[Dict[str, str]] = {
        "energy": "energy",
        "potential_energy": "energy",
        "free_energy": "free_energy",
        "forces": "forces",
        "stress": "stress",
        "dipole": "dipole_moment",
        "dipole_moment": "dipole_moment",
        "dipolemoment": "dipole_moment",
        "atomic_polar_tensor": "dipole_gradients",
        "apt": "dipole_gradients",
        "dipole_gradients": "dipole_gradients",
        "dipole_derivatives": "dipole_gradients",
        "charges": "charges",
        "magmom": "magmom",
        "magmoms": "magmoms",
    }
    _PROPERTY_NATIVE_UNITS: ClassVar[Dict[str, str]] = {
        "energy": "eV",
        "free_energy": "eV",
        "forces": "eV/Ang",
        "stress": "eV/Ang^3",
        "dipole_moment": "e*Ang",
        "dipole_gradients": "e",
        "charges": "e",
    }

    @classmethod
    def build(
        cls,
        *,
        engine_id: str,
        calculator: str,
        python: ASEPythonSettings | Dict[str, JsonValue] | None = None,
        calculator_kwargs: Dict[str, JsonValue] | None = None,
        **kwargs: JsonValue,
    ) -> "ASEEngine":
        combined_kwargs = dict(calculator_kwargs or {})
        combined_kwargs.update(kwargs)
        return cls(
            engine_id=engine_id,
            calculator=calculator,
            python=python or ASEPythonSettings(),
            calculator_kwargs=combined_kwargs,
        )

    @classmethod
    def build_tblite(
        cls,
        engine_id: str,
        method: str = "GFN2-xTB",
        calculator_kwargs: Dict[str, JsonValue] | None = None,
        **kwargs: JsonValue,
    ) -> "ASEEngine":
        return cls.build(
            engine_id=engine_id,
            calculator="tblite.ase.TBLite",
            calculator_kwargs=calculator_kwargs,
            method=method,
            **kwargs,
        )

    def is_finetunable(self) -> bool:
        return self.finetunable

    @property
    def has_committee(self) -> bool:
        return len(self._model_paths_list()) > 1

    @property
    def committee_members(self) -> List["ASEEngine"]:
        model_paths = self._model_paths_list()
        if len(model_paths) <= 1:
            return []
        members: List[ASEEngine] = []
        for index, model_path in enumerate(model_paths):
            calculator_kwargs = dict(self.calculator_kwargs)
            calculator_kwargs["model_paths"] = model_path
            members.append(
                ASEEngine(
                    engine_id=f"{self.engine_id}_m{index:03d}",
                    calculator=self.calculator,
                    python=self.python,
                    calculator_kwargs=calculator_kwargs,
                    finetunable=self.finetunable,
                )
            )
        return members

    def _model_paths_list(self) -> List[str]:
        value = self.calculator_kwargs.get("model_paths")
        if isinstance(value, str) and value:
            return [value]
        if isinstance(value, (list, tuple)):
            return [item for item in value if isinstance(item, str) and item]
        return []

    def _validate_properties(self, requested_properties: List[PropertyInfo]) -> List[str]:
        names = [prop.name for prop in requested_properties]
        unsupported = {name for name in names if name.lower() not in self._PROPERTY_ALIASES}
        if unsupported:
            raise ValueError(
                f"Not available properties: {unsupported} | Available are: {sorted(self._PROPERTY_ALIASES.keys())}"
            )
        return [self._PROPERTY_ALIASES[name.lower()] for name in names]

    def run_single_point(
        self,
        properties: List[PropertyInfo],
        dataset: ConcreteInterfaces,
        parallel_settings: ParallelStrategy,
        **kwargs,
    ) -> Iterable[Dict[str | Literal["FAILURE"], Any | str]]:
        del parallel_settings, kwargs
        canonical_properties = self._validate_properties(properties)
        calculator = self._build_calculator()

        for row in dataset:
            try:
                atoms = self._to_ase_atoms(self._extract_system(row))
                atoms.calc = calculator

                computed: Dict[str, Any] = {}
                result: Dict[str | Literal["FAILURE"], Any | str] = {}
                for prop, canonical in zip(properties, canonical_properties, strict=True):
                    if canonical not in computed:
                        getter = self._property_getters()[canonical]
                        computed[canonical] = getter(atoms)
                    result[prop.name] = self._convert_property_value(computed[canonical], prop, canonical)
                yield result
            except Exception as exc:  # noqa: BLE001
                yield {"FAILURE": str(exc) or exc.__class__.__name__}

    def _build_calculator(self):
        calculator_cls = self._resolve_calculator_class()
        return calculator_cls(**self.calculator_kwargs)

    def _resolve_calculator_class(self):
        try:
            return resolve_dotted_object(
                self.calculator,
                kind="calculator class",
            )
        except ValueError as exc:
            if "." not in self.calculator:
                raise ValueError(
                    "calculator class must be a fully qualified class path like 'package.module.Symbol'"
                ) from exc
            raise

    @staticmethod
    def _extract_system(row: Any) -> Any:
        return extract_system(row)

    @staticmethod
    def _to_ase_atoms(system: Any):
        return to_ase_atoms(system)

    @staticmethod
    def _property_getters() -> Dict[str, Callable[[Any], Any]]:
        return {
            "energy": lambda atoms: atoms.get_potential_energy(),
            "free_energy": lambda atoms: atoms.get_potential_energy(force_consistent=True),
            "forces": lambda atoms: atoms.get_forces(),
            "stress": lambda atoms: atoms.get_stress(),
            "dipole_moment": lambda atoms: atoms.get_dipole_moment(),
            "dipole_gradients": lambda atoms: ASEEngine._get_calculator_property(
                atoms,
                candidate_names=("atomic_polar_tensor", "apt", "dipole_gradients", "dipole_derivatives"),
            ),
            "charges": lambda atoms: atoms.get_charges(),
            "magmom": lambda atoms: atoms.get_magnetic_moment(),
            "magmoms": lambda atoms: atoms.get_magnetic_moments(),
        }

    @staticmethod
    def _get_calculator_property(atoms: Any, candidate_names: Sequence[str]) -> Any:
        calculator = getattr(atoms, "calc", None)
        if calculator is None:
            raise ValueError("ASE atoms object has no attached calculator.")

        getter = getattr(calculator, "get_property", None)
        if getter is not None:
            for name in candidate_names:
                try:
                    return getter(name, atoms, allow_calculation=True)
                except Exception:  # noqa: BLE001
                    continue

        results = getattr(calculator, "results", {})
        for name in candidate_names:
            if name in results:
                return results[name]

        raise ValueError(f"Calculator {calculator.__class__.__name__} does not provide any of {candidate_names!r}.")

    def _convert_property_value(self, value: Any, prop: PropertyInfo, canonical: str) -> Any:
        native_unit = self._PROPERTY_NATIVE_UNITS.get(canonical)
        if native_unit is None:
            return value
        return value * prop.unit_conversion_from(native_unit)

    @property
    def settings(self) -> Settings:
        if self.has_committee:
            return hybrid_committee_engine_settings([member.settings for member in self.committee_members])
        return self._single_engine_settings()

    def _single_engine_settings(self) -> Settings:
        # Compose an AMS ASE engine block while allowing caller-provided overrides.
        s = Settings()
        s.input.ASE.Type = "Import"
        s.input.ASE.Python = self.python.settings
        s.input.ASE.Import = self.calculator
        sep = "\n    "
        Arguments = sep + sep.join(f"{key}={value!r}" for key, value in self.calculator_kwargs.items()) + "\n  End"
        s.input.ASE.Arguments = Arguments
        return s

    @property
    def ams_engine(self) -> AMSEngine:
        return AMSEngine(
            engine_id=self.engine_id,
            source_settings=self.settings.as_dict(),
            finetunable=self.finetunable,
        )
