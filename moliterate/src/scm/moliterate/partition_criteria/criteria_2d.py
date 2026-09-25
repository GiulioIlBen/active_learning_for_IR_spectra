from __future__ import annotations

import traceback
import warnings
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from pydantic import PrivateAttr

from scm.moliterate.core import BaseChemDataSet, BasePartitionCriteria
from scm.moliterate.transforms import MinMaxAtomsDistance

__all__ = ["BasePartitionCriteria", "MBTRCriterion", "GyrationRFmaxCriterion"]


def _ensure_ase_get_constraints_compat() -> None:
    """Backfill the private ASE API used by dscribe<=2.1.1 on ASE>=3.26."""
    if hasattr(Atoms, "_get_constraints"):
        return

    def _get_constraints(self):
        return self.constraints

    setattr(Atoms, "_get_constraints", _get_constraints)


class GyrationRFmaxCriterion(BasePartitionCriteria):
    type: Literal["GyrationRFmaxCriterion"] = "GyrationRFmaxCriterion"
    prop_name: Tuple[str, str] = ("gyration_radius", "fmax")
    forces_key: str = "forces"

    def __call__(self, db: BaseChemDataSet):
        property_values = []
        for data_row in db:
            grad = float(data_row.get_gyration_radius())
            fmax = (data_row.properties[self.forces_key] ** 2).sum(1).max() ** 0.5
            property_values.append([grad, fmax])
        return np.array(property_values)


class MBTRCriterion(BasePartitionCriteria):
    type: Literal["MBTRCriterion"] = "MBTRCriterion"
    geometry_function: Literal["atomic_number", "distance", "inverse_distance", "angle", "cosine"] = "distance"
    species: Tuple[str, ...] = ("H", "C", "O", "N")
    grid: Dict[Literal["min", "max", "sigma", "n"], Union[float, int]] = {"min": 0.9, "max": 5.0, "sigma": 0.1, "n": 30}
    normalization: Literal["l2", "valle_oganov", "n_atoms", "none"] = "l2"
    normalize_gaussians: bool = True
    weighting: Optional[Dict[Literal["function", "r_cut", "threshold"], Union[str, float, int]]] = None
    # {"function": "exp", "r_cut": 10, "threshold": 1e-3}
    periodic: bool = False
    sparse: bool = False
    dtype: Literal["float32", "float64"] = "float64"
    prop_name: Tuple[str, str] = ("MBTR[0]", "MBTR[1]")
    warning_atoms_calc_off: bool = True

    _mbtr: Any = PrivateAttr()
    _init_vals: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def geometry(self):
        return {"function": self.geometry_function}

    @classmethod
    def build_weighting_function(
        cls,
        function: Literal["unity", "exp", "inverse_square", "smooth_cutoff"] = "exp",
        r_cut: Optional[float] = 10,
        threshold: Optional[float] = 1e-3,
        **kwargs,
    ):
        ret = {}
        ret["function"] = function
        if r_cut is not None:
            ret["r_cut"] = r_cut
        if threshold is not None:
            ret["threshold"] = threshold
        if kwargs:
            ret.update(kwargs)
        return ret

    @property
    def sorted_species(self):
        return list(sorted(self.species))

    @property
    def number_of_features(self):
        return self._mbtr.get_number_of_features()

    def model_post_init(self, __context):
        error = self.is_installed()
        if error != "":
            raise ModuleNotFoundError(error)
        else:
            from dscribe.descriptors import MBTR

        if self.warning_atoms_calc_off:
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*Please use atoms.calc.*",
            )
        self._mbtr = MBTR(
            geometry=self.geometry,
            grid=self.grid,
            normalize_gaussians=self.normalize_gaussians,
            species=self.sorted_species,
            normalization=self.normalization,
            periodic=self.periodic,
            weighting=self.weighting,
            sparse=self.sparse,
            dtype=self.dtype,
        )
        self._init_vals = dict(
            geometry=self.geometry,
            grid=self.grid,
            model_type="MBTR",
            model_size=self.number_of_features,
            species=self.sorted_species,
            normalization=self.normalization,
            periodic=self.periodic,
            weighting=self.weighting,
            sparse=self.sparse,
            normalize_gaussians=self.normalize_gaussians,
            dtype=self.dtype,
        )

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        _ensure_ase_get_constraints_compat()
        all_atoms = [row.atoms for row in db]
        output = self._mbtr.create(all_atoms)
        return output

    def is_installed(self):
        try:
            from dscribe.descriptors import MBTR  # noqa F401

            return ""
        except ModuleNotFoundError:
            return "You have to install discribe: amspython -m pip install dscribe==2.1.1"

    @classmethod
    def auto_determine_mbtr_grid(cls, db: BaseChemDataSet, delta_x: float = 0.1, **kwargs):
        min_max = MinMaxAtomsDistance(mic=True)
        atoms = None
        species = set()
        d_mins = [np.inf]
        d_maxs = [0.0]
        for data_row in db:
            if len(data_row.atoms) > 1:
                row_trn = min_max(data_row)
                d_mins.append(row_trn.properties["min_distance"])
                d_maxs.append(row_trn.properties["max_distance"])
            species = species.union(set(data_row.atoms.get_chemical_symbols()))
        d_max = np.max(d_maxs)
        d_min = np.min(d_mins)
        periodic = False
        if atoms is not None:
            periodic = any(atoms.get_pbc())
        if periodic and "weighting" not in kwargs:
            kwargs["weighting"] = cls.build_weighting_function()
        n = int((d_max - d_min) / delta_x)
        species = tuple(sorted(species))
        return cls(
            grid={"min": d_min, "max": d_max, "sigma": delta_x / 10, "n": n},
            species=species,
            periodic=periodic,
            **kwargs,
        )


class GetMACERepresentation(BasePartitionCriteria):
    type: Literal["GetMACERepresentation"] = "GetMACERepresentation"
    prop_name: Tuple[str, str] = ("MACE[0]", "MACE[1]")

    model_type: Literal["off", "mp"] = "off"
    model_size: Literal["large", "medium", "small"] = "medium"
    device: Literal["cuda", "cpu"] = "cuda"
    default_dtype: Literal["float64", "float32"] = "float32"
    extra_kwargs: dict = {}
    warning_pickle_off: bool = False

    _calc: Any = PrivateAttr()
    _init_vals: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        error = self.is_installed()
        if error != "":
            raise ModuleNotFoundError(error)
        else:
            from mace.calculators import mace_mp, mace_off

        if self.warning_pickle_off:
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=r".*torch\.load.*weights_only=False.*",
            )
        if self.model_type == "off":
            self._calc = mace_off(
                model=self.model_size, device=self.device, default_dtype=self.default_dtype, **self.extra_kwargs
            )
        else:
            self._calc = mace_mp(
                model=self.model_size, device=self.device, default_dtype=self.default_dtype, **self.extra_kwargs
            )

        self._init_vals = dict(
            model_type=str(self.model_type),
            model_size=self.model_size,
            device=self.device,
            default_dtype=self.default_dtype,
            **self.extra_kwargs,
        )

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        try:
            descriptors = np.array([self._calc.get_descriptors(row.atoms).ravel() for row in db])
        except ValueError as e:
            for frame, _ in traceback.walk_tb(e.__traceback__):
                if frame.f_code.co_name == "z_to_index":
                    raise ValueError(f"Found an atomic species that are not supported by the model: {str(e)}") from e
            raise

        return descriptors

    def is_installed(self):
        try:
            from mace.calculators import mace_mp, mace_off  # noqa F401

            return ""
        except ModuleNotFoundError:
            return "You have to install mace: pip install .[mace]"
