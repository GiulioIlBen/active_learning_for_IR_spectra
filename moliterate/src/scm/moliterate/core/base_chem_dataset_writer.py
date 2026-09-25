from abc import abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

from scm.moliterate.core.base_chem_dataset import BaseChemDataSet
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.utils.batch_collector import batch_collector
from scm.moliterate.utils.progress_bar import moliterate_tqdm


class BaseChemDataSetWriter(BaseChemDataSet):
    data_source: str

    @classmethod
    @abstractmethod
    def create(
        cls,
        data_source: str = "",
        available_properties: Optional[List[PropertyInfo]] = None,
        distance_unit: str = "Ang",
        **kwargs,
    ) -> "BaseChemDataSetWriter":
        pass

    def add_systems(self, mol_data_rows: Iterable[ChemDataEntry], **kwargs):
        """add many systems, override for performance improvement"""
        for m in moliterate_tqdm(mol_data_rows, **kwargs):
            self.add_system(m)

    @abstractmethod
    def add_system(self, mol_data_row: ChemDataEntry):
        """add one system. NOTE: available_properties requires that the properties keys are present in MolDataRow"""

    @abstractmethod
    def update_metadata(self, **kwargs):
        """update the metadata of the dataset"""

    @abstractmethod
    def update_row_metadata(self, idx: int, absolute: bool = False, **update):
        pass

    def unlink(self, missing_ok: bool = True):
        if Path(self.data_source).is_file():
            Path(self.data_source).unlink(missing_ok=missing_ok)

    def batch_collector(self, batch_size: int = 1000):
        return batch_collector(flush=self.add_systems, batch_size=batch_size)
