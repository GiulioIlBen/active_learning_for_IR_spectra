from typing import Dict, List, Literal, Union

import numpy as np
from pydantic import Field
from typing_extensions import Annotated

from ..core import BaseChemDataSet, BasePartitionCriteria


class TreePartitionCriterion(BasePartitionCriteria):
    type: Literal["TreePartitionCriterion"] = "TreePartitionCriterion"
    # union first so dicts (e.g. YAML) deserialize; any BasePartitionCriteria instance is still accepted from Python
    partition_criteria: List[
        Annotated[Union["UnionPartitionCriterion", BasePartitionCriteria], Field(union_mode="left_to_right")]
    ]
    concat_symbol: Union[str, Literal["KeepLast"]] = "%"

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        dict_concat = self.make_clusters(db)
        return self.to_array(db, dict_concat)

    def make_clusters(self, db: BaseChemDataSet) -> Dict[str, np.ndarray]:
        cluster_dbs: Dict[str, BaseChemDataSet] = {"": db}
        for c in self.partition_criteria:
            cluster_dbs = self._cluster_i(c, cluster_dbs)
        ret = {}
        for k, v in cluster_dbs.items():
            ret[k] = db.from_absolute_to_relative(v.select_indices())
        return ret

    def _cluster_i(self, clustering_strategy: BasePartitionCriteria, cluster_dbs: Dict[str, BaseChemDataSet]):
        cluster_dbs_ret = {}
        for k_i, db_i in cluster_dbs.items():
            cluster_res = clustering_strategy.group_by(clustering_strategy(db_i))
            for k, v in cluster_res.items():
                if self.concat_symbol == "KeepLast":
                    cluster_dbs_ret[k] = db_i.subset(v)
                else:
                    cluster_dbs_ret[f"{k_i}{self.concat_symbol}{k}"] = db_i.subset(v)
        return cluster_dbs_ret
