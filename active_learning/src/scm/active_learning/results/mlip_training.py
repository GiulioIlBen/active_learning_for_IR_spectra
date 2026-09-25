from __future__ import annotations

from typing import Dict, List, Union

from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.logging.path_serializable_model import PathSerializableModel


class MLIPTrainerResults(PathSerializableModel):
    engine: ConcreteEngines
    log_metrics: Dict[str, List[Union[int, float, str]]] = {}
    train_infos: Dict[str, Union[int, float, str]] = {}
    # result.get_logs_metrics_dataframe()
