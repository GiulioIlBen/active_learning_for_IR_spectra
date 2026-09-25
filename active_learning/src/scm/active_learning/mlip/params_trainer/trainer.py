from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import Field, JsonValue
from scm.moliterate.interfaces import ParAMSData
from scm.plams import Settings

from scm.active_learning.engines import ConcreteEngines, ParAMSEngine
from scm.active_learning.mlip.core import MLIPTrainer, MLIPTrainerSaveSettings
from scm.active_learning.mlip.params_trainer._optional_dependencies import (
    _get_m3gnet_backend_error as _get_m3gnet_backend_error_impl,
)
from scm.active_learning.mlip.params_trainer._optional_dependencies import (
    _load_params_deps as _load_params_deps_impl,
)
from scm.active_learning.mlip.params_trainer.settings_builder import (
    MLIPParAMSSettingsBuilder,
)
from scm.active_learning.mlip.splitting_strategy import DataAndSplittingStrategy

if TYPE_CHECKING:
    from scm.params import ParAMSResults
    from scm.params.common.dataset_evaluator import GroupedResult

    from scm.active_learning.results import MLIPTrainerResults
else:
    ParAMSResults = Any
    GroupedResult = Any


class ParAMSTrainerSaveSettings(MLIPTrainerSaveSettings):
    settings_file_name: str = "params_trainer_settings.json"


class ParAMSTrainer(MLIPTrainer[ParAMSEngine]):
    type: Literal["ParAMSTrainer"] = "ParAMSTrainer"
    datapath: str = ""  # params_data
    source_settings: Dict[str, JsonValue] = {}
    save: ParAMSTrainerSaveSettings = Field(default_factory=ParAMSTrainerSaveSettings)

    @property
    def ml_name(self):
        path = ("input", "MachineLearning", "Backend")
        ret = str(self.settings.get_nested(path, default="MLIP"))
        if ret.lower() == "test":
            return "MLIPTest"
        return ret

    def validate_engine(
        self,
        engine: ConcreteEngines,
    ) -> ParAMSEngine:
        if isinstance(engine, ParAMSEngine):
            return engine
        raise TypeError(f"{type(engine)=} is not ParAMSEngine")

    @property
    def settings(self):
        return Settings(self.source_settings)

    @staticmethod
    def _load_params_deps():
        return _load_params_deps_impl()

    @classmethod
    def _get_m3gnet_backend_error(cls, model: str) -> Optional[str]:
        del cls
        return _get_m3gnet_backend_error_impl(model)

    def _ensure_backend_ready(self) -> None:
        backend = str(self.settings.get_nested(("input", "MachineLearning", "Backend"), default=""))
        if backend != "M3GNet":
            return
        model = str(self.settings.get_nested(("input", "MachineLearning", "M3GNet", "Model"), default=""))
        error = self._get_m3gnet_backend_error(model)
        if error is not None:
            raise FileNotFoundError(error)

    def train(self, data_split: DataAndSplittingStrategy, engine_id: str, **kwargs) -> "MLIPTrainerResults":
        self._ensure_backend_ready()
        self.to_data_format(data_split=data_split)
        params_job, _, _ = self._load_params_deps()
        j = params_job.from_yaml(self.datapath, settings=self.settings, name="ParAMSTrainer")
        res = j.run()
        results = self._build_results(res, engine_id=engine_id)
        self._persist_json_artifacts(job_path=j.path, results=results)
        return results

    def finetune(
        self,
        engine: ParAMSEngine,
        data_split: DataAndSplittingStrategy,
        engine_id: str,
        **kwargs,
    ) -> MLIPTrainerResults:
        self._ensure_backend_ready()
        self.to_data_format(data_split=data_split)
        ParAMSJob, _, _ = self._load_params_deps()
        j = ParAMSJob.from_yaml(self.datapath, settings=self.settings, name="ParAMSTrainer")
        # to load the model you need the results folder
        j.settings.input.MachineLearning.LoadModel = engine.job_path + "/results"
        res = j.run()
        results = self._build_results(res, engine_id=engine_id)
        self._persist_json_artifacts(job_path=j.path, results=results)
        return results

    def to_data_format(
        self,
        data_split: DataAndSplittingStrategy,
    ):
        if len(data_split.dataset) == 0:
            raise ValueError("Something went wrong here...")
        ri = ParAMSData._importer_from_entries(
            data_split.dataset,
            data_split.dataset.out_properties,
            dataset_metadata_key=data_split.split_key,
        )
        if len(ri.data_set) == 0:
            raise ValueError("Something went wrong here...")
        ri.store(self.datapath)
        return Path(self.datapath)

    ##############################################################################
    #####################     Private Methods     ################################
    ##############################################################################

    def _last_epoch(self, params_results: ParAMSResults) -> List[str]:
        epochs = params_results.regex_file("$JN.out", r"Epoch:\s*(\d+)")

        def try_to_int(v):
            try:
                return int(v)
            except Exception:
                return -1

        if len(epochs) == 0:
            return -1
        return max(map(try_to_int, epochs))

    def _build_results(self, params_results: ParAMSResults, engine_id: str):
        from scm.active_learning.results import MLIPTrainerResults

        return MLIPTrainerResults(
            engine=ParAMSEngine.from_params_results(
                params_results=params_results,
                engine_id=engine_id,
            ),
            log_metrics=self._get_table_from_params_results(params_results),
            train_infos={"n_epochs": self._last_epoch(params_results)},
        )

    def _persist_json_artifacts(self, *, job_path: Optional[str], results: "MLIPTrainerResults") -> None:
        if not self.save.settings and not self.save.results:
            return
        if job_path is None:
            raise ValueError("ParAMSTrainer cannot save JSON artifacts because job.path is not available after run().")

        run_dir = Path(job_path).expanduser().resolve(strict=False)
        settings_path = self.save.persist_trainer(trainer=self, run_dir=run_dir)
        self.save.persist_results(results=results, run_dir=run_dir, settings_path=settings_path)

    def _get_table_from_params_results(self, params_results: ParAMSResults):
        df = defaultdict(list)
        for dataset_i in ["training_set", "validation_set"]:
            try:
                dse = params_results.get_data_set_evaluator(data_set=dataset_i)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"maybe RunAMSAtEnd = False? or some bug in the ML models: {params_results.job.path}",
                ) from e
            self._parse_dse_results_to_error_table(df, dse.results, dataset_i)
        return dict(df)

    def _parse_dse_results_to_error_table(
        self,
        df: Dict[str, List[Union[int, float, str]]],
        dse_results: GroupedResult,
        dataset: str,
    ):
        for k in dse_results:
            df["property"].append(k)
            df["units"].append(dse_results[k].unit)
            df["n_entries"].append(len(dse_results[k].residuals))
            df["mae"].append(dse_results[k].mae)
            df["rmse"].append(dse_results[k].rmse)
            df["r2"].append(dse_results[k].r2)
            df["dataset"].append(dataset)

    ######################################################################################################
    #################################         Builders          ##########################################
    ######################################################################################################

    class Builder(MLIPParAMSSettingsBuilder): ...
