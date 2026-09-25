from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

from scm.moliterate.analysis import PairwiseResult
from tabulate import tabulate

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.logging import log_level

if TYPE_CHECKING:
    from scm.active_learning.loop.iteration_state import IterationState
    from scm.active_learning.loop.loop import ActiveLearningLoop
    from scm.active_learning.results import EngineCheckResult


def _compact_check_result_row(result: object) -> dict:
    if hasattr(result, "to_compact_dict"):
        return result.to_compact_dict()
    row = result.model_dump(exclude={"lower_is_better"})  # type: ignore[attr-defined]
    if row.pop("per_n_atoms"):
        row["units"] += "/NAtoms"
    return row


def check_res_table(res: Optional[List[PairwiseResult] | List[EngineCheckResult]]):
    if res is None:
        return ""
    return tabulate([_compact_check_result_row(x) for x in res], headers="keys")


class ALStateLogger(ALCallback):
    type: Literal["ALStateLogger"] = "ALStateLogger"

    def after_run_task(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        if state.task_results is None:
            return

        log_level(
            "iAL:{iAL:02} | Validity  Results\n{check_results_table}",
            iAL=state.iteration_al,
            check_results_table=check_res_table(
                [x for r in state.task_results.validity_get_results for x in r.validity_results]
            ),
        )
        log_level(
            "iAL:{iAL:02} | Getter Results\n{check_results_table}",
            iAL=state.iteration_al,
            check_results_table=state.task_results.getter_results_table(),
        )

    def after_accuracy(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        if state.task_results is None:
            return

        log_level(
            "iAL:{iAL:02} | Accuracy  Results\n{accuracy}",
            iAL=state.iteration_al,
            accuracy=check_res_table(state.accuracy_checks_results),
        )

    def after_split_and_add(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        tr, val = al_loop.splitting.get_training_and_validation_datasets()
        log_level(
            "iAL:{iAL:02} | #train={ntrain_set} | #val={nval_set}",
            iAL=state.iteration_al,
            ntrain_set=len(tr),
            nval_set=len(val),
        )

    def after_train(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        if state.train_results is not None:
            log_level(
                "iAL:{iAL:02} | Train Results:\n{table}",
                table=tabulate(state.train_results.log_metrics, headers="keys"),
                iAL=state.iteration_al,
                level="DEBUG",
            )
        log_level(
            "iAL:{iAL:02} | Candidate Engine settings:\n{settings}",
            settings=(state.train_results.engine.settings if state.train_results else "No Engine Results"),
            iAL=state.iteration_al,
        )

    def after_train_check(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        if state.trainer_check_result is None:
            return
        log_level(
            "iAL:{iAL:02} | Trainer Check Results\n{accuracy}",
            iAL=state.iteration_al,
            accuracy=check_res_table(state.trainer_check_result.accuracy_checks_results),
        )

    def after_loop(self, al_loop: "ActiveLearningLoop", state=None) -> None:
        pass
