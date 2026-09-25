from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Tuple, Union

from pydantic import PrivateAttr

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.checker_getters.getters.core import Getter
from scm.active_learning.logging import log_level

if TYPE_CHECKING:
    from scm.active_learning.loop.iteration_state import IterationState
    from scm.active_learning.loop.loop import ActiveLearningLoop
    from scm.active_learning.results.collection_check_get import CollectionCheckGetResults
    from scm.active_learning.results.engine_check import EngineCheckResult


class EarlyStopAccuracy(ALCallback):
    type: Literal["EarlyStopAccuracy"] = "EarlyStopAccuracy"
    patience: int = 3
    check_results_query: Optional[str] = None
    _no_improve_steps: int = PrivateAttr(default=0)
    _info_history: List[str] = PrivateAttr(default_factory=list)

    @staticmethod
    def _check_metadata_key(check: "EngineCheckResult") -> Tuple:
        return (
            check.task_id,
            check.system_id,
            check.checker_id,
            check.property,
            check.metric,
            check.units,
            check.per_n_atoms,
            check.atom_type,
            check.components,
            check.lower_is_better,
        )

    @staticmethod
    def _is_better(check: "EngineCheckResult", prev_value: Union[float, int, str, None]) -> bool:
        if prev_value is None or isinstance(prev_value, str) or isinstance(check.value, str):
            return False
        if check.lower_is_better:
            return check.value < prev_value
        return check.value > prev_value

    @staticmethod
    def _failing_checks(
        checks: Iterable["EngineCheckResult"],
    ) -> List["EngineCheckResult"]:
        return [check for check in checks if not check.is_success()]

    def _collect_failed_details(
        self,
        failed_checks: Iterable["EngineCheckResult"],
        prev_by_metadata: Dict[Tuple, Union[float, int]],
        max_error_details: int,
    ) -> List[str]:
        failed_details: List[str] = []

        for check in failed_checks:
            if len(failed_details) >= max_error_details:
                break

            value = f"{check.value:.3f}" if isinstance(check.value, float) else check.value
            detail = f"{check.property}-{check.metric}: {value}"
            prev_value = prev_by_metadata.get(self._check_metadata_key(check))
            if prev_value is not None and not isinstance(prev_value, str):
                comparator = "<" if self._is_better(check, prev_value) else ">="
                detail += f" ({comparator} prev {prev_value:g})"
            failed_details.append(detail)

        return failed_details

    def after_convergence(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        max_error_details = 3

        if state.task_results is None:
            return
        if al_loop.result is None:
            return
        n_previous_iterations = len(al_loop.result.iterations)
        if n_previous_iterations == 0:
            return

        previous_iter_idx = min(n_previous_iterations, max(1, self._no_improve_steps + 1))
        last_iter = al_loop.result.iterations[-previous_iter_idx]
        if last_iter.accuracy_checks_results is None or state.accuracy_checks_results is None:
            return

        selected_last_checks = list(
            Getter.query_checker_results(self.check_results_query, last_iter.accuracy_checks_results)
        )
        selected_current_checks = list(
            Getter.query_checker_results(self.check_results_query, state.accuracy_checks_results)
        )

        current_failed_checks = self._failing_checks(selected_current_checks)
        if not current_failed_checks:
            self.it_is_going_good("all selected accuracy checks decreasing")
            return

        previous_failed_checks = self._failing_checks(selected_last_checks)
        previous_failed_keys = {self._check_metadata_key(check) for check in previous_failed_checks}
        current_failed_keys = {self._check_metadata_key(check) for check in current_failed_checks}
        prev_by_metadata = {
            self._check_metadata_key(check): check.value
            for check in selected_last_checks
            if not isinstance(check.value, str)
        }

        failed_checks_reduced = current_failed_keys < previous_failed_keys
        failed_values_improved = any(
            self._is_better(check, prev_by_metadata.get(self._check_metadata_key(check)))
            for check in current_failed_checks
        )

        if failed_checks_reduced or failed_values_improved:
            self.it_is_going_good("selected accuracy failures improved")
            return

        history_details = self._collect_failed_details(
            current_failed_checks,
            prev_by_metadata,
            max_error_details,
        )
        remaining_errors = len(current_failed_checks) - len(history_details)
        if remaining_errors > 0:
            history_details.append(f"... +{remaining_errors} more")

        self._no_improve_steps += 1

        log_level(
            (
                "EarlyStopAccuracy is monitoring... It is going BAD. N Fails = {nfails} (<={patience})"
                "\nNo reduction in selected accuracy failures\n{table}"
            ),
            level="DEBUG",
            table="\n".join(history_details),
            nfails=self._no_improve_steps,
            patience=self.patience,
        )
        self._info_history.append(
            "Errors| " + (" | ".join(history_details) if history_details else "no-change details")
        )
        if self._no_improve_steps >= self.patience:
            from scm.active_learning.loop.iteration_phase import IterPhase

            fmt_history = "\n".join(self._info_history)
            state.al_finished_reason = (
                f"EarlyStopAccuracy: "
                f"accuracy errors not reduced for {self.patience} iterations. "
                f"History:\n{fmt_history}"
            )
            state.skip[IterPhase.SPLIT_AND_ADD] = "EarlyStopAccuracy"
            state.skip[IterPhase.TRAIN] = "EarlyStopAccuracy"

    def it_is_going_good(self, reason: str) -> None:
        log_level(
            "EarlyStopAccuracy is monitoring... It is going GOOD! {reason}",
            level="DEBUG",
            reason=reason,
        )
        self._no_improve_steps = 0
        self._info_history.clear()


class EarlyStopValidity(ALCallback):
    type: Literal["EarlyStopValidity"] = "EarlyStopValidity"
    check_results_query: Optional[str] = None

    @staticmethod
    def _failing_checks(
        checks: Iterable["EngineCheckResult"],
    ) -> List["EngineCheckResult"]:
        return [check for check in checks if not check.is_success()]

    @staticmethod
    def _validity_checks(task_results: "CollectionCheckGetResults") -> Iterable["EngineCheckResult"]:
        for check_get_result in task_results.validity_get_results:
            yield from check_get_result.validity_results

    @staticmethod
    def _format_failed_details(
        failed_checks: Iterable["EngineCheckResult"],
        max_error_details: int,
    ) -> List[str]:
        failed_details: List[str] = []

        for check in failed_checks:
            if len(failed_details) >= max_error_details:
                break
            value = f"{check.value:.3f}" if isinstance(check.value, float) else check.value
            failed_details.append(f"{check.property}-{check.metric}: {value}")

        return failed_details

    def after_run_task(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        del al_loop
        max_error_details = 3

        if state.task_results is None:
            return
        if state.al_finished_reason is not None:
            return

        selected_checks = list(
            Getter.query_checker_results(
                self.check_results_query,
                list(self._validity_checks(state.task_results)),
            )
        )
        failed_checks = self._failing_checks(selected_checks)
        if not failed_checks:
            log_level(
                "EarlyStopValidity is monitoring... It is going GOOD! {reason}",
                level="DEBUG",
                reason="all selected validity checks passed",
            )
            return

        error_details = self._format_failed_details(failed_checks, max_error_details)
        remaining_errors = len(failed_checks) - len(error_details)
        if remaining_errors > 0:
            error_details.append(f"... +{remaining_errors} more")

        from scm.active_learning.loop.iteration_phase import IterPhase

        fmt_errors = "\n".join(error_details) if error_details else "selected validity checks failed"
        state.al_finished_reason = f"EarlyStopValidity: selected validity checks failed.\n{fmt_errors}"
        state.skip[IterPhase.LABEL] = "EarlyStopValidity"
        state.skip[IterPhase.ACCURACY] = "EarlyStopValidity"
        state.skip[IterPhase.CONVERGENCE] = "EarlyStopValidity"
        state.skip[IterPhase.SPLIT_AND_ADD] = "EarlyStopValidity"
        state.skip[IterPhase.TRAIN] = "EarlyStopValidity"

        log_level(
            "EarlyStopValidity is monitoring... It is going BAD.\n{table}",
            level="DEBUG",
            table=fmt_errors,
        )
