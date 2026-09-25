from types import SimpleNamespace

from ase import Atoms
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces import InMemoryMolData

from scm.active_learning.checker_getters.checkers import AMSTrajChecker
from scm.active_learning.checker_getters.getters import AMSTrajGetter
from scm.active_learning.results import EngineCheckResult


def _checker_result(*, property: str, metric: str, value, success: str) -> EngineCheckResult:
    return EngineCheckResult(
        engine_id="engine",
        task_id="go_nms",
        system_id="system",
        checker_id="AMSTrajChecker",
        property=property,
        metric=metric,
        value=value,
        target="",
        units="",
        n_entries=1,
        value_type="str" if isinstance(value, str) else "float",
        success=success,
    )


def _dataset():
    dataset = InMemoryMolData.create(available_properties=[])
    for index in range(5):
        dataset.add_system(ChemDataEntry(system=Atoms("H", positions=[[float(index), 0.0, 0.0]])))
    return dataset


def test_remove_unreasonable_traj_keeps_the_inclusive_reasonable_final_frame():
    selected = AMSTrajGetter.RemoveUnreasonableTraj().run(
        _dataset(),
        [_checker_result(property="trajectory", metric="reasonable_final_frame", value=2.0, success="ENERGY_CHANGE")],
    )

    assert list(selected.absolute_idxs) == [0, 1, 2]


def test_remove_unreasonable_traj_keeps_next_energy_change_frame_only_for_converged_go():
    selected = AMSTrajGetter.RemoveUnreasonableTraj(energy_change_frame="next_if_go_converged").run(
        _dataset(),
        [
            _checker_result(property="trajectory", metric="reasonable_final_frame", value=2.0, success="ENERGY_CHANGE"),
            _checker_result(
                property="ConvergedGO", metric="TerminationStatus", value="NORMAL TERMINATION", success="OK"
            ),
        ],
    )

    assert list(selected.absolute_idxs) == [0, 1, 2, 3]


def test_remove_unreasonable_traj_does_not_keep_next_energy_change_frame_for_nonconverged_go():
    selected = AMSTrajGetter.RemoveUnreasonableTraj(energy_change_frame="next_if_go_converged").run(
        _dataset(),
        [
            _checker_result(property="trajectory", metric="reasonable_final_frame", value=2.0, success="ENERGY_CHANGE"),
            _checker_result(
                property="ConvergedGO", metric="TerminationStatus", value="MAXIMUM ITERATIONS", success="GONoConv"
            ),
        ],
    )

    assert list(selected.absolute_idxs) == [0, 1, 2]


class _GeometryOptimizationJob:
    settings = SimpleNamespace(get_nested=lambda *args, **kwargs: None)

    def get_task(self):
        return "GeometryOptimization"

    def ok(self):
        return True

    def check(self):
        return True

    def get_errormsg(self):
        return None


class _GeometryOptimizationResults:
    job = _GeometryOptimizationJob()

    def readrkf(self, section, variable):
        if (section, variable) == ("History", "nEntries"):
            return 1
        if (section, variable) == ("General", "termination status"):
            return "NORMAL TERMINATION"
        raise KeyError(variable)

    def get_history_property(self, key, history_section=None):
        return [0.0] if key == "Energy" else None

    def get_history_molecule(self, step):
        return None

    def get_exit_condition_message(self):
        return ""

    def get_main_molecule(self):
        return Atoms("H")


def test_ams_traj_checker_emits_go_convergence_result_when_enabled():
    result = SimpleNamespace(
        plams_results=_GeometryOptimizationResults(),
        from_task=SimpleNamespace(task_id="go_nms", system_id="system"),
        from_engine=SimpleNamespace(engine_id="engine"),
    )

    check_results = AMSTrajChecker(check_geometry_optimization_convergence=True).run(result)

    assert [(check.property, check.metric, check.success) for check in check_results] == [
        ("trajectory", "reasonable_final_frame", "OK"),
        ("ConvergedGO", "TerminationStatus", "OK"),
    ]
