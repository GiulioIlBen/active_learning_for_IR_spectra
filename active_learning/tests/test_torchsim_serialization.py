from __future__ import annotations

from pydantic import TypeAdapter
from scm.plams import Atom, Molecule

from scm.active_learning.engines import ConcreteEngines, PLAMSMolecule, TorchEngine
from scm.active_learning.logging.state_paths import collect_runtime_paths, rewrite_payload_paths
from scm.active_learning.results.task import ConcreteTaskResult, TorchSimMDResult
from scm.active_learning.tasks import ConcreteTask, TorchSimMDTask


def _make_container() -> PLAMSMolecule:
    molecule = Molecule()
    molecule.add_atom(Atom(symbol="Ar", coords=(0.0, 0.0, 0.0)))
    return PLAMSMolecule.from_molecules(system_id="Ar", systems=molecule)


def test_torchsim_types_round_trip_through_discriminated_unions(tmp_path):
    engine = TorchEngine(engine_id="torchsim", model_factory="tests.test_torchsim_tasks.build_task_model")
    task = TorchSimMDTask(atomistic_system=_make_container(), save_folder=str(tmp_path))
    result = TorchSimMDResult(
        task=task,
        engine=engine,
        save_folder=str(tmp_path),
        trajectory_path=str(tmp_path / "results_out" / "trajectory.h5"),
        requested_steps=10,
        executed_steps=10,
        integrator="nvt_langevin",
        final_energy=1.5,
    )

    engine_adapter = TypeAdapter(ConcreteEngines)
    task_adapter = TypeAdapter(ConcreteTask)
    result_adapter = TypeAdapter(ConcreteTaskResult)

    assert engine_adapter.validate_python(engine.model_dump(mode="json")).engine_id == "torchsim"
    assert task_adapter.validate_python(task.model_dump(mode="json")).task_id == "TorchSimMDTask"
    validated_result = result_adapter.validate_python(result.model_dump(mode="json"))
    assert isinstance(validated_result, TorchSimMDResult)
    assert validated_result.trajectory_path.endswith("trajectory.h5")


def test_json_paths_registers_trajectory_path(tmp_path):
    trajectory_path = tmp_path / "run" / "trajectory.h5"
    trajectory_path.parent.mkdir(parents=True)
    trajectory_path.write_text("trajectory", encoding="utf-8")
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    payload = {"result": {"save_folder": str(tmp_path / "run"), "trajectory_path": str(trajectory_path)}}

    rewritten = rewrite_payload_paths(payload, json_dir, relative=True)
    runtime_paths = set(collect_runtime_paths(payload))

    assert rewritten["result"]["trajectory_path"] == "../run/trajectory.h5"
    assert trajectory_path.resolve(strict=False) in runtime_paths
