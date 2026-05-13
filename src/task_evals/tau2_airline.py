"""Evaluation for the tau2 airline task."""

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import dspy

DEFAULT_TAU2_BENCH_PATH = Path("/mnt/data_4tb/datasets/tau2-bench")
_env_tau2 = os.environ.get("TAU2_BENCH_PATH")
TAU2_BENCH_PATH = Path(_env_tau2) if _env_tau2 else DEFAULT_TAU2_BENCH_PATH


def _ensure_tau2_path() -> None:
    if not TAU2_BENCH_PATH.exists():
        raise FileNotFoundError(
            f"tau2-bench not found at {TAU2_BENCH_PATH}. "
            "Set TAU2_BENCH_PATH to the benchmark checkout."
        )
    tau2_str = str(TAU2_BENCH_PATH / "src")
    if tau2_str not in sys.path:
        sys.path.insert(0, tau2_str)
    tau2_src = TAU2_BENCH_PATH / "src"
    package_paths = {
        "tau2": tau2_src / "tau2",
        "tau2.agent": tau2_src / "tau2" / "agent",
        "tau2.agent.base": tau2_src / "tau2" / "agent" / "base",
    }
    for package_name, package_path in package_paths.items():
        if package_name in sys.modules:
            continue
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package


def _deserialize_message(entry: dict[str, Any]) -> Any:
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

    kind = entry["kind"]
    data = entry["data"]
    if kind == "assistant":
        return AssistantMessage.model_validate(data)
    if kind == "user":
        return UserMessage.model_validate(data)
    if kind == "tool":
        message = ToolMessage.model_validate(data)
        if message.error:
            message.content = _normalize_replay_error_content(message.content)
        return message
    raise ValueError(f"Unknown trajectory message kind: {kind}")


def _normalize_replay_error_content(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(payload, dict) and set(payload) == {"error"}:
        return f"Error: {payload['error']}"
    return content


def _build_feedback(reward_info: Any) -> str:
    failures = []
    if reward_info.db_check is not None and not reward_info.db_check.db_match:
        failures.append("database state does not match the benchmark end state")
    if reward_info.communicate_checks:
        missing = [check.info for check in reward_info.communicate_checks if not check.met]
        if missing:
            failures.append(
                "missing communicated info: " + ", ".join(sorted(missing))
            )
    if not failures:
        return "Correct: trajectory matches tau2 airline DB and communication checks."
    return "Validation failed: " + " | ".join(failures)


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    del lm
    _ensure_tau2_path()

    from tau2.data_model.tasks import Task
    from tau2.domains.airline.environment import get_environment as airline_get_environment
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    workspace = Path(workspace_dir)
    state_dir = workspace / ".tau2_airline"
    trajectory_path = state_dir / "trajectory.json"
    status_path = state_dir / "conversation_status.json"

    if not trajectory_path.exists():
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "No tau2 airline trajectory found — task setup or MCP server failed.",
        }

    with open(trajectory_path) as f:
        trajectory_entries = json.load(f)
    messages = [_deserialize_message(entry) for entry in trajectory_entries]

    if not status_path.exists():
        status = {"done": False, "start_time": None, "end_time": None}
    else:
        with open(status_path) as f:
            status = json.load(f)

    task = Task.model_validate(example["task"])
    env_reward = EnvironmentEvaluator.calculate_reward(
        environment_constructor=airline_get_environment,
        task=task,
        full_trajectory=messages,
        solo_mode=False,
    )
    communicate_reward = CommunicateEvaluator.calculate_reward(
        task=task,
        full_trajectory=messages,
    )

    reward_components: list[float] = []
    reward_breakdown: dict[str, float] = {}
    if "DB" in task.evaluation_criteria.reward_basis:
        reward_components.append(env_reward.reward_breakdown["DB"])
        reward_breakdown.update(env_reward.reward_breakdown)
    if "COMMUNICATE" in task.evaluation_criteria.reward_basis:
        reward_components.append(communicate_reward.reward_breakdown["COMMUNICATE"])
        reward_breakdown.update(communicate_reward.reward_breakdown)
    reward = sum(reward_components) / len(reward_components)

    reward_info = types.SimpleNamespace(
        reward=reward,
        db_check=env_reward.db_check,
        communicate_checks=communicate_reward.communicate_checks,
        reward_breakdown=reward_breakdown,
    )

    return {
        "workspace_dir": workspace_dir,
        "score": reward_info.reward,
        "feedback": _build_feedback(reward_info),
        "reward_breakdown": reward_info.reward_breakdown,
    }
