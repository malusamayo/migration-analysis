"""Workspace setup and MCP config for the tau2 airline task.

This task adapts tau2-bench's airline domain into this repo's single-agent
execution loop by exposing both the airline environment and the user simulator
through a stateful MCP server.
"""

import json
import os
import shutil
import sys
import types
from pathlib import Path

DEFAULT_TAU2_BENCH_PATH = Path("/mnt/data_4tb/datasets/tau2-bench")
DEFAULT_USER_SIM_MODEL = "vertex_ai/gemini-3-flash-preview"
_env_tau2 = os.environ.get("TAU2_BENCH_PATH")
TAU2_BENCH_PATH = Path(_env_tau2) if _env_tau2 else DEFAULT_TAU2_BENCH_PATH


def _ensure_tau2_path() -> Path:
    if not TAU2_BENCH_PATH.exists():
        raise FileNotFoundError(
            f"tau2-bench not found at {TAU2_BENCH_PATH}. "
            "Set TAU2_BENCH_PATH to the benchmark checkout."
        )
    tau2_src = TAU2_BENCH_PATH / "src"
    tau2_str = str(tau2_src)
    if tau2_str not in sys.path:
        sys.path.insert(0, tau2_str)
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
    return TAU2_BENCH_PATH


def _get_user_sim_model() -> str:
    user_model = os.environ.get("TAU2_USER_SIM_MODEL")
    if user_model:
        return user_model
    return DEFAULT_USER_SIM_MODEL


def _get_google_application_credentials() -> str | None:
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials:
        return credentials
    vertex_credentials = os.environ.get("VERTEX_CREDENTIALS")
    if vertex_credentials:
        return vertex_credentials
    default_credentials = Path.cwd() / ".vertex-ai.json"
    if default_credentials.exists():
        return str(default_credentials.resolve())
    return None


def _state_dir(workspace_dir: str) -> Path:
    return Path(workspace_dir) / ".tau2_airline"


def _placeholder_task() -> dict[str, object]:
    return {
        "id": "proposer",
        "description": {
            "purpose": "Placeholder tau2 airline task for proposer workspace startup.",
            "relevant_policies": None,
            "notes": None,
        },
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": "Placeholder proposer task.",
                "domain": "airline",
                "reason_for_call": "Placeholder proposer task.",
                "known_info": "",
                "unknown_info": None,
            },
        },
        "initial_state": None,
        "evaluation_criteria": {
            "actions": [],
            "communicate_info": [],
            "nl_assertions": [],
            "reward_basis": ["DB", "COMMUNICATE"],
        },
        "annotations": None,
    }


def _initialize_state_files(workspace_dir: str, task: dict[str, object]) -> None:
    from tau2.domains.airline.data_model import FlightDB
    from tau2.domains.airline.utils import AIRLINE_DB_PATH

    state_dir = _state_dir(workspace_dir)
    local_state_dir = state_dir / "state"
    local_state_dir.mkdir(parents=True, exist_ok=True)

    FlightDB.load(AIRLINE_DB_PATH).dump(local_state_dir / "airline_db.json")

    with open(state_dir / "task.json", "w") as f:
        json.dump(task, f, indent=2)

    with open(state_dir / "trajectory.json", "w") as f:
        json.dump([], f, indent=2)

    with open(state_dir / "conversation_status.json", "w") as f:
        json.dump(
            {
                "started": False,
                "done": False,
                "stop_reason": None,
                "start_time": None,
                "end_time": None,
            },
            f,
            indent=2,
        )

    user_state_path = state_dir / "user_state.json"
    if user_state_path.exists():
        user_state_path.unlink()

    shutil.copy2(
        Path(__file__).with_name("tau2_airline_mcp_server.py"),
        state_dir / "tau2_airline_mcp_server.py",
    )

    shutil.copy2(
        TAU2_BENCH_PATH / "data" / "tau2" / "domains" / "airline" / "policy.md",
        state_dir / "policy.md",
    )


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    del log_dir
    _ensure_tau2_path()

    workspace = Path(workspace_dir)
    _initialize_state_files(workspace_dir, example["task"])


def setup_proposer_workspace(workspace_dir: str) -> None:
    _ensure_tau2_path()
    _initialize_state_files(workspace_dir, _placeholder_task())


def _build_server_config(server_script: Path, env: dict[str, str], docker_workspace: bool) -> dict:
    project_root = server_script.parent.parent.parent
    if docker_workspace:
        return {
            "command": "/workspace/.venv/bin/python",
            "args": [str(server_script)],
            "env": env,
        }
    return {
        "command": "uv",
        "args": ["--directory", str(project_root), "run", "python", str(server_script)],
        "env": env,
    }


def get_mcp_config(workspace_dir: str) -> dict:
    _ensure_tau2_path()
    user_model = _get_user_sim_model()

    workspace = Path(workspace_dir).resolve()
    docker_workspace = str(workspace).startswith("/workspace/")
    tau2_root = Path("/tau2-bench") if docker_workspace else TAU2_BENCH_PATH
    server_script = workspace / ".tau2_airline" / "tau2_airline_mcp_server.py"

    env = {
        "TAU2_AIRLINE_WORKSPACE": str(workspace),
        "TAU2_BENCH_PATH": str(tau2_root),
        "TAU2_USER_SIM_MODEL": user_model,
        "VERTEXAI_LOCATION": os.environ.get("VERTEXAI_LOCATION", "global"),
    }
    forwarded_env_vars = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_SERVICE_ACCOUNT_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "VERTEXAI_LOCATION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
    )
    for env_var in forwarded_env_vars:
        value = os.environ.get(env_var)
        if value:
            env[env_var] = value
    google_application_credentials = _get_google_application_credentials()
    if google_application_credentials:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = google_application_credentials
    user_llm_args = os.environ.get("TAU2_USER_SIM_LLM_ARGS")
    if user_llm_args:
        env["TAU2_USER_SIM_LLM_ARGS"] = user_llm_args

    return {
        "mcpServers": {
            "tau2_airline": _build_server_config(
                server_script,
                env,
                docker_workspace,
            )
        }
    }
