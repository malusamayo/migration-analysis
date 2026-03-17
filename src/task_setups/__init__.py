import os
import shutil

from openhands.sdk import Tool
from openhands.tools.browser_use import BrowserToolSet
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool

from .webarena_servers import (
    preprocess_example as _webarena_preprocess_example,
    collect_required_sites,
    set_default_webarena_urls,
    start_webarena_servers,
    stop_webarena_servers,
)


def _webarena_browser_tool_params(workspace_dir: str | None) -> dict:
    """Build per-workspace browser-use paths to avoid profile collisions."""
    if not workspace_dir:
        return {}

    browser_root = os.path.join(workspace_dir, ".browser_use")
    return {
        "user_data_dir": os.path.join(browser_root, "profile"),
        "downloads_path": os.path.join(browser_root, "downloads"),
    }


def preprocess_example(task_id: str, example: dict) -> dict:
    """Apply task-specific preprocessing to an example before running the agent."""
    if task_id == "webarena":
        return _webarena_preprocess_example(example)
    return example


def get_tools(task_id: str, workspace_dir: str | None = None) -> list:
    """Return the list of Tools appropriate for a task."""
    if task_id == "webarena":
        return [
            Tool(
                name=BrowserToolSet.name,
                params=_webarena_browser_tool_params(workspace_dir),
            )
        ]
    return [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)]


def setup_workspace(task_id: str, workspace_dir: str, log_dir: str, example: dict) -> None:
    """Clean and recreate workspace/log dirs, then write any task-specific files."""
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if task_id == "webtest":
        with open(os.path.join(workspace_dir, "index.html"), "w") as f:
            f.write(example["html_content"])


def setup_servers(
    task_id: str,
    args_list: list = None,
    start_servers: bool = False,
    timeout: int = 300,
) -> dict:
    """Initialize task-specific server URLs and optionally start servers.

    Returns a dict to pass to teardown_servers.
    """
    if task_id == "webarena":
        required_sites = collect_required_sites(args_list) if args_list else None
        set_default_webarena_urls(required_sites)
        if start_servers:
            return start_webarena_servers(sites=required_sites, timeout=timeout)
    return {}


def teardown_servers(task_id: str, servers_started: dict) -> None:
    """Stop any servers started by setup_servers."""
    if task_id == "webarena" and servers_started:
        stop_webarena_servers(list(servers_started))


def get_eval_config(task_id: str) -> dict:
    """Return task-specific evaluation configuration."""
    if task_id == "webgen":
        from ..task_evals.webgen import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": True, "max_workers": 4}
    elif task_id == "webtest":
        from ..task_evals.webtest import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": True, "max_workers": 32}
    elif task_id == "webarena":
        from ..task_evals.webarena import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 32}
    elif task_id == "build-pov-ray":
        from ..task_evals.build_pov_ray import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 32}
    else:
        raise ValueError(f"Unknown task_id: {task_id!r}")


def get_seed_candidate(task_id: str) -> dict[str, str]:
    """Return the seed agent candidate code for a task."""
    if task_id == "webarena":
        code = '''\
from openhands.sdk import LLM, Agent, Tool
from openhands.tools.browser_use import BrowserToolSet
import os

def build_agent(base_dir, lm_model, seed_prompt):
    prompt_path = os.path.join(base_dir, "system_prompt.md")
    with open(prompt_path, "w") as f:
        f.write(seed_prompt)
    browser_root = os.path.join(base_dir, ".browser_use")
    return Agent(
        llm=LLM(model=lm_model),
        tools=[Tool(
            name=BrowserToolSet.name,
            params={
                "user_data_dir": os.path.join(browser_root, "profile"),
                "downloads_path": os.path.join(browser_root, "downloads"),
            },
        )],
        system_prompt_filename=prompt_path,
    )
'''
    else:
        code = '''\
from openhands.sdk import LLM, Agent, Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
import os

def build_agent(base_dir, lm_model, seed_prompt):
    prompt_path = os.path.join(base_dir, "system_prompt.md")
    with open(prompt_path, "w") as f:
        f.write(seed_prompt)
    return Agent(
        llm=LLM(model=lm_model),
        tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
        system_prompt_filename=prompt_path,
    )
'''
    return {"agent_code": code}


def requires_eval_lm(task_id: str) -> bool:
    """Return True if the task requires an eval LM to be specified."""
    return task_id == "webtest"
