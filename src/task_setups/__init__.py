import json
import os
import random
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
from . import ab_testing as _ab_testing
from . import replicatorbench as _replicatorbench
from . import browsecompplus as _browsecompplus
from .corpus_reader import get_corpus_reader


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
    if task_id == "ab_testing":
        example = dict(example)
        example["prompt"] = _ab_testing.TASK_INSTRUCTION
        return example
    if task_id == "replicatorbench":
        example = dict(example)
        example["prompt"] = (
            "Read task_context.json and original_paper.pdf, extract the focal claim "
            "into post_registration.json, write a replication plan to replication_info.json, "
            "use the available files in replication_data/ to execute the replication and "
            "save the executed result to execution_results.json, then summarize the "
            "run in interpret_results.json."
        )
        return example
    if task_id == "browsecomplongcontext":
        example = dict(example)
        example["prompt"] = (
            f"The web pages are in `context.txt` in your workspace.\n\nQuestion: {example['query']}"
        )
        return example
    if task_id == "browsecompplus":
        example = dict(example)
        example["prompt"] = _browsecompplus.format_prompt(example["query"])
        return example
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

    if task_id == "oolong":
        with open(os.path.join(workspace_dir, "context.txt"), "w") as f:
            f.write(example["context_window_text"])

    if task_id == "docbench":
        shutil.copy(example["pdf_path"], os.path.join(workspace_dir, "document.pdf"))

    if task_id == "browsecomplongcontext":
        reader = get_corpus_reader()
        all_docids = list(example["gold_docs"]) + list(example["negative_docs"])
        rng = random.Random(int(example.get("query_id", 0)))
        rng.shuffle(all_docids)
        pages = []
        for docid in all_docids:
            doc = reader.get(docid)
            pages.append(f"--- Document: {doc['url']} ---\n{doc['text']}\n--- End of Document ---")
        with open(os.path.join(workspace_dir, "context.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(pages))

    if task_id == "ab_testing":
        _ab_testing.setup_workspace(workspace_dir, str(log_dir), example)

    if task_id == "replicatorbench":
        _replicatorbench.setup_workspace(workspace_dir, str(log_dir), example)


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
    if task_id == "browsecompplus" and start_servers:
        proc = _browsecompplus.start_server()
        return {"browsecompplus_server": proc}
    return {}


def teardown_servers(task_id: str, servers_started: dict) -> None:
    """Stop any servers started by setup_servers."""
    if task_id == "webarena" and servers_started:
        stop_webarena_servers(list(servers_started))
    if task_id == "browsecompplus" and servers_started.get("browsecompplus_server"):
        _browsecompplus.stop_server(servers_started["browsecompplus_server"])


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
    # elif task_id == "build-pov-ray":
    #     from ..task_evals.build_pov_ray import run_single_instance_eval
    #     return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 32}
    elif task_id == "ab_testing":
        from ..task_evals.ab_testing import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 16}
    elif task_id == "oolong":
        from ..task_evals.oolong import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 32}
    elif task_id == "replicatorbench":
        from ..task_evals.replicatorbench import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 16}
    elif task_id == "docbench":
        from ..task_evals.docbench import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 16}
    elif task_id == "browsecomplongcontext":
        from ..task_evals.browsecomplongcontext import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 16}
    elif task_id == "browsecompplus":
        from ..task_evals.browsecompplus import run_single_instance_eval
        return {"eval_function": run_single_instance_eval, "use_process": False, "max_workers": 16}
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
    elif task_id == "ab_testing":
        code = '''\
from openhands.sdk import LLM, Agent, Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
import os

def build_agent(base_dir, lm_model, seed_prompt):
    prompt_path = os.path.join(base_dir, "system_prompt.md")
    with open(prompt_path, "w") as f:
        f.write(seed_prompt)
    from src.task_setups.ab_testing import get_mcp_config
    mcp_config = get_mcp_config(base_dir)
    return Agent(
        llm=LLM(model=lm_model),
        tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
        system_prompt_filename=prompt_path,
        mcp_config=mcp_config,
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
    return task_id in ("webtest", "docbench", "browsecomplongcontext", "browsecompplus")


def get_mcp_config(task_id: str, workspace_dir: str) -> dict:
    """Return an mcp_config dict for tasks that require MCP servers, else {}."""
    if task_id == "ab_testing":
        return _ab_testing.get_mcp_config(workspace_dir)
    if task_id == "browsecompplus":
        return _browsecompplus.get_mcp_config()
    return {}
