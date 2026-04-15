import pandas as pd
import dspy
import json
import yaml
import os
import platform
import traceback
from functools import partial
from copy import deepcopy
from typing import List, Any, Optional

from openhands.sdk.context.agent_context import AgentContext
from .review.trajectory_utils import convert_json_to_markdown

import os
import numpy as np
import copy
import json
from pathlib import Path

import platform
from dotenv import load_dotenv, dotenv_values

from openhands.sdk import (
    LLM, Agent, 
    Conversation, RemoteConversation,
    Tool, AgentContext
)
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.context import (
    Skill,
)
from openhands.tools.delegate import (
    DelegationVisualizer,
)
from openhands.workspace import DockerWorkspace

from contextlib import contextmanager

from .debug_utils import patch_llm_for_debugging
from .docker_utils import get_forward_env, make_docker_kwargs
from .task_setups import setup_workspace, get_mcp_config
from .utils import build_sdk_llm


_TRACE_DELEGATE_EXECUTOR_ATTR = "_trace_delegate_executor"
_SUBAGENTS_DIR = "subagents"


@contextmanager
def get_workspace_context(
    workspace_path: str,
    use_docker: bool = False,
    server_image: str = "migration-analysis:latest",
    docker_network: Optional[str] = None,
    docker_workspace_path: str = "/workspace/project",
    skills: list = None,
):
    """Context manager that returns the appropriate workspace object.

    Args:
        workspace_path: Local filesystem path to workspace directory
        use_docker: Whether to use Docker workspace
        server_image: Docker image to use
        docker_network: Docker network to connect to
        docker_workspace_path: Path inside Docker container
        skills: List of skills to mount (will be modified in-place for Docker paths)

    Yields:
        tuple: (workspace_obj, workspace_path_for_agent)
            - workspace_obj: DockerWorkspace or string path for Conversation
            - workspace_path_for_agent: Path to use in agent config (Docker path or local path)
    """
    if use_docker:
        docker_volumes = [f"{os.path.abspath(workspace_path)}:{docker_workspace_path}"]

        # Mount vertex-ai.json if it exists
        vertex_ai_path = os.path.abspath('.vertex-ai.json')
        if os.path.exists(vertex_ai_path):
            docker_volumes.append(f"{vertex_ai_path}:/workspace/.vertex-ai.json:ro")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/.vertex-ai.json"

        # Mount skills and update their paths for Docker
        if skills:
            for skill in skills:
                skill_dir = os.path.dirname(os.path.abspath(skill.source))
                docker_skill_dir = f"/workspace/skills/{os.path.basename(skill_dir)}"
                docker_volumes.append(f"{skill_dir}:{docker_skill_dir}:ro")
                skill.source = os.path.join(docker_skill_dir, "SKILL.md")

        with DockerWorkspace(**make_docker_kwargs(
            docker_workspace_path,
            server_image,
            docker_volumes,
            docker_network
        )) as docker_workspace:
            yield docker_workspace, docker_workspace_path
    else:
        yield workspace_path, workspace_path


def _get_delegate_executor(conversation: Conversation):
    """Return the live delegate executor when available.

    During long runs the agent's tool definitions may be replaced or
    reinitialized, which can leave ``conversation.agent.tools_map['delegate']``
    pointing at a fresh executor with no sub-agent state. Fall back to the
    executor cached on the conversation object by DelegateExecutor.
    """
    try:
        delegate_tool = conversation.agent.tools_map.get("delegate")
    except RuntimeError:
        delegate_tool = None

    delegate_executor = getattr(delegate_tool, "executor", None)
    cached_executor = getattr(conversation, _TRACE_DELEGATE_EXECUTOR_ATTR, None)

    live_sub_agents = getattr(delegate_executor, "_sub_agents", None)
    if live_sub_agents:
        return delegate_executor

    cached_sub_agents = getattr(cached_executor, "_sub_agents", None)
    if cached_sub_agents:
        return cached_executor

    if delegate_executor is not None:
        return delegate_executor

    return cached_executor


def run_with_agent(
        agent: Agent,
        example: dict,
        workspace: str,
        task_id: str = None,
        workspace_scripts: dict[str, str] = None,
    ) -> dict:
    """Run an agent built by candidate code on a single task example.

    Args:
        agent: Pre-built Agent object (from build_agent()).
        example: Example data dict containing 'prompt' field.
        workspace: Workspace directory for the agent.
        task_id: Task identifier (for workspace setup).
        workspace_scripts: Optional {filename: content} scripts to place in workspace.

    Returns:
        dict: Example with added 'run_result' field.
    """
    example = copy.deepcopy(example)
    workspace_dir = Path(workspace)
    log_dir = workspace_dir.parent / f"{workspace_dir.name}_logs"

    if task_id:
        setup_workspace(task_id, workspace, log_dir, example)

    # Deploy workspace scripts if provided
    if workspace_scripts:
        for filename, content in workspace_scripts.items():
            script_path = workspace_dir / filename
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(content)

    return _run_agentic_conversation(
        agent=agent,
        workspace_obj=str(workspace_dir),
        log_dir=str(log_dir),
        example=example,
    )


def construct_docker_workspace(workspace_dir, system_prompt_path, skills):
    """
    Construct Docker workspace volumes and paths for the collect/agent path.

    Returns:
        Tuple of (docker_workspace_path, docker_system_prompt_path, docker_volumes, workspace_dir)
    """
    docker_volumes = []

    docker_workspace_path = "/workspace/project"
    docker_volumes.append(f"{workspace_dir}:{docker_workspace_path}")

    docker_system_prompt_path = "/workspace/prompt.md"
    docker_volumes.append(f"{system_prompt_path}:{docker_system_prompt_path}")

    docker_volumes.append(f"{os.path.abspath('.vertex-ai.json')}:/workspace/.vertex-ai.json:ro")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/.vertex-ai.json"

    for skill in skills:
        skill_dir = os.path.dirname(os.path.abspath(skill.source))
        docker_skill_dir = f"/workspace/skills/{os.path.basename(skill_dir)}"
        docker_volumes.append(f"{skill_dir}:{docker_skill_dir}:ro")
        skill.source = os.path.join(docker_skill_dir, "SKILL.md")

    return (docker_workspace_path, docker_system_prompt_path, docker_volumes, workspace_dir)


def _fetch_remote_conversation_info(client, conversation_id: str) -> dict[str, Any]:
    """Fetch a remote conversation info snapshot."""
    resp = client.get(f"/api/conversations/{conversation_id}")
    resp.raise_for_status()
    return resp.json()


def _fetch_remote_conversation_events(client, conversation_id: str) -> list[dict[str, Any]]:
    """Fetch all non-state-update events for a remote conversation."""
    events: list[dict[str, Any]] = []
    page_id = None
    while True:
        params = {"limit": 100}
        if page_id:
            params["page_id"] = page_id
        resp = client.get(
            f"/api/conversations/{conversation_id}/events/search",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
        events.extend(data.get("items", []))
        page_id = data.get("next_page_id")
        if not page_id:
            break

    return [
        event for event in events
        if event.get("kind") != "ConversationStateUpdateEvent"
    ]


def _extract_text_content(content: list[dict[str, Any]] | None) -> str:
    """Extract plain text from event/message content blocks."""
    if not content:
        return ""
    text_parts = []
    for block in content:
        if block.get("type") == "text" and block.get("text"):
            text_parts.append(block["text"])
    return "\n".join(text_parts).strip()


def _extract_first_user_message_text(events: list[dict[str, Any]]) -> str:
    """Return the first user-authored message text in a conversation."""
    for event in events:
        if event.get("kind") != "MessageEvent" or event.get("source") != "user":
            continue
        llm_message = event.get("llm_message") or {}
        text = _extract_text_content(llm_message.get("content"))
        if text:
            return text
    return ""


def _extract_delegate_tasks(conversation: Conversation) -> dict[str, str]:
    """Collect the first delegated task text for each sub-agent."""
    tasks_by_agent: dict[str, str] = {}
    for event in conversation.state.events:
        event_dict = event.model_dump()
        if event_dict.get("kind") != "ActionEvent" or event_dict.get("tool_name") != "delegate":
            continue
        action = event_dict.get("action") or {}
        if action.get("command") != "delegate":
            continue
        for agent_id, task in (action.get("tasks") or {}).items():
            tasks_by_agent.setdefault(agent_id, task)
    return tasks_by_agent


def _extract_spawn_order(conversation: Conversation) -> list[str]:
    """Collect sub-agent IDs in first-seen spawn order."""
    seen: set[str] = set()
    spawn_order: list[str] = []
    for event in conversation.state.events:
        event_dict = event.model_dump()
        if event_dict.get("kind") != "ActionEvent" or event_dict.get("tool_name") != "delegate":
            continue
        action = event_dict.get("action") or {}
        if action.get("command") != "spawn":
            continue
        for agent_id in action.get("ids") or []:
            if agent_id in seen:
                continue
            seen.add(agent_id)
            spawn_order.append(agent_id)
    return spawn_order


def _extract_metrics_from_remote_info(info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover metrics and per-usage breakdown from remote conversation info."""
    metrics = info.get("metrics") or {}
    raw_usage = (info.get("stats") or {}).get("usage_to_metrics") or {}
    metrics_breakdown = raw_usage

    if raw_usage:
        try:
            stats = ConversationStats.model_validate({"usage_to_metrics": raw_usage})
            metrics_breakdown = {
                usage_id: usage_metrics.get()
                for usage_id, usage_metrics in stats.usage_to_metrics.items()
            }
            if not metrics:
                metrics = stats.get_combined_metrics().get()
        except Exception as e:
            print(f"Warning: Failed to derive metrics from remote stats: {e}")

    return metrics or {}, metrics_breakdown or {}


def _read_subagent_events_from_docker(workspace, subagent_dir: str) -> list[dict]:
    """Read all event JSON files from a sub-agent's events directory inside Docker."""
    cmd = f"""python3 -c "
import json
from pathlib import Path

events_dir = Path('{subagent_dir}') / 'events'
if not events_dir.exists():
    print('[]')
else:
    events = []
    for event_file in sorted(events_dir.glob('*.json')):
        try:
            with open(event_file) as f:
                events.append(json.load(f))
        except Exception:
            pass
    print(json.dumps(events))
"
"""
    try:
        result = workspace.execute_command(cmd, timeout=30.0)
        if result.exit_code == 0:
            return json.loads(result.stdout)
        return []
    except Exception:
        return []


def _fetch_remote_subagent_data(conversation) -> dict:
    """Fetch sub-agent data for RemoteConversation by reading files from Docker.

    Sub-agents created by DelegateExecutor are not registered in ConversationService,
    so we can't use the HTTP API to fetch them. Instead, we use workspace.execute_command
    to read the event files directly from /workspace/workspace/conversations/subagents/
    inside the running Docker container.

    Metrics are available from the parent conversation's metrics_breakdown under
    keys like "delegate:agent_id", so we extract them from there.

    Returns a dict keyed by agent_id with "conversation_id", "metrics",
    and "events" for each sub-agent. Returns an empty dict if no sub-agents found.
    """
    workspace = getattr(conversation, "workspace", None)
    if workspace is None:
        return {}

    # Check if workspace has execute_command (RemoteWorkspace/DockerWorkspace)
    if not hasattr(workspace, "execute_command"):
        return {}

    # Extract delegate tasks to match sub-agent directories to agent IDs
    delegate_tasks = _extract_delegate_tasks(conversation)
    spawn_order = _extract_spawn_order(conversation)

    # Get sub-agent metrics from parent conversation's metrics_breakdown
    # Metrics are stored under keys like "delegate:agent_id"
    client = getattr(conversation, "_client", None)
    subagent_metrics_by_id = {}
    if client:
        try:
            info = _fetch_remote_conversation_info(client, str(conversation.id))
            _, metrics_breakdown = _extract_metrics_from_remote_info(info)
            for key, metrics in metrics_breakdown.items():
                if key.startswith("delegate:"):
                    agent_id = key[len("delegate:"):]
                    subagent_metrics_by_id[agent_id] = metrics
        except Exception as e:
            print(f"Warning: Failed to extract subagent metrics from parent conversation: {e}")

    # List sub-agent directories in Docker
    # Sub-agents are stored under: /workspace/workspace/conversations/{parent_conv_id_hex}/subagents/{sub_agent_id_hex}/
    parent_conv_id_hex = conversation.id.hex
    subagents_root = f"/workspace/workspace/conversations/{parent_conv_id_hex}/subagents"
    cmd = f"ls -1 {subagents_root} 2>/dev/null || true"
    try:
        result = workspace.execute_command(cmd, timeout=10.0)
        if result.exit_code != 0 or not result.stdout.strip():
            return {}

        subagent_dirs = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    except Exception as e:
        print(f"Warning: Failed to list sub-agent directories: {e}")
        return {}

    if not subagent_dirs:
        return {}

    # Read each sub-agent's data
    subagent_data = {}
    for subagent_hex_id in subagent_dirs:
        subagent_dir = f"{subagents_root}/{subagent_hex_id}"

        # Read events
        events = _read_subagent_events_from_docker(workspace, subagent_dir)
        if not events:
            continue

        # Match to agent_id by comparing first user message with delegate tasks
        first_user_text = _extract_first_user_message_text(events)
        agent_id = None

        if first_user_text:
            # Try to match by task text
            for aid, task_text in delegate_tasks.items():
                if task_text == first_user_text and aid not in subagent_data:
                    agent_id = aid
                    break

        # Fallback: match by spawn order
        if agent_id is None and len(subagent_data) < len(spawn_order):
            agent_id = spawn_order[len(subagent_data)]

        if agent_id is None:
            print(f"Warning: Could not match sub-agent directory {subagent_hex_id} to agent_id")
            continue

        # Get metrics for this agent from parent conversation's breakdown
        metrics = subagent_metrics_by_id.get(agent_id, {})

        subagent_data[agent_id] = {
            "conversation_id": subagent_hex_id,
            "metrics": metrics,
            "events": events,
        }

    return subagent_data


def _get_subagent_data(conversation: Conversation) -> dict:
    """Extract per-subagent cost breakdown and events from the DelegateExecutor.

    Returns a dict keyed by agent_id with "metrics" and "events" for each subagent.
    Returns an empty dict if the conversation has no DelegateTool.
    """
    delegate_executor = _get_delegate_executor(conversation)
    if delegate_executor is None:
        # LocalConversation path failed — try fetching from server (RemoteConversation).
        return _fetch_remote_subagent_data(conversation)

    sub_agents = getattr(delegate_executor, "_sub_agents", {})

    # For RemoteConversation, sub_agents dict is empty because sub-agents run server-side
    # Fall back to reading from Docker filesystem
    if not sub_agents:
        return _fetch_remote_subagent_data(conversation)

    subagent_data = {}
    for agent_id, sub_conv in sub_agents.items():
        events = [e.model_dump() for e in sub_conv.state.events]
        events = [e for e in events if e["kind"] != "ConversationStateUpdateEvent"]
        subagent_data[agent_id] = {
            "conversation_id": str(sub_conv.id),
            "metrics": sub_conv.conversation_stats.get_combined_metrics().get(),
            "events": events,
        }
    return subagent_data


def save_conversation_trace(conversation: Conversation,
                            log_dir: str,
                            error: Optional[Exception] = None) -> str:
    events = [event.model_dump() for event in conversation.state.events]

    raw_event_path = os.path.join(log_dir, f"raw_trace_{conversation.id}.json")
    with open(raw_event_path, "w") as raw_file:
        json.dump(events, raw_file, indent=2)

    events = [e for e in events if e["kind"] != "ConversationStateUpdateEvent"]

    # Handle empty events (e.g., conversation failed during initialization)
    if not events:
        print(f"Warning: No events captured in conversation {conversation.id}")
        conversation_data = {
            "conversation_id": str(conversation.id),
            "eval_output": "",
            "events": [],
            "metrics": {},
            "metrics_breakdown": {},
            "subagents": {},
            "error": str(error) if error else "No events captured",
        }
    else:
        if events[-1]["kind"] == "ObservationEvent":
            final_message = events[-1]["observation"]["content"][0]["text"]
        elif events[-1]["kind"] == "MessageEvent":
            final_message = events[-1]["llm_message"]["content"][0]["text"]
        else:
            print(events[-1])
            print(f"Unexpected final event type {events[-1]['kind']}")
            return None

        # Combined cost across orchestrator + all subagents.
        # For RemoteConversation the client-side stats cache is stale
        # (the server never sends a stats update over WebSocket), so fetch
        # a fresh snapshot directly from the server HTTP API instead.
        client = getattr(conversation, "_client", None)
        if client is not None:
            try:
                info = _fetch_remote_conversation_info(client, str(conversation.id))
                metrics, metrics_breakdown = _extract_metrics_from_remote_info(info)
            except Exception as e:
                print(f"Warning: Failed to fetch fresh metrics from server: {e}")
                metrics = conversation.conversation_stats.get_combined_metrics().get()
                metrics_breakdown = {
                    usage_id: m.get()
                    for usage_id, m in conversation.conversation_stats.usage_to_metrics.items()
                }
        else:
            metrics = conversation.conversation_stats.get_combined_metrics().get()
            # Per-agent cost breakdown from usage_to_metrics
            # Keys: "default" (orchestrator), "delegate:<agent_id>" (each subagent)
            metrics_breakdown = {
                usage_id: m.get()
                for usage_id, m in conversation.conversation_stats.usage_to_metrics.items()
            }

        # Subagent conversations and their per-agent metrics
        subagents = _get_subagent_data(conversation)

        conversation_data = {
            "conversation_id": str(conversation.id),
            "eval_output": final_message,
            "events": events,
            "metrics": metrics,
            "metrics_breakdown": metrics_breakdown,
            "subagents": subagents,
            "error": str(error) if error else None,
        }

    trace_path = os.path.join(log_dir, f"trace_{conversation.id}.json")
    with open(trace_path, "w") as trace_file:
        json.dump(conversation_data, trace_file, indent=2)

    try:
        markdown_content = convert_json_to_markdown(conversation_data)
        markdown_path = os.path.join(log_dir, f"trace_{conversation.id}.md")
        with open(markdown_path, "w") as md_file:
            md_file.write(markdown_content)
    except Exception as e:
        print(f"Error converting JSON to markdown: {e}")

    return conversation_data

def _run_agentic_conversation(
        agent: Agent,
        workspace_obj: Any,
        log_dir: str,
        example: dict,
        skill_mode: str = "",
        max_time: Optional[float] = None,
        agent_file: Optional[str] = None,
    ):
    """
    Core logic for running an agentic conversation.

    Args:
        agent: The agent to use for conversation
        workspace_obj: Workspace object (string path or DockerWorkspace)
        log_dir: Directory to save conversation trace
        example: Example data dictionary containing 'prompt' field
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]
        max_time: Optional maximum runtime in seconds for RemoteConversation.run()
        agent_file: Optional path (inside container) to agent file for
            server-side loading.  Only used with RemoteConversation/Docker.

    Returns:
        dict: Example with added 'eval_result' field containing evaluation output and scores
    """
    conversation = None
    error = None
    try:
        visualizer = DelegationVisualizer(name="main")
        conversation = Conversation(agent=agent, workspace=workspace_obj, visualizer=visualizer, agent_file=agent_file)
        instruction = example['prompt']

        if skill_mode == "agent_decided":
            instruction += "\n\n### Hints\nThere are also some provided skills listed above. Please read the markdown file for more details when you find any skills relevant to your current context."

        conversation.send_message(instruction)

        kwargs = {}
        if isinstance(conversation, RemoteConversation):
            kwargs["timeout"] = max_time if max_time is not None else 1200
        
        conversation.run(**kwargs)

        return example

    except TimeoutError as e:
        print(f"⏰ Conversation timed out: {e}")
        error = e
    except Exception as e:
        print(f"Error during agentic execution: {e}")
        error = e
        
        traceback.print_exc()
        return example

    finally:
        if conversation:
            conversation_data = save_conversation_trace(conversation, log_dir, error)
            example["run_result"] = copy.deepcopy(conversation_data)
            
            print("🧹 Cleaning up conversation...")
            conversation.close()

def run_single_instance_agentic(
        lm: dspy.LM,
        system_prompt_path: str,
        example: dict,
        workspace: str,
        task_id: str = None,
        skills: List[Skill] = [],
        skill_mode: str = "",
        use_docker: bool = False,
        server_image: str = "",
        docker_network: str = None,
        setup_commands: List[str] = [],
        tools: List[Tool] = None,
        workspace_fn = None,
        agent_file: str = None,
        post_docker_fn = None,
        max_time: Optional[float] = None,
    ):
    """
    Run a single instance using OpenHands agents.

    Args:
        lm: Language model to use
        system_prompt_path: Path to the system prompt file
        example: Example data dictionary containing 'prompt' field
        workspace: Workspace directory for the agent
        skills: Optional list of pre-loaded Skill objects
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]
        use_docker: If True, use DockerWorkspace for containerized execution
        max_time: Optional maximum runtime in seconds for the task conversation

    Returns:
        dict: Example with added 'eval_result' field containing evaluation output and scores

    Note:
        When use_docker=True:
        - Sets up a sandboxed Docker container environment
        - Mounts workspace and system prompt into container
        - Uses server_image "migration_analysis:latest"
    """
    example = copy.deepcopy(example)

    llm = build_sdk_llm(lm)
    system_prompt_path = os.path.abspath(system_prompt_path)

    workspace_dir = Path(workspace)
    log_dir = workspace_dir.parent / f"{workspace_dir.name}_logs"
    if task_id:
        setup_workspace(task_id, workspace, log_dir, example)

    def _build_agent(base_dir, prompt_path):
        if agent_file is not None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_agent_module", agent_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            agent = mod.build_agent(base_dir=str(workspace_dir), llm=llm)
            # Remap system_prompt_filename to an absolute path under base_dir so that
            # render_template can find it regardless of the CWD.
            # In Docker mode, base_dir is the container-internal path (/workspace/project).
            # In non-Docker mode, base_dir is workspace_dir.absolute() (the host path).
            fname = os.path.basename(agent.system_prompt_filename)
            agent = agent.model_copy(update={"system_prompt_filename": os.path.join(base_dir, fname)})
            # Remap tool params (e.g. browser user_data_dir) from local workspace path to
            # Docker path. build_agent writes files using local paths (so host writes work),
            # but any runtime paths embedded in tool params must use the container-side path.
            local_workspace = str(workspace_dir)
            if base_dir != local_workspace:
                remapped_tools = []
                for tool in agent.tools:
                    if tool.params:
                        new_params = {
                            k: v.replace(local_workspace, base_dir, 1)
                            if isinstance(v, str) and v.startswith(local_workspace) else v
                            for k, v in tool.params.items()
                        }
                        remapped_tools.append(tool.model_copy(update={"params": new_params}))
                    else:
                        remapped_tools.append(tool)
                agent = agent.model_copy(update={"tools": remapped_tools})
            fn = getattr(mod, "get_workspace_scripts", None)
            if fn is not None:
                for filename, content in fn().items():
                    script_path = workspace_dir / filename
                    script_path.parent.mkdir(parents=True, exist_ok=True)
                    script_path.write_text(content)
            return agent
        if tools is None:
            _tools = [
                Tool(name=TerminalTool.name),
                Tool(name=FileEditorTool.name),
            ]
        else:
            _tools = tools
        agent_context = AgentContext(skills=skills or [])
        mcp_cfg = get_mcp_config(task_id, str(base_dir)) if task_id else {}
        agent_kwargs = dict(
            llm=llm,
            tools=_tools,
            system_prompt_filename=prompt_path,
            agent_context=agent_context,
        )
        if mcp_cfg:
            agent_kwargs["mcp_config"] = mcp_cfg
        return Agent(**agent_kwargs)

    if use_docker:
        # Docker execution path
        (docker_workspace_path,
            docker_system_prompt_path,
            docker_volumes,
            workspace_dir) = construct_docker_workspace(workspace_dir.absolute(), system_prompt_path, skills)

        docker_agent_file = None
        if agent_file is not None:
            abs_agent_file = os.path.abspath(agent_file)
            docker_agent_file = f"/workspace/_agent_config.py"
            docker_volumes.append(f"{abs_agent_file}:{docker_agent_file}:ro")

        agent = _build_agent(docker_workspace_path, docker_system_prompt_path)

        with DockerWorkspace(**make_docker_kwargs(docker_workspace_path, server_image, docker_volumes, docker_network)) as docker_workspace:
            for cmd in setup_commands:
                docker_workspace.execute_command(cmd, timeout=90.0)

            if workspace_fn:
                return workspace_fn(
                    workspace=docker_workspace,
                    system_prompt_path=docker_system_prompt_path,
                    example=example,
                    log_dir=log_dir,
                )

            result = _run_agentic_conversation(
                agent=agent,
                workspace_obj=docker_workspace,
                log_dir=log_dir,
                example=example,
                skill_mode=skill_mode,
                max_time=max_time,
                agent_file=docker_agent_file,
            )
            if post_docker_fn:
                post_docker_fn(
                    workspace=docker_workspace,
                    example=example,
                    workspace_dir=str(workspace_dir),
                )
            return result
    else:
        # patch_llm_for_debugging(Path(workspace))
        agent = _build_agent(workspace_dir.absolute(), system_prompt_path)

        return _run_agentic_conversation(
            agent=agent,
            workspace_obj=workspace,
            log_dir=log_dir,
            example=example,
            skill_mode=skill_mode,
            max_time=max_time,
        )
