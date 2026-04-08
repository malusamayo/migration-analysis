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
from openhands.sdk.context import (
    Skill,
)
from openhands.workspace import DockerWorkspace

from .debug_utils import patch_llm_for_debugging
from .task_setups import setup_workspace, get_mcp_config


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


def detect_platform() -> str:
    """Detects the correct platform string for container images."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"


def construct_docker_workspace(workspace_dir, system_prompt_path, skills):
    """
    Construct Docker workspace configuration with volumes and environment variables.

    Returns:
        Tuple of (docker_workspace_path, docker_system_prompt_path, docker_volumes, forward_env, workspace_dir)
    """
    all_envs = os.environ.copy()
    env_config = dotenv_values(".env")
    forward_env = [key for key in env_config.keys() if key in all_envs]

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

    return (docker_workspace_path, docker_system_prompt_path, docker_volumes, forward_env, workspace_dir)


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

        # export cost and total tokens
        metrics = conversation.conversation_stats.get_combined_metrics().get()

        conversation_data = {
            "conversation_id": str(conversation.id),
            "eval_output": final_message,
            "events": events,
            "metrics": metrics,
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
    ):
    """
    Core logic for running an agentic conversation.

    Args:
        agent: The agent to use for conversation
        workspace_obj: Workspace object (string path or DockerWorkspace)
        log_dir: Directory to save conversation trace
        example: Example data dictionary containing 'prompt' field
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]

    Returns:
        dict: Example with added 'eval_result' field containing evaluation output and scores
    """
    conversation = None
    error = None
    try:
        conversation = Conversation(agent=agent, workspace=workspace_obj)
        instruction = example['prompt']

        if skill_mode == "agent_decided":
            instruction += "\n\n### Hints\nThere are also some provided skills listed above. Please read the markdown file for more details when you find any skills relevant to your current context."

        conversation.send_message(instruction)

        kwargs = {}
        if isinstance(conversation, RemoteConversation):
            kwargs["timeout"] = 1200  # seconds
        
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
        setup_commands: List[str] = [],
        tools: List[Tool] = None,
        workspace_fn = None,
        agent_file: str = None,
        post_docker_fn = None,
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

    Returns:
        dict: Example with added 'eval_result' field containing evaluation output and scores

    Note:
        When use_docker=True:
        - Sets up a sandboxed Docker container environment
        - Mounts workspace and system prompt into container
        - Uses server_image "migration_analysis:latest"
    """
    example = copy.deepcopy(example)

    llm = LLM(model=lm.model) #, temperature=lm.kwargs.get("temperature"))
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
            seed_prompt = Path(prompt_path).read_text()
            agent = mod.build_agent(base_dir=str(base_dir), lm_model=lm.model, seed_prompt=seed_prompt)
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
            forward_env,
            workspace_dir) = construct_docker_workspace(workspace_dir.absolute(), system_prompt_path, skills)

        agent = _build_agent(docker_workspace_path, docker_system_prompt_path)

        with DockerWorkspace(
            working_dir=docker_workspace_path,
            server_image=server_image,
            platform=detect_platform(),
            volumes=docker_volumes,
            forward_env=forward_env,
            user=f"{os.getuid()}:{os.getgid()}",
        ) as docker_workspace:
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
        )
