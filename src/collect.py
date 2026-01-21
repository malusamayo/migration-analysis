import pandas as pd
import dspy
import json
import os
import shutil
import platform
import litellm
from datasets import load_dataset, concatenate_datasets
from functools import partial
from copy import deepcopy
from typing import List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import LM_DICT, batch_inference, use_lm
from .dataloader import CollectDataLoader, load_and_validate_results
from .review.trajectory_loader import convert_json_to_markdown

import os
import numpy as np
import tqdm
import copy
import time
import json
import argparse
import re
from pathlib import Path
from dotenv import dotenv_values

from openhands.sdk import LLM, Agent, Conversation, Tool, AgentContext
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.sdk.context import (
    Skill,
)
from openhands.workspace import DockerWorkspace

from .debug_utils import patch_llm_for_debugging

def detect_platform() -> str:
    """Detects the correct platform string for container images."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"

def discover_skills(
    skill_version: Optional[str] = None,
    skill_mode: str = "all_loaded",
) -> List[Skill]:
    """
    Discover and load skills for a given rollout version.

    Args:
        task_id: Task identifier (not used when skill_version contains full path)
        model_name: Model name (not used when skill_version contains full path)
        prompt_name: Prompt name (not used when skill_version contains full path)
        skill_version: Full relative path to skill folder (e.g., "results/webtest/model_default/skills/v1"), or None for no skills
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]

    Returns:
        List of loaded Skill objects (empty list if skill_version is None)
    """
    # If no skill_version specified, no skills to load
    if skill_version is None:
        return []

    # skill_version is now the full relative path
    skill_dir = Path(skill_version)

    skills = []

    if skill_dir.exists():
        # Load all SKILL.md files from skill_dir subdirectories
        skill_paths = []
        for skill_folder in skill_dir.iterdir():
            if skill_folder.is_dir():
                skill_file = skill_folder / "SKILL.md"
                if skill_file.exists():
                    try:
                        skills.append(Skill.load(path=str(skill_file), strict=False))
                        if skill_mode == "all_loaded":
                            skills[-1].is_agentskills_format = False
                        skill_paths.append(str(skill_file))
                    except Exception as e:
                        print(f"⚠️  Failed to load skill from {skill_file}: {e}")

        print(f"✅ Found {len(skills)} skills in {skill_dir}")

    else:
        print(f"⚠️  Warning: Skill directory not found: {skill_dir}")

    return skills

def run_single_instance(
        lm: dspy.LM,
        system_prompt: str,
        example: dict,
        seed: int,
        max_retries: int = 2,
        validation_fn: Optional[callable] = None,
        retry_gen_fn: Optional[callable] = None,
    ):
    """
    Run a single instance with retry logic and optional output validation.

    Args:
        lm: Language model to use
        system_prompt: System prompt for the model
        example: Example data dictionary
        seed: Random seed/rollout ID
        max_retries: Maximum number of retry attempts (default: 3)
        validation_fn: Optional function that takes the output string and returns (is_valid: bool, error_msg: str)
                      If None, no validation is performed.

    Returns:
        dict: Example with added fields including 'output', 'rollout_id', and optionally 'validation_errors'
    """
    example = copy.deepcopy(example)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['prompt']}
    ]

    validation_errors = []

    for _ in range(max_retries):
        response = lm(messages=messages, rollout_id=seed)
        output = response[0]

        # Validate output if validation function is provided
        if validation_fn is not None:
            is_valid, error_msg = validation_fn(output)
            if not is_valid:
                lm, messages = retry_gen_fn(lm, messages, error_msg)
                validation_errors.append(error_msg)
                continue
        
        # Success - output is valid or no validation required
        example["rollout_id"] = seed
        example["output"] = output
        if validation_errors:
            example["validation_errors"] = validation_errors
        return example

    return example

def save_conversation_trace(conversation: Conversation, workspace: str) -> str:
    events = [event.model_dump() for event in conversation.state.events]

    raw_event_path = os.path.join(workspace, f"raw_trace_{conversation.id}.json")
    with open(raw_event_path, "w") as raw_file:
        json.dump(events, raw_file, indent=2)

    events = [e for e in events if e["kind"] != "ConversationStateUpdateEvent"]
    if events[-1]["kind"] == "ObservationEvent":
        final_message = events[-1]["observation"]["content"][0]["text"]
    elif events[-1]["kind"] == "MessageEvent":
        final_message = events[-1]["llm_message"]["content"][0]["text"]
    else:
        print(events[-1])
        print(f"Unexpected final event type {events[-1]['kind']}")

    # export cost and total tokens
    metrics = conversation.conversation_stats.get_combined_metrics().get()
    
    conversation_data = {
        "conversation_id": str(conversation.id),
        "eval_output": final_message,
        "events": events,
        "metrics": metrics,
    }
    
    trace_path = os.path.join(workspace, f"trace_{conversation.id}.json")
    with open(trace_path, "w") as trace_file:
        json.dump(conversation_data, trace_file, indent=2)
    
    markdown_content = convert_json_to_markdown(conversation_data)
    markdown_path = os.path.join(workspace, f"trace_{conversation.id}.md")
    with open(markdown_path, "w") as md_file:
        md_file.write(markdown_content)
    
    return conversation_data

def construct_docker_workspace(workspace_dir, system_prompt_path, skills):
    """
    Construct Docker workspace configuration with volumes and environment variables.

    Args:
        workspace_dir: Local workspace directory path
        system_prompt_path: Path to system prompt file

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

def _run_agentic_conversation(
        agent: Agent,
        workspace_obj: Any,
        workspace_dir: str,
        example: dict,
        skill_mode: str = "",
    ):
    """
    Core logic for running an agentic conversation.

    Args:
        agent: The agent to use for conversation
        workspace_obj: Workspace object (string path or DockerWorkspace)
        workspace_dir: Directory to save conversation trace
        example: Example data dictionary containing 'prompt' field
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]

    Returns:
        dict: Example with added 'eval_result' field containing evaluation output and scores
    """
    conversation = None
    try:
        conversation = Conversation(agent=agent, workspace=workspace_obj)
        instruction = example['prompt']

        if skill_mode == "agent_decided":
            instruction += "\n\n### Hints\nThere are also some provided skills listed above. Please read the markdown file for more details when you find any skills relevant to your current context."

        conversation.send_message(instruction)
        conversation.run()

        conversation_data = save_conversation_trace(conversation, workspace_dir)
        example["eval_result"] = copy.deepcopy(conversation_data)

        return example

    except Exception as e:
        print(f"Error during agentic execution: {e}")
        import traceback
        traceback.print_exc()
        return example

    finally:
        if conversation:
            print("🧹 Cleaning up conversation...")
            conversation.close()

def _setup_workspace(task_id, workspace_dir: str, example: dict):
    # clean up previous contents
    shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    # set up workspace files
    if task_id == "webtest":
        with open(os.path.join(workspace_dir, "index.html"), "w") as f:
            f.write(example["html_content"])

def run_single_instance_agentic(
        lm: dspy.LM,
        system_prompt_path: str,
        example: dict,
        workspace: str,
        seed: int,
        task_id: str,
        skills: List[Skill] = [],
        skill_mode: str = "",
        use_docker: bool = True,
    ):
    """
    Run a single instance using OpenHands agents.

    Args:
        lm: Language model to use
        system_prompt_path: Path to the system prompt file
        example: Example data dictionary containing 'prompt' field
        workspace: Workspace directory for the agent
        seed: Random seed for reproducibility
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

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
    ]

    llm = LLM(model=lm.model)
    system_prompt_path = os.path.abspath(system_prompt_path)

    # Check if workspace already has trace*.md file - if so, skip execution
    workspace_path = Path(workspace)
    if workspace_path.exists():
        existing_traces = list(workspace_path.glob("trace*.md"))
        if existing_traces:
            print(f"⏭️  Skipping execution - workspace already has trace file: {existing_traces[0].name}")
            # Return example with existing trace data if available
            trace_json_files = list(workspace_path.glob("trace*.json"))
            if trace_json_files:
                try:
                    with open(trace_json_files[0], 'r') as f:
                        existing_data = json.load(f)
                        example["eval_result"] = existing_data
                        return example
                except Exception as e:
                    print(f"⚠️  Warning: Could not load existing trace data: {e}")
            return example

    _setup_workspace(task_id, workspace, example)

    if use_docker:
        # Docker execution path
        workspace_dir = os.path.abspath(workspace)
        (docker_workspace_path,
            docker_system_prompt_path,
            docker_volumes,
            forward_env,
            workspace_dir) = construct_docker_workspace(workspace_dir, system_prompt_path, skills)

        agent_context = AgentContext(skills=skills or [])
        agent = Agent(
            llm=llm,
            tools=tools,
            system_prompt_filename=docker_system_prompt_path,
            agent_context=agent_context,
        )

        with DockerWorkspace(
            working_dir=docker_workspace_path,
            server_image="migration-analysis:latest",
            platform=detect_platform(),
            volumes=docker_volumes,
            forward_env=forward_env,
        ) as docker_workspace:
            return _run_agentic_conversation(
                agent=agent,
                workspace_obj=docker_workspace,
                workspace_dir=workspace_dir,
                example=example,
                skill_mode=skill_mode,
            )
    else:
        patch_llm_for_debugging(Path(workspace))
        agent_context = AgentContext(skills=skills or [])
        agent = Agent(
            llm=llm,
            tools=tools,
            system_prompt_filename=system_prompt_path,
            agent_context=agent_context,
        )

        return _run_agentic_conversation(
            agent=agent,
            workspace_obj=workspace,
            workspace_dir=workspace,
            example=example,
            skill_mode=skill_mode,
        )

def run_task(
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        is_agentic: bool = False,
        max_examples: Optional[int] = None,
        n_responses: int = 1,
        batch_size: int = 16,
        resume: bool = True,
        skill_version: Optional[str] = None,
        skill_mode: str = "all_loaded",
    ):
    """
    Run a task with specified model and prompt.

    Args:
        task_id: Task identifier
        model_name: Name of the model to use
        prompt_name: Name of the prompt template (default: "default")
        is_agentic: Whether to use agentic execution
        max_examples: Maximum number of examples to process
        n_responses: Number of responses to generate per example
        batch_size: Batch size for collection
        resume: Whether to resume from existing results
        skill_version: Path to skill folder (e.g., "skills/v1"), or None for no skills
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]

    Note:
        For agentic execution, volume mounts are automatically inferred from:
        - System prompt directory (mounted as /workspace/prompts:ro)
        - Workspace path (mounted as /workspace/data)
    """
    from .dataloader import generate_rollout_version

    # Auto-generate rollout_version
    rollout_version = generate_rollout_version(
        skill_version=skill_version,
        skill_mode=skill_mode,
    )
    print(f"📦 Auto-generated rollout version: {rollout_version}")

    # Construct full skill path if skill_version is provided
    full_skill_path = None
    if skill_version:
        # If skill_version already starts with "results/", use it as-is
        if skill_version.startswith("results/"):
            full_skill_path = skill_version
        else:
            # Otherwise, construct the full path
            full_skill_path = f"results/{task_id}/{model_name}_{prompt_name}/{skill_version}"

    # Discover and load skills for this rollout version
    skills = discover_skills(
        skill_version=full_skill_path,
        skill_mode=skill_mode,
    )

    # Initialize data loader
    data_loader = CollectDataLoader(
        task_id=task_id,
        model_name=model_name,
        prompt_name=prompt_name,
        is_agentic=is_agentic,
        max_examples=max_examples,
        n_responses=n_responses,
        rollout_version=rollout_version,
        skill_version=skill_version,
        skill_mode=skill_mode,
    )

    output_path = f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}/run.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results if resuming
    if resume:
        results, start_idx = load_and_validate_results(output_path, data_loader)
    else:
        results, start_idx = [], 0

    # Determine which function to use
    if is_agentic:
        run_function = run_single_instance_agentic
        use_process = True
        max_workers = 16
    else:
        run_function = run_single_instance
        use_process = False
        max_workers = 32

    # Process remaining data in batches
    for i in range(start_idx, len(data_loader), batch_size):
        # Get batch arguments from data loader
        args_list = data_loader.get_batch_args(
            batch_start=i,
            batch_size=batch_size
        )

        # Add skills to agentic args
        if is_agentic and skills:
            for args in args_list:
                args["skills"] = skills
                args["skill_mode"] = skill_mode

        batch_results = batch_inference(
            run_function,
            args_list,
            use_process=use_process,
            max_workers=max_workers,
        )
        results.extend(batch_results)

        # Write partial results after each batch
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved partial results ({len(results)}/{len(data_loader)} completed)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task with specified model and prompt.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use.")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to run.")
    parser.add_argument("--prompt_name", type=str, default="default", help="Prompt name to use.")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process.")
    parser.add_argument("--n_responses", type=int, default=1, help="Number of responses to generate per example.")
    parser.add_argument("--is_agentic", action="store_true", help="Whether to use agentic execution.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for collection.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh instead of resuming from existing results.")

    # New parameters for automatic rollout versioning
    parser.add_argument("--skill_version", type=str, default=None,
                        help="Path to skill folder (e.g., 'skills/v1'), or None for no skills")
    parser.add_argument("--skill_mode", type=str, default="all_loaded",
                        choices=["all_loaded", "agent_decided", "monitor_decided"],
                        help="Skill mode: all_loaded, agent_decided, or monitor_decided")

    args = parser.parse_args()

    run_task(
        task_id=args.task_id,
        model_name=args.model,
        prompt_name=args.prompt_name,
        is_agentic=args.is_agentic,
        max_examples=args.max_examples,
        n_responses=args.n_responses,
        batch_size=args.batch_size,
        resume=args.resume,
        skill_version=args.skill_version,
        skill_mode=args.skill_mode,
    )
