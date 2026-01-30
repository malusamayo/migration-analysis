import pandas as pd
import dspy
import json
import yaml
import os
import shutil
import platform
import traceback
import litellm
from datasets import load_dataset, concatenate_datasets
from functools import partial
from copy import deepcopy
from typing import List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openhands.sdk.context.agent_context import AgentContext
from .utils import LM_DICT, batch_inference, use_lm
from .dataloader import CollectDataLoader
from .review.trajectory_utils import convert_json_to_markdown

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
from .review.skill_manager import SkillManager

def detect_platform() -> str:
    """Detects the correct platform string for container images."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"

def discover_skills(
    skill_path: Optional[str] = None,
    skill_mode: str = "all_loaded",
    subset_mode: str = "all",
    subset_k: Optional[int] = None,
    subset_seed: Optional[int] = None,
) -> List[Skill]:
    """
    Discover and load skills based on metadata file.

    Args:
        skill_path: Path to metadata.yaml file or directory containing it (e.g., "results/webtest/model_default/skills/v1/metadata.yaml"), or None for no skills
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]
        subset_mode: One of ["all", "top_k", "random"] - method to select skill subset
        subset_k: Number of skills to select when subset_mode is "top_k" or "random"
        subset_seed: Random seed for reproducibility when subset_mode is "random"

    Returns:
        List of loaded Skill objects (empty list if skill_path is None)
    """
    # If no skill_path specified, no skills to load
    if skill_path is None:
        return []

    # Convert to Path and determine metadata file path
    skill_path = Path(skill_path)

    # If skill_path is a directory, look for metadata.yaml inside it
    if skill_path.is_dir():
        metadata_path = skill_path / "metadata.yaml"
        base_dir = skill_path
    else:
        # Assume skill_path points directly to metadata.yaml
        metadata_path = skill_path
        base_dir = skill_path.parent

    skills = []

    assert metadata_path.exists(), f"Metadata file not found: {metadata_path}"

    # Step 1: Use SkillManager to load skills and get subset
    print(f"Loading skills from {metadata_path}...")
    manager = SkillManager()
    manager.load_skills(metadata_path)

    # Step 2: Select skill subset based on mode
    if subset_mode == "all":
        selected_skills = manager.skills
        print(f"Using all {len(selected_skills)} skills")
    elif subset_mode == "top_k":
        if subset_k is None:
            raise ValueError("subset_k must be specified when subset_mode='top_k'")
        selected_skills = manager.get_top_k_skills(subset_k)
        print(f"Selected top {len(selected_skills)} skills by duplicate count")
    elif subset_mode == "random":
        if subset_k is None:
            raise ValueError("subset_k must be specified when subset_mode='random'")
        selected_skills = manager.get_random_skills(subset_k, seed=subset_seed)
        print(f"Selected {len(selected_skills)} random skills (seed={subset_seed})")
    else:
        raise ValueError(f"Invalid subset_mode: {subset_mode}. Must be one of ['all', 'top_k', 'random']")

    # Step 3: Load selected skills as openhands Skill objects
    for skill in selected_skills:
        skill_file = base_dir / skill.skill_name / "SKILL.md"
        if skill_file.exists():
            try:
                openhands_skill = Skill.load(path=str(skill_file), strict=False)
                if skill_mode == "all_loaded":
                    openhands_skill.is_agentskills_format = False
                skills.append(openhands_skill)
            except Exception as e:
                print(f"⚠️  Failed to load skill from {skill_file}: {e}")
        else:
            print(f"⚠️  Warning: Skill file not found: {skill_file}")

    print(f"✅ Loaded {len(skills)}/{len(selected_skills)} skills")

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

def save_conversation_trace(conversation: Conversation, 
                            workspace: str,
                            error: Optional[Exception] = None) -> str:
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
    
    trace_path = os.path.join(workspace, f"trace_{conversation.id}.json")
    with open(trace_path, "w") as trace_file:
        json.dump(conversation_data, trace_file, indent=2)
    
    try:
        markdown_content = convert_json_to_markdown(conversation_data)
        markdown_path = os.path.join(workspace, f"trace_{conversation.id}.md")
        with open(markdown_path, "w") as md_file:
            md_file.write(markdown_content)
    except Exception as e:
        print(f"Error converting JSON to markdown: {e}")
    
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
            conversation_data = save_conversation_trace(conversation, workspace_dir, error)
            example["run_result"] = copy.deepcopy(conversation_data)
            
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

    llm = LLM(model=lm.model) #, temperature=lm.kwargs.get("temperature"))
    system_prompt_path = os.path.abspath(system_prompt_path)

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
        skill_path: Optional[str] = None,
        skill_mode: str = "all_loaded",
        subset_mode: str = "all",
        subset_k: Optional[int] = None,
        subset_seed: Optional[int] = None,
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
        skill_path: Path to skill folder containing metadata.yaml (e.g., "skills/v1"), or None for no skills
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]
        subset_mode: One of ["all", "top_k", "random"] - method to select skill subset
        subset_k: Number of skills to select when subset_mode is "top_k" or "random"
        subset_seed: Random seed for reproducibility when subset_mode is "random"

    Note:
        For agentic execution, volume mounts are automatically inferred from:
        - System prompt directory (mounted as /workspace/prompts:ro)
        - Workspace path (mounted as /workspace/data)
    """
    from .dataloader import generate_rollout_version

    # Auto-generate rollout_version
    rollout_version = generate_rollout_version(
        skill_version=skill_path,
        skill_mode=skill_mode,
        subset_mode=subset_mode,
        subset_k=subset_k,
        subset_seed=subset_seed,
    )
    print(f"📦 Auto-generated rollout version: {rollout_version}")

    # Construct full skill metadata path if skill_path is provided
    skill_metadata_path = None
    if skill_path:
        # If skill_path already starts with "results/", use it as-is
        if skill_path.startswith("results/"):
            skill_metadata_path = skill_path
        else:
            # Otherwise, construct the full path
            skill_metadata_path = f"results/{task_id}/{model_name}_{prompt_name}/{skill_path}"

        # Append metadata.yaml if not already included
        if not skill_metadata_path.endswith(".yaml"):
            skill_metadata_path = f"{skill_metadata_path}/metadata.yaml"

    # Discover and load skills for this rollout version
    skills = discover_skills(
        skill_path=skill_metadata_path,
        skill_mode=skill_mode,
        subset_mode=subset_mode,
        subset_k=subset_k,
        subset_seed=subset_seed,
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
        skill_version=skill_path,
        skill_mode=skill_mode,
        subset_mode=subset_mode,
        subset_k=subset_k,
        subset_seed=subset_seed,
        resume=resume,
    )

    output_path = f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}/run.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get completed and pending tasks from data loader
    results = data_loader.get_completed_results()
    args_list = data_loader.get_pending_args()

    # Determine which function to use
    if is_agentic:
        run_function = run_single_instance_agentic
        use_process = True
        max_workers = 16
    else:
        run_function = run_single_instance
        use_process = False
        max_workers = 32

    # Add skills to agentic args
    if is_agentic and skills:
        for args in args_list:
            args["skills"] = skills
            args["skill_mode"] = skill_mode

    # Define callback to save partial results
    def write_partial_results(completed_results, total_count):
        # Combine already-completed results with new completions
        all_results = results + completed_results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Saved partial results ({len(all_results)}/{len(data_loader)} completed)")

    # Process remaining data with periodic callbacks
    if args_list:
        batch_results = batch_inference(
            run_function,
            args_list,
            use_process=use_process,
            max_workers=max_workers,
            on_batch_complete=write_partial_results,
            batch_size=batch_size,
        )
        results.extend(batch_results)

    # Write final results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Completed all {len(results)}/{len(data_loader)} examples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task with specified model and prompt.")

    # Config file support
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file with default arguments")

    parser.add_argument("--model_name", type=str, help="Model name to use.")
    parser.add_argument("--task_id", type=str, help="Task ID to run.")
    parser.add_argument("--prompt_name", type=str, default=None, help="Prompt name to use.")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process.")
    parser.add_argument("--n_responses", type=int, default=None, help="Number of responses to generate per example.")
    parser.add_argument("--is_agentic", action="store_true", help="Whether to use agentic execution.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for collection.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh instead of resuming from existing results.")

    # Parameters for automatic rollout versioning
    parser.add_argument("--skill_path", type=str, default=None,
                        help="Path to skill folder containing metadata.yaml (e.g., 'skills/v1'), or None for no skills")
    parser.add_argument("--skill_mode", type=str, default=None,
                        choices=["all_loaded", "agent_decided", "monitor_decided"],
                        help="Skill mode: all_loaded, agent_decided, or monitor_decided")

    # Parameters for skill subset selection
    parser.add_argument("--subset_mode", type=str, default=None,
                        choices=["all", "top_k", "random"],
                        help="Skill subset mode: all, top_k, or random")
    parser.add_argument("--subset_k", type=int, default=None,
                        help="Number of skills to select when subset_mode is 'top_k' or 'random'")
    parser.add_argument("--subset_seed", type=int, default=None,
                        help="Random seed for reproducibility when subset_mode is 'random'")

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"📋 Loaded config from {args.config}")

    # Merge config file with command line arguments (command line takes precedence)
    # Set defaults from config
    model_name = args.model_name if args.model_name is not None else config.get("model_name")
    task_id = args.task_id if args.task_id is not None else config.get("task_id")
    prompt_name = args.prompt_name if args.prompt_name is not None else config.get("prompt_name", "default")
    max_examples = args.max_examples if args.max_examples is not None else config.get("max_examples")
    n_responses = args.n_responses if args.n_responses is not None else config.get("n_responses", 1)
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 16)
    skill_path = args.skill_path if args.skill_path is not None else config.get("skill_path", config.get("skill_version"))
    skill_mode = args.skill_mode if args.skill_mode is not None else config.get("skill_mode", "all_loaded")
    subset_mode = args.subset_mode if args.subset_mode is not None else config.get("subset_mode", "all")
    subset_k = args.subset_k if args.subset_k is not None else config.get("subset_k")
    subset_seed = args.subset_seed if args.subset_seed is not None else config.get("subset_seed")

    # Handle boolean flags specially
    is_agentic = args.is_agentic or config.get("is_agentic", False)
    resume = args.resume if args.resume else config.get("resume", True)

    # Validate required arguments
    if not model_name:
        parser.error("--model_name is required (either via command line or config file)")
    if not task_id:
        parser.error("--task_id is required (either via command line or config file)")

    run_task(
        task_id=task_id,
        model_name=model_name,
        prompt_name=prompt_name,
        is_agentic=is_agentic,
        max_examples=max_examples,
        n_responses=n_responses,
        batch_size=batch_size,
        resume=resume,
        skill_path=skill_path,
        skill_mode=skill_mode,
        subset_mode=subset_mode,
        subset_k=subset_k,
        subset_seed=subset_seed,
    )
