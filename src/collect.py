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

from .utils import batch_inference
from .dataloader import CollectDataLoader

import os
import numpy as np
import tqdm
import copy
import time
import json
import argparse
import re
from pathlib import Path

from .runner import run_single_instance_agentic
from .task_setups import preprocess_example, get_tools, setup_servers, teardown_servers


def _print_effective_config(config: dict) -> None:
    print("Effective config:")
    print(yaml.safe_dump(config, sort_keys=False).rstrip())


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

def run_task(
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        is_agentic: bool = False,
        max_examples: Optional[int] = None,
        n_responses: int = 1,
        agent_batch_size: int = 16,
        resume: bool = True,
        rollout_version: str = "v0",
        use_docker: bool = True,
        server_image: str = "migration-analysis:latest",
        docker_network: Optional[str] = None,
        data_path: Optional[str] = None,
        start_servers: bool = False,
        server_start_timeout: int = 300,
        agent_file: Optional[str] = None,
        max_time: Optional[float] = None,
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
        agent_batch_size: Maximum number of concurrent agent/example workers
        resume: Whether to resume from existing results
        rollout_version: Rollout version identifier (default: "v0")

    Note:
        For agentic execution, volume mounts are automatically inferred from:
        - System prompt directory (mounted as /workspace/prompts:ro)
        - Workspace path (mounted as /workspace/data)
    """
    print(f"📦 Rollout version: {rollout_version}")
    _print_effective_config(
        {
            "task_id": task_id,
            "model_name": model_name,
            "prompt_name": prompt_name,
            "max_examples": max_examples,
            "n_responses": n_responses,
            "is_agentic": is_agentic,
            "agent_batch_size": agent_batch_size,
            "resume": resume,
            "rollout_version": rollout_version,
            "use_docker": use_docker,
            "server_image": server_image,
            "docker_network": docker_network,
            "data_path": data_path,
            "start_servers": start_servers,
            "server_start_timeout": server_start_timeout,
            "agent_file": agent_file,
            "max_time": max_time,
        }
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
        resume=resume,
        data_path=data_path,
    )

    output_path = f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}/run.json"
    rollout_dir = os.path.dirname(output_path)
    os.makedirs(rollout_dir, exist_ok=True)

    if agent_file is not None:
        shutil.copy(agent_file, os.path.join(rollout_dir, os.path.basename(agent_file)))

    # Get completed and pending tasks from data loader
    results = data_loader.get_completed_results()
    args_list = data_loader.get_pending_args()

    # Determine which function to use
    if is_agentic:
        run_function = run_single_instance_agentic
        use_process = True
    else:
        run_function = run_single_instance
        use_process = False
    max_workers = agent_batch_size

    if agent_file is not None:
        print(f"🤖 Using custom agent builder from {agent_file}")

    # Add docker settings to agentic args
    if is_agentic:
        for args in args_list:
            args["use_docker"] = use_docker
            args["server_image"] = server_image
            args["docker_network"] = docker_network
            args["tools"] = get_tools(task_id, args.get("workspace"))
            args["max_time"] = max_time
            if agent_file is not None:
                args["agent_file"] = os.path.abspath(agent_file)

    servers_started = setup_servers(
        task_id, args_list, start_servers=start_servers, timeout=server_start_timeout,
        docker_network=docker_network if is_agentic else None,
    )
    for args in args_list:
        args["example"] = preprocess_example(task_id, args["example"])

    # Define callback to save partial results
    def write_partial_results(completed_results, total_count):
        # Combine already-completed results with new completions
        all_results = results + completed_results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"💾 Saved partial results ({len(all_results)}/{len(data_loader)} completed)")

    try:
        # Process remaining data with periodic callbacks
        if args_list:
            batch_results = batch_inference(
                run_function,
                args_list,
                use_process=use_process,
                max_workers=max_workers,
                on_batch_complete=write_partial_results
            )
            results.extend(batch_results)
    finally:
        teardown_servers(task_id, servers_started)

    # Write final results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Completed all {len(results)}/{len(data_loader)} examples")
    print(f"📂 Results saved to {output_path}")

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
    parser.add_argument(
        "--agent_batch_size",
        type=int,
        default=None,
        help="Maximum number of concurrent agent/example workers.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=None,
        help="Start fresh instead of resuming from existing results.",
    )
    parser.add_argument("--agent_file", type=str, default=None,
                        help="Path to a Python file defining build_agent(base_dir, llm) -> Agent")
    parser.add_argument("--rollout_version", type=str, default=None,
                        help="Rollout version identifier (default: 'v0')")
    parser.add_argument(
        "--max_time",
        type=float,
        default=None,
        help="Maximum runtime in seconds for each agentic task conversation.",
    )

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
    agent_batch_size = (
        args.agent_batch_size
        if args.agent_batch_size is not None
        else config.get("agent_batch_size", 16)
    )
    rollout_version = args.rollout_version if args.rollout_version is not None else config.get("rollout_version", "v0")

    # Handle boolean flags specially
    is_agentic = args.is_agentic or config.get("is_agentic", False)
    resume = args.resume if args.resume is not None else config.get("resume", True)
    use_docker = config.get("use_docker", False)
    server_image = config.get("server_image", "migration-analysis:latest")
    docker_network = config.get("docker_network", None)
    data_path = config.get("data_path")
    start_servers = config.get("start_servers", False)
    server_start_timeout = config.get("server_start_timeout", 300)
    agent_file = args.agent_file if args.agent_file is not None else config.get("agent_file")
    max_time = args.max_time if args.max_time is not None else config.get("max_time")
    if agent_file is not None:
        rollout_version += f"_{Path(agent_file).stem}"

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
        agent_batch_size=agent_batch_size,
        resume=resume,
        rollout_version=rollout_version,
        use_docker=use_docker,
        server_image=server_image,
        docker_network=docker_network,
        data_path=data_path,
        start_servers=start_servers,
        server_start_timeout=server_start_timeout,
        agent_file=agent_file,
        max_time=max_time,
    )
