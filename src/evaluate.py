import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Optional

import yaml

from .utils import batch_inference, LM_DICT
from .optimize.common import LiteralBlockDumper
from .dataloader import EvalDataLoader
from .task_setups import get_eval_config, requires_eval_lm


def _print_effective_config(config: dict) -> None:
    print("Effective config:")
    print(yaml.safe_dump(config, sort_keys=False).rstrip())


def run_task_eval(
        task_id: str,
        model_name: str,
        eval_lm_name: Optional[str] = None,
        prompt_name: str = "default",
        max_examples: Optional[int] = None,
        n_responses: int = 1,
        eval_batch_size: Optional[int] = None,
        resume: bool = True,
        rollout_version: str = "v0",
        data_path: Optional[str] = None,
        server_image: Optional[str] = None,
    ):
    """
    Run evaluation on a task.

    Args:
        task_id: Task identifier (e.g., "webtest", "webgen")
        model_name: Model name used for workspace directory
        eval_lm_name: LM name for evaluation feedback (required for webtest)
        prompt_name: Prompt name used for workspace directory
        max_examples: Maximum number of examples to evaluate
        n_responses: Number of rollouts per example
        eval_batch_size: Maximum number of concurrent evaluation workers
        resume: Whether to resume from existing results
        rollout_version: Rollout version identifier (e.g., "v0", "v1")
        server_image: Docker image to use for evaluation (if applicable)
    """
    config = get_eval_config(task_id)
    max_workers = eval_batch_size if eval_batch_size is not None else config["max_workers"]
    _print_effective_config(
        {
            "task_id": task_id,
            "model_name": model_name,
            "prompt_name": prompt_name,
            "max_examples": max_examples,
            "n_responses": n_responses,
            "eval_lm": eval_lm_name,
            "eval_batch_size": max_workers,
            "eval_use_process": config["use_process"],
            "resume": resume,
            "rollout_version": rollout_version,
            "data_path": data_path,
            "server_image": server_image,
        }
    )

    workspace_base_dir = f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}"
    output_path = os.path.join(workspace_base_dir, "eval_results.yaml")

    # Initialize data loader (loads data via prepare_task and constructs workspaces)
    data_loader = EvalDataLoader(
        task_id=task_id,
        model_name=model_name,
        prompt_name=prompt_name,
        max_examples=max_examples,
        n_responses=n_responses,
        rollout_version=rollout_version,
        resume=resume,
        output_path=output_path,
        data_path=data_path,
    )

    # Get completed and pending tasks from data loader
    results = data_loader.get_completed_results()
    args_list = data_loader.get_pending_args()

    eval_function = config["eval_function"]
    if requires_eval_lm(task_id) and not eval_lm_name:
        raise ValueError(f"eval_lm_name is required for {task_id} evaluation.")
    if eval_lm_name:
        if eval_lm_name not in LM_DICT:
            raise ValueError(f"Unknown eval_lm_name: {eval_lm_name}. Check configs/models.yaml.")
        eval_lm = LM_DICT[eval_lm_name]
    else:
        eval_lm = None
    args_list = [{**args, "lm": eval_lm} for args in args_list]

    eval_signature = inspect.signature(eval_function)
    if server_image and "server_image" in eval_signature.parameters:
        args_list = [{**args, "server_image": server_image} for args in args_list]

    # Define callback to save partial results
    def write_partial_results(completed_results, total_count):
        # Combine already-completed results with new completions
        all_results = results + completed_results
        with open(output_path, "w") as f:
            yaml.dump(all_results, f, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, allow_unicode=True)
        print(f"💾 Saved partial results ({len(all_results)}/{len(data_loader)} completed)")

    # Process all remaining data with periodic callbacks
    if args_list:
        batch_results = batch_inference(
            eval_function,
            args_list,
            use_process=config["use_process"],
            max_workers=max_workers,
            on_batch_complete=write_partial_results,
        )
        results.extend(batch_results)

    # Write final results
    with open(output_path, "w") as f:
        yaml.dump(results, f, Dumper=LiteralBlockDumper, indent=2, sort_keys=False, allow_unicode=True)
    print(f"✅ Completed all {len(results)}/{len(data_loader)} evaluations")
    print(f"Final results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task evaluation")

    # Config file support
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file with default arguments")

    parser.add_argument("--task_id", type=str, help="Task identifier (e.g., webtest, webgen)")
    parser.add_argument("--model_name", type=str, help="Model name used for workspace directory")
    parser.add_argument("--prompt_name", type=str, default=None, help="Prompt name used for workspace directory")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to evaluate")
    parser.add_argument("--n_responses", type=int, default=None, help="Number of rollouts per example")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Maximum number of concurrent evaluation workers.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=None,
        help="Start fresh instead of resuming from existing results",
    )
    parser.add_argument("--eval_lm", type=str, default=None, help="LM name for evaluation feedback (required for webtest)")
    parser.add_argument("--agent_file", type=str, default=None,
                        help="Path to a Python file defining build_agent(base_dir, llm) -> Agent")
    parser.add_argument("--rollout_version", type=str, default=None,
                        help="Rollout version identifier (default: 'v0')")
    
    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"📋 Loaded config from {args.config}")

    # Merge config file with command line arguments (command line takes precedence)
    task_id = args.task_id if args.task_id is not None else config.get("task_id")
    model_name = args.model_name if args.model_name is not None else config.get("model_name")
    prompt_name = args.prompt_name if args.prompt_name is not None else config.get("prompt_name", "default")
    max_examples = args.max_examples if args.max_examples is not None else config.get("max_examples")
    n_responses = args.n_responses if args.n_responses is not None else config.get("n_responses", 1)
    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else config.get("eval_batch_size")
    )
    eval_lm_name = args.eval_lm if args.eval_lm is not None else config.get("eval_lm")
    rollout_version = args.rollout_version if args.rollout_version is not None else config.get("rollout_version", "v0")
    agent_file = args.agent_file if args.agent_file is not None else config.get("agent_file")
    if agent_file is not None:
        rollout_version += f"_{Path(agent_file).stem}"


    # Handle boolean flag specially
    resume = args.resume if args.resume is not None else config.get("resume", True)
    data_path = config.get("data_path")
    server_image = config.get("server_image")

    # Validate required arguments
    if not task_id:
        parser.error("--task_id is required (either via command line or config file)")
    if not model_name:
        parser.error("--model_name is required (either via command line or config file)")

    run_task_eval(
        task_id=task_id,
        model_name=model_name,
        eval_lm_name=eval_lm_name,
        prompt_name=prompt_name,
        max_examples=max_examples,
        n_responses=n_responses,
        eval_batch_size=eval_batch_size,
        resume=resume,
        rollout_version=rollout_version,
        data_path=data_path,
        server_image=server_image,
    )
