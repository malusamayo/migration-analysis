import json
import os
import argparse
from typing import Optional
from .utils import batch_inference
from .dataloader import EvalDataLoader, load_and_validate_results, generate_rollout_version

def get_config(task_id: str):
    if task_id == "webgen":
        from .task_evals.webgen import run_single_instance_eval
        return {
            "eval_function": run_single_instance_eval,
            "use_process": True,
            "max_workers": 4,
        }
    elif task_id == "webtest":
        from .task_evals.webtest import run_single_instance_eval
        return {
            "eval_function": run_single_instance_eval,
            "use_process": True,
            "max_workers": 32,
        }
    else:
        return {
            "eval_function": run_single_instance_eval,
            "use_process": False,
            "max_workers": 32,
        }


def run_task_eval(
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        max_examples: Optional[int] = None,
        n_responses: int = 1,
        batch_size: int = 16,
        resume: bool = True,
        rollout_version: str = "v0",
    ):
    """
    Run evaluation on a task.

    Args:
        task_id: Task identifier (e.g., "webtest", "webgen")
        model_name: Model name used for workspace directory
        prompt_name: Prompt name used for workspace directory
        max_examples: Maximum number of examples to evaluate
        n_responses: Number of rollouts per example
        batch_size: Batch size for evaluation
        resume: Whether to resume from existing results
        rollout_version: Rollout version identifier (e.g., "v0", "v1")
    """
    config = get_config(task_id)

    # Initialize data loader (loads data via prepare_task and constructs workspaces)
    data_loader = EvalDataLoader(
        task_id=task_id,
        model_name=model_name,
        prompt_name=prompt_name,
        max_examples=max_examples,
        n_responses=n_responses,
        rollout_version=rollout_version,
    )

    workspace_base_dir = f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}"
    output_path = os.path.join(workspace_base_dir, "eval_results.json")

    # Load existing results if resuming
    if resume:
        results, start_idx = load_and_validate_results(output_path, data_loader)
    else:
        results, start_idx = [], 0

    eval_function = config["eval_function"]

    # Process remaining data in batches
    for i in range(start_idx, len(data_loader), batch_size):
        # Get batch arguments from data loader
        args_list = data_loader.get_batch_args(
            batch_start=i,
            batch_size=batch_size
        )

        batch_results = batch_inference(
            eval_function,
            args_list,
            use_process=config["use_process"],
            max_workers=config["max_workers"],
        )
        results.extend(batch_results)

        # Write partial results after each batch
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved partial results ({len(results)}/{len(data_loader)} completed)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task evaluation")
    parser.add_argument("--task_id", type=str, required=True, help="Task identifier (e.g., webtest, webgen)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name used for workspace directory")
    parser.add_argument("--prompt_name", type=str, default="default", help="Prompt name used for workspace directory")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to evaluate")
    parser.add_argument("--n_responses", type=int, default=1, help="Number of rollouts per example")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh instead of resuming from existing results")

    # New parameters for automatic rollout versioning
    parser.add_argument("--skill_version", type=str, default=None,
                        help="Path to skill folder (e.g., 'skills/v1'), or None for no skills")
    parser.add_argument("--skill_mode", type=str, default="all_loaded",
                        choices=["all_loaded", "agent_decided", "monitor_decided"],
                        help="Skill mode: all_loaded, agent_decided, or monitor_decided")

    args = parser.parse_args()

    # Auto-generate rollout_version from skill parameters
    rollout_version = generate_rollout_version(
        skill_version=args.skill_version,
        skill_mode=args.skill_mode,
    )
    print(f"📦 Auto-generated rollout version: {rollout_version}")

    run_task_eval(
        task_id=args.task_id,
        model_name=args.model_name,
        prompt_name=args.prompt_name,
        max_examples=args.max_examples,
        n_responses=args.n_responses,
        batch_size=args.batch_size,
        resume=args.resume,
        rollout_version=rollout_version,
    )