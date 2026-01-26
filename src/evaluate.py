import json
import yaml
import os
import argparse
from typing import Optional
from .utils import batch_inference
from .utils import LM_DICT
from .dataloader import EvalDataLoader, generate_rollout_version

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
        eval_lm_name: Optional[str] = None,
        prompt_name: str = "default",
        max_examples: Optional[int] = None,
        n_responses: int = 1,
        batch_size: int = 16,
        resume: bool = True,
        rollout_version: str = "v0",
        subset_mode: str = "all",
        subset_k: Optional[int] = None,
        subset_seed: Optional[int] = None,
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
        batch_size: Batch size for evaluation
        resume: Whether to resume from existing results
        rollout_version: Rollout version identifier (e.g., "v0", "v1")
        subset_mode: One of ["all", "top_k", "random"] - method to select skill subset
        subset_k: Number of skills to select when subset_mode is "top_k" or "random"
        subset_seed: Random seed for reproducibility when subset_mode is "random"
    """
    config = get_config(task_id)

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
        subset_mode=subset_mode,
        subset_k=subset_k,
        subset_seed=subset_seed,
        resume=resume,
        output_path=output_path,
    )

    # Get completed and pending tasks from data loader
    results = data_loader.get_completed_results()
    args_list = data_loader.get_pending_args()

    eval_function = config["eval_function"]
    if task_id == "webtest" and not eval_lm_name:
        raise ValueError("eval_lm_name is required for webtest evaluation.")
    if eval_lm_name:
        if eval_lm_name not in LM_DICT:
            raise ValueError(f"Unknown eval_lm_name: {eval_lm_name}. Check configs/models.yaml.")
        eval_lm = LM_DICT[eval_lm_name]
        args_list = [{**args, "lm": eval_lm} for args in args_list]

    # Define callback to save partial results
    def write_partial_results(completed_results, total_count):
        # Combine already-completed results with new completions
        all_results = results + completed_results
        with open(output_path, "w") as f:
            yaml.dump(all_results, f, indent=2, allow_unicode=True)
        print(f"💾 Saved partial results ({len(all_results)}/{len(data_loader)} completed)")

    # Process all remaining data with periodic callbacks
    if args_list:
        batch_results = batch_inference(
            eval_function,
            args_list,
            use_process=config["use_process"],
            max_workers=config["max_workers"],
            on_batch_complete=write_partial_results,
            batch_size=batch_size,
        )
        results.extend(batch_results)

    # Write final results
    with open(output_path, "w") as f:
        yaml.dump(results, f, indent=2, allow_unicode=True)
    print(f"✅ Completed all {len(results)}/{len(data_loader)} evaluations")

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
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh instead of resuming from existing results")
    parser.add_argument("--eval_lm", type=str, default=None, help="LM name for evaluation feedback (required for webtest)")

    # Parameters for automatic rollout versioning
    parser.add_argument("--skill_version", type=str, default=None,
                        help="Path to skill folder (e.g., 'skills/v1'), or None for no skills")
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
    task_id = args.task_id if args.task_id is not None else config.get("task_id")
    model_name = args.model_name if args.model_name is not None else config.get("model_name")
    prompt_name = args.prompt_name if args.prompt_name is not None else config.get("prompt_name", "default")
    max_examples = args.max_examples if args.max_examples is not None else config.get("max_examples")
    n_responses = args.n_responses if args.n_responses is not None else config.get("n_responses", 1)
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 16)
    eval_lm_name = args.eval_lm if args.eval_lm is not None else config.get("eval_lm")
    skill_version = args.skill_version if args.skill_version is not None else config.get("skill_version")
    skill_mode = args.skill_mode if args.skill_mode is not None else config.get("skill_mode", "all_loaded")
    subset_mode = args.subset_mode if args.subset_mode is not None else config.get("subset_mode", "all")
    subset_k = args.subset_k if args.subset_k is not None else config.get("subset_k")
    subset_seed = args.subset_seed if args.subset_seed is not None else config.get("subset_seed")

    # Handle boolean flag specially
    resume = args.resume if args.resume else config.get("resume", True)

    # Validate required arguments
    if not task_id:
        parser.error("--task_id is required (either via command line or config file)")
    if not model_name:
        parser.error("--model_name is required (either via command line or config file)")

    # Auto-generate rollout_version from skill parameters
    rollout_version = generate_rollout_version(
        skill_version=skill_version,
        skill_mode=skill_mode,
        subset_mode=subset_mode,
        subset_k=subset_k,
        subset_seed=subset_seed,
    )
    print(f"📦 Auto-generated rollout version: {rollout_version}")

    run_task_eval(
        task_id=task_id,
        model_name=model_name,
        eval_lm_name=eval_lm_name,
        prompt_name=prompt_name,
        max_examples=max_examples,
        n_responses=n_responses,
        batch_size=batch_size,
        resume=resume,
        rollout_version=rollout_version,
        subset_mode=subset_mode,
        subset_k=subset_k,
        subset_seed=subset_seed,
    )
