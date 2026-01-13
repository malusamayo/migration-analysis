import json
import os
import argparse
from typing import List, Any, Optional
from .utils import batch_inference
from .collect import prepare_task

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
            "max_workers": 4,
        }
    else:
        return {
            "eval_function": run_single_instance_eval,
            "use_process": False,
            "max_workers": 32,
        }


def load_and_validate_results(
        output_path: str,
        data_loader: 'EvalDataLoader'
    ) -> tuple[List[dict], int]:
    """
    Load existing results and validate they match the data loader's configuration.

    Args:
        output_path: Path to the results file
        data_loader: EvalDataLoader instance to validate against

    Returns:
        tuple: (results, start_idx) where results is the list of existing results
               and start_idx is the index to resume from (0 if starting fresh)
    """
    if not os.path.exists(output_path):
        return [], 0

    try:
        with open(output_path, "r") as f:
            results = json.load(f)
    except json.JSONDecodeError:
        print("⚠️  Could not parse existing results file, starting fresh")
        return [], 0

    # Calculate expected total results
    expected_total = len(data_loader)

    # Validate that existing results don't exceed expected
    if len(results) > expected_total:
        print(f"⚠️  Existing results ({len(results)}) exceed expected total ({expected_total}), starting fresh")
        return [], 0

    # Each workspace gets one result
    start_idx = len(results)
    print(f"🔄 Resuming from {start_idx}/{len(data_loader.data)} completed workspaces")
    return results, start_idx

class EvalDataLoader:
    """
    Data loader for evaluation that provides arguments for batch inference.

    Each evaluation task requires:
    - workspace_dir: Path to the workspace directory
    - example: Dict containing prompt, metadata, etc.
    """
    def __init__(
        self,
        task_id: str,
        model_name: str,
        prompt_name: str,
        max_examples: Optional[int] = None,
        n_responses: int = 1,
    ):
        """
        Initialize the data loader by loading task data and constructing workspace mappings.

        Args:
            task_id: Task identifier (e.g., "webtest", "webgen")
            model_name: Model name used for workspace directory
            prompt_name: Prompt name used for workspace directory
            max_examples: Maximum number of examples to load
            n_responses: Number of rollouts per example
        """
        # Load task data using prepare_task
        examples, _, _ = prepare_task(
            task_id=task_id,
            model_name=model_name,
            prompt_name=prompt_name,
            max_examples=max_examples,
            n_responses=n_responses,
        )

        # Construct workspace data
        workspace_base_dir = f"results/{task_id}/{model_name}_{prompt_name}_agentic_workspace"
        self.data = self._construct_workspace_data(workspace_base_dir, examples, n_responses)

    def _construct_workspace_data(
        self,
        workspace_base_dir: str,
        examples: List[dict],
        n_responses: int
    ) -> List[dict]:
        """
        Construct workspace data by matching workspaces with examples.

        Args:
            workspace_base_dir: Base directory containing workspaces
            examples: List of example dicts from prepare_task
            n_responses: Number of rollouts per example

        Returns:
            List of workspace items with matched example data
        """
        from pathlib import Path
        import re

        base_path = Path(workspace_base_dir)
        if not base_path.exists():
            raise ValueError(f"Workspace base directory does not exist: {workspace_base_dir}")

        # Create mapping: example0 -> examples[0], example1 -> examples[1], etc.
        example_data_map = {f"example{idx}": example for idx, example in enumerate(examples)}
        print(f"Loaded {len(example_data_map)} examples")

        # Find all workspace directories matching pattern: example<N>_rollout<M>
        workspace_pattern = re.compile(r'^(example\d+)_(rollout\d+)$')

        workspace_data = []
        for item in sorted(base_path.iterdir()):
            if item.is_dir():
                match = workspace_pattern.match(item.name)
                if match:
                    example_id, rollout_id = match.groups()

                    # Skip if no matching example data
                    if example_id not in example_data_map:
                        print(f"⚠️  No example data found for {example_id}, skipping")
                        continue

                    workspace_data.append({
                        "workspace_dir": str(item),
                        "example_id": example_id,
                        "rollout_id": rollout_id,
                        "example": example_data_map[example_id],
                    })

        print(f"Found {len(workspace_data)} workspaces in {workspace_base_dir}")
        return workspace_data

    def get_batch_args(self, batch_start: int, batch_size: int) -> List[dict]:
        """
        Get arguments for a specific batch.

        Args:
            batch_start: Starting index in the original data
            batch_size: Number of examples per batch

        Returns:
            List of argument dictionaries for batch_inference
        """
        batch_end = min(batch_start + batch_size, len(self.data))
        batch_data = self.data[batch_start:batch_end]

        return [
            {
                "workspace_dir": item["workspace_dir"],
                "example": item["example"],
            }
            for item in batch_data
        ]

    def __len__(self):
        """Return total number of evaluation tasks."""
        return len(self.data)

def run_task_eval(
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        max_examples: Optional[int] = None,
        n_responses: int = 1,
        batch_size: int = 16,
        resume: bool = True,
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
    """
    config = get_config(task_id)

    # Initialize data loader (loads data via prepare_task and constructs workspaces)
    data_loader = EvalDataLoader(
        task_id=task_id,
        model_name=model_name,
        prompt_name=prompt_name,
        max_examples=max_examples,
        n_responses=n_responses,
    )

    workspace_base_dir = f"results/{task_id}/{model_name}_{prompt_name}_agentic_workspace"
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
    args = parser.parse_args()

    run_task_eval(
        task_id=args.task_id,
        model_name=args.model_name,
        prompt_name=args.prompt_name,
        max_examples=args.max_examples,
        n_responses=args.n_responses,
        batch_size=args.batch_size,
        resume=args.resume,
    )