"""
Data loader classes for evaluation and collection tasks.
"""
import json
import os
import pandas as pd
from typing import List, Optional
from pathlib import Path
import re
from .utils import LM_DICT

def prepare_task(
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        n_responses: int = 1,
        max_examples: Optional[int] = None,
    ):
    assert task_id in ["webgen", "webtest"], "Unsupported task_id"

    data_path = f"data/{task_id}/{task_id}.csv"
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} examples from {data_path}, keeping max_examples={max_examples}")
    data = data.to_dict(orient="records")[:max_examples]

    task_prompt_path = f"data/{task_id}/prompts/{prompt_name}.md"
    with open(task_prompt_path, "r") as f:
        task_prompt = f.read()
    
    eval_prompt_path = f"data/{task_id}/eval.md"
    with open(eval_prompt_path, "r") as f:
        eval_prompt = f.read()

    for example_id, example in enumerate(data):
        for rollout_id in range(n_responses):
            workspace = f"results/{task_id}/{model_name}_{prompt_name}_agentic_workspace/example{example_id}_rollout{rollout_id}/"
            os.makedirs(workspace, exist_ok=True)
            if task_id == "webtest":
                with open(os.path.join(workspace, "index.html"), "w") as f:
                    f.write(example["html_content"])
    return data, task_prompt, eval_prompt

def load_and_validate_results(
        output_path: str,
        data_loader: 'BaseDataLoader'
    ) -> tuple[List[dict], int]:
    """
    Load existing results and validate they match the data loader's configuration.

    Args:
        output_path: Path to the results file
        data_loader: DataLoader instance to validate against

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

    # Each item gets one result
    start_idx = len(results)
    print(f"🔄 Resuming from {start_idx}/{len(data_loader)} completed items")
    return results, start_idx


class BaseDataLoader:
    """Base class for data loaders."""

    def __len__(self):
        """Return total number of items."""
        return len(self.data)


class EvalDataLoader(BaseDataLoader):
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


class CollectDataLoader(BaseDataLoader):
    """
    Data loader for collection that provides arguments for batch inference.

    Each collection task requires:
    - lm: Language model
    - example: Dict containing prompt, metadata, etc.
    - seed/rollout_id: For generation
    - Additional task-specific parameters
    """
    def __init__(
        self,
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        is_agentic: bool = False,
        max_examples: Optional[int] = None,
        n_responses: int = 1,
    ):
        """
        Initialize the data loader by loading task data.

        Args:
            task_id: Task identifier (e.g., "webtest", "webgen")
            model_name: Model name to use
            prompt_name: Prompt name to use
            is_agentic: Whether to use agentic execution
            max_examples: Maximum number of examples to load
            n_responses: Number of rollouts per example
        """

        self.task_id = task_id
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.is_agentic = is_agentic
        self.n_responses = n_responses

        # Load task data using prepare_task
        examples, task_prompt, _ = prepare_task(
            task_id=task_id,
            model_name=model_name,
            prompt_name=prompt_name,
            max_examples=max_examples,
            n_responses=n_responses,
        )

        self.lm = LM_DICT[model_name]
        self.task_prompt = task_prompt

        # Construct collection data
        self.data = self._construct_collection_data(examples)

    def _construct_collection_data(self, examples: List[dict]) -> List[dict]:
        """
        Construct collection data for all examples and rollouts.

        Args:
            examples: List of example dicts from prepare_task

        Returns:
            List of collection items with all necessary parameters
        """
        collection_data = []

        if self.is_agentic:
            # For agentic mode: each (example, rollout) pair becomes one item
            for example_id, example in enumerate(examples):
                for rollout_id in range(self.n_responses):
                    workspace = f"results/{self.task_id}/{self.model_name}_{self.prompt_name}_agentic_workspace/example{example_id}_rollout{rollout_id}/"
                    collection_data.append({
                        "lm": self.lm,
                        "system_prompt_path": f"data/{self.task_id}/prompts/{self.prompt_name}.md",
                        "example": example,
                        "workspace": workspace,
                    })
        else:
            # For non-agentic mode: each (example, seed) pair becomes one item
            from .task_evals.webgen import validate_webpage, generate_retry_function_webpage

            for example in examples:
                for seed in range(self.n_responses):
                    collection_data.append({
                        "lm": self.lm,
                        "system_prompt": self.task_prompt,
                        "example": example,
                        "seed": seed,
                        "validation_fn": validate_webpage,
                        "retry_gen_fn": generate_retry_function_webpage,
                    })

        print(f"Constructed {len(collection_data)} collection items")
        return collection_data

    def get_batch_args(self, batch_start: int, batch_size: int) -> List[dict]:
        """
        Get arguments for a specific batch.

        Args:
            batch_start: Starting index in the original data
            batch_size: Number of items per batch

        Returns:
            List of argument dictionaries for batch_inference
        """
        batch_end = min(batch_start + batch_size, len(self.data))
        return self.data[batch_start:batch_end]
