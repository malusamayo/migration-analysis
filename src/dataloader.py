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

def generate_rollout_version(
    skill_version: str = "",
    skill_mode: str = "all_loaded",
) -> str:
    """
    Generate a rollout version name based on configuration parameters.

    Args:
        skill_version: Path to the skill folder (e.g., "skills/v1"), or empty string for no skills
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]

    Returns:
        Rollout version string (e.g., "v0", "v1_all", "v1_agent")

    Examples:
        >>> generate_rollout_version("", "all_loaded")
        'v0'
        >>> generate_rollout_version("skills/v1", "all_loaded")
        'v1_all'
        >>> generate_rollout_version("skills/v1", "agent_decided")
        'v1_agent'
    """
    # If no skills, always return v0
    if skill_version == "" or skill_version is None:
        return "v0"

    skill_version_name = Path(skill_version).name

    # Map skill_mode to short name
    mode_map = {
        "all_loaded": "all",
        "agent_decided": "agent",
        "monitor_decided": "monitor",
    }

    if skill_mode not in mode_map:
        raise ValueError(f"Invalid skill_mode: {skill_mode}. Must be one of {list(mode_map.keys())}")

    mode_short = mode_map[skill_mode]

    # Generate version string
    return f"{skill_version_name}_{mode_short}"

def prepare_task(
        task_id: str,
        model_name: str,
        rollout_version: str,
        prompt_name: str = "default",
        n_responses: int = 1,
        max_examples: Optional[int] = None,
        skill_version: Optional[str] = None,
        skill_mode: str = "all_loaded",
    ):
    """
    Prepare task data and create workspace directories.

    Args:
        task_id: Task identifier (e.g., "webtest", "webgen")
        model_name: Model name
        rollout_version: Rollout version identifier (required)
        prompt_name: Prompt name
        n_responses: Number of rollouts per example
        max_examples: Maximum number of examples to process
        skill_version: Path to skill folder (e.g., "skills/v1"), or None for no skills
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]

    Returns:
        Tuple of (data, task_prompt, eval_prompt)
    """
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

    # Create version subdirectory
    workspace_base = f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}"
    os.makedirs(workspace_base, exist_ok=True)

    # Create rollout_config.json in version directory
    config_path = os.path.join(workspace_base, "rollout_config.json")
    if not os.path.exists(config_path):
        from datetime import datetime
        config = {
            "task_id": task_id,
            "model_name": model_name,
            "prompt_name": prompt_name,
            "rollout_version": rollout_version,
            "skill_version": skill_version,
            "skill_mode": skill_mode,
            "timestamp": datetime.now().isoformat(),
            "max_examples": max_examples,
            "n_responses": n_responses
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created rollout config: {config_path}")

    for example_id, example in enumerate(data):
        for rollout_id in range(n_responses):
            workspace = f"{workspace_base}/example{example_id}_rollout{rollout_id}/"
            os.makedirs(workspace, exist_ok=True)
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
        rollout_version: str,
        prompt_name: str = "default",
        max_examples: Optional[int] = None,
        n_responses: int = 1,
        skill_version: Optional[str] = None,
        skill_mode: str = "all_loaded",
    ):
        """
        Initialize the data loader by loading task data and constructing workspace mappings.

        Args:
            task_id: Task identifier (e.g., "webtest", "webgen")
            model_name: Model name used for workspace directory
            rollout_version: Rollout version identifier (required)
            prompt_name: Prompt name used for workspace directory
            max_examples: Maximum number of examples to load
            n_responses: Number of rollouts per example
            skill_version: Path to skill folder (e.g., "skills/v1"), or None for no skills
            skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]
        """

        # Load task data using prepare_task
        examples, _, _ = prepare_task(
            task_id=task_id,
            model_name=model_name,
            rollout_version=rollout_version,
            prompt_name=prompt_name,
            max_examples=max_examples,
            n_responses=n_responses,
            skill_version=skill_version,
            skill_mode=skill_mode,
        )

        # Construct workspace data
        workspace_base_dir = f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}"
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
        rollout_version: str,
        prompt_name: str = "default",
        is_agentic: bool = False,
        max_examples: Optional[int] = None,
        n_responses: int = 1,
        skill_version: Optional[str] = None,
        skill_mode: str = "all_loaded",
    ):
        """
        Initialize the data loader by loading task data.

        Args:
            task_id: Task identifier (e.g., "webtest", "webgen")
            model_name: Model name to use
            rollout_version: Rollout version identifier (required)
            prompt_name: Prompt name to use
            is_agentic: Whether to use agentic execution
            max_examples: Maximum number of examples to load
            n_responses: Number of rollouts per example
            skill_version: Path to skill folder (e.g., "skills/v1"), or None for no skills
            skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]
        """

        self.task_id = task_id
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.is_agentic = is_agentic
        self.n_responses = n_responses
        self.rollout_version = rollout_version

        # Load task data using prepare_task
        examples, task_prompt, _ = prepare_task(
            task_id=task_id,
            model_name=model_name,
            rollout_version=rollout_version,
            prompt_name=prompt_name,
            max_examples=max_examples,
            n_responses=n_responses,
            skill_version=skill_version,
            skill_mode=skill_mode,
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
                    workspace = f"results/{self.task_id}/{self.model_name}_{self.prompt_name}/rollouts/{self.rollout_version}/example{example_id}_rollout{rollout_id}/"
                    collection_data.append({
                        "lm": self.lm,
                        "system_prompt_path": f"data/{self.task_id}/prompts/{self.prompt_name}.md",
                        "example": example,
                        "workspace": workspace,
                        "seed": rollout_id,
                        "task_id": self.task_id,
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
    
    def __getitem__(self, index: int) -> dict:
        """Get a single item by index."""
        return self.data[index]
