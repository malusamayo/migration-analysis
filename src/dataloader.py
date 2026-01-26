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
    subset_mode: str = "all",
    subset_k: Optional[int] = None,
    subset_seed: Optional[int] = None,
) -> str:
    """
    Generate a rollout version name based on configuration parameters.

    Args:
        skill_version: Path to the skill folder or metadata file (e.g., "skills/v1" or "skills/v1/metadata_filtered.yaml"), or empty string for no skills
        skill_mode: One of ["all_loaded", "agent_decided", "monitor_decided"]
        subset_mode: One of ["all", "top_k", "random"] - method to select skill subset
        subset_k: Number of skills to select when subset_mode is "top_k" or "random"
        subset_seed: Random seed for reproducibility when subset_mode is "random"

    Returns:
        Rollout version string (e.g., "v0", "v1_all", "v1_filtered_all", "v1_agent_top5", "v1_all_rand10s42")

    Examples:
        >>> generate_rollout_version("", "all_loaded")
        'v0'
        >>> generate_rollout_version("skills/v1", "all_loaded")
        'v1_all'
        >>> generate_rollout_version("skills/v1/metadata.yaml", "all_loaded")
        'v1_all'
        >>> generate_rollout_version("skills/v1/metadata_filtered.yaml", "all_loaded")
        'v1_filtered_all'
        >>> generate_rollout_version("skills/v1", "agent_decided")
        'v1_agent'
        >>> generate_rollout_version("skills/v1", "all_loaded", "top_k", 5)
        'v1_all_top5'
        >>> generate_rollout_version("skills/v1", "all_loaded", "random", 10, 42)
        'v1_all_rand10s42'
    """
    # If no skills, always return v0
    if skill_version == "" or skill_version is None:
        return "v0"

    skill_path = Path(skill_version)

    # Check if skill_version points to a file with pattern metadata_{id}.yaml
    metadata_id = None
    if skill_path.suffix in ['.yaml', '.yml']:
        # Extract filename
        filename = skill_path.stem  # e.g., "metadata" or "metadata_filtered"
        # Check for pattern metadata_{id}
        match = re.match(r'metadata_(.+)', filename)
        if match:
            metadata_id = match.group(1)

        # Use parent directory name as base version
        skill_version_name = skill_path.parent.parent.name
    else:
        # It's a directory path
        skill_version_name = skill_path.parent.name

    # Map skill_mode to short name
    mode_map = {
        "all_loaded": "all",
        "agent_decided": "agent",
        "monitor_decided": "monitor",
    }

    if skill_mode not in mode_map:
        raise ValueError(f"Invalid skill_mode: {skill_mode}. Must be one of {list(mode_map.keys())}")

    mode_short = mode_map[skill_mode]

    # Base version string
    if metadata_id:
        # Include metadata ID in version string (e.g., "v1_filtered_all")
        version_str = f"{skill_version_name}_{metadata_id}_{mode_short}"
    else:
        version_str = f"{skill_version_name}_{mode_short}"

    # Add subset information if not using all skills
    if subset_mode == "top_k" and subset_k is not None:
        version_str += f"_top{subset_k}"
    elif subset_mode == "random" and subset_k is not None:
        version_str += f"_rand{subset_k}"
        if subset_seed is not None:
            version_str += f"s{subset_seed}"

    return version_str

def prepare_task(
        task_id: str,
        model_name: str,
        rollout_version: str,
        prompt_name: str = "default",
        n_responses: int = 1,
        max_examples: Optional[int] = None,
        skill_version: Optional[str] = None,
        skill_mode: str = "all_loaded",
        subset_mode: str = "all",
        subset_k: Optional[int] = None,
        subset_seed: Optional[int] = None,
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
        subset_mode: One of ["all", "top_k", "random"] - method to select skill subset
        subset_k: Number of skills to select when subset_mode is "top_k" or "random"
        subset_seed: Random seed for reproducibility when subset_mode is "random"

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
            "subset_mode": subset_mode,
            "subset_k": subset_k,
            "subset_seed": subset_seed,
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
        subset_mode: str = "all",
        subset_k: Optional[int] = None,
        subset_seed: Optional[int] = None,
        resume: bool = True,
        output_path: Optional[str] = None,
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
            subset_mode: One of ["all", "top_k", "random"] - method to select skill subset
            subset_k: Number of skills to select when subset_mode is "top_k" or "random"
            subset_seed: Random seed for reproducibility when subset_mode is "random"
            resume: Whether to skip already-evaluated tasks
            output_path: Path to output file for resume checking (optional)
        """

        self.resume = resume
        self.output_path = output_path

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
            subset_mode=subset_mode,
            subset_k=subset_k,
            subset_seed=subset_seed,
        )

        # Construct workspace data
        workspace_base_dir = f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}"
        self.data = self._construct_workspace_data(workspace_base_dir, examples, n_responses)

        # Separate completed and pending tasks
        self.completed_results = []
        self.pending_data = []
        self._filter_completed_tasks()

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

    def _filter_completed_tasks(self):
        """
        Filter out already-evaluated tasks by loading from output file.
        Only applies when resume=True and output_path is provided.
        """
        if not self.resume or not self.output_path:
            self.pending_data = self.data
            self.completed_results = []
            return

        if not os.path.exists(self.output_path):
            self.pending_data = self.data
            self.completed_results = []
            return

        try:
            import yaml
            with open(self.output_path, "r") as f:
                existing_results = yaml.safe_load(f) or []

            # Build set of completed workspace directories
            completed_workspaces = set()
            for result in existing_results:
                if isinstance(result, dict) and "workspace_dir" in result:
                    completed_workspaces.add(result["workspace_dir"])

            # Split data into completed and pending based on workspace_dir
            self.completed_results = existing_results
            self.pending_data = []

            for item in self.data:
                workspace_dir = item["workspace_dir"]
                if workspace_dir not in completed_workspaces:
                    self.pending_data.append(item)

            print(f"📊 Found {len(self.completed_results)} completed evaluations, {len(self.pending_data)} pending")

        except Exception as e:
            print(f"⚠️  Could not load existing results from {self.output_path}: {e}")
            print("Starting fresh")
            self.pending_data = self.data
            self.completed_results = []

    def get_pending_args(self) -> List[dict]:
        """
        Get all pending task arguments.

        Returns:
            List of argument dictionaries for tasks that need to be evaluated
        """
        return [
            {
                "workspace_dir": item["workspace_dir"],
                "example": item["example"],
            }
            for item in self.pending_data
        ]

    def get_completed_results(self) -> List[dict]:
        """
        Get results from already-completed evaluations.

        Returns:
            List of completed results loaded from output file
        """
        return self.completed_results

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
        subset_mode: str = "all",
        subset_k: Optional[int] = None,
        subset_seed: Optional[int] = None,
        resume: bool = True,
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
            subset_mode: One of ["all", "top_k", "random"] - method to select skill subset
            subset_k: Number of skills to select when subset_mode is "top_k" or "random"
            subset_seed: Random seed for reproducibility when subset_mode is "random"
            resume: Whether to skip already-completed tasks
        """

        self.task_id = task_id
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.is_agentic = is_agentic
        self.n_responses = n_responses
        self.rollout_version = rollout_version
        self.resume = resume

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
            subset_mode=subset_mode,
            subset_k=subset_k,
            subset_seed=subset_seed,
        )
        self.lm = LM_DICT[model_name]
        self.task_prompt = task_prompt

        # Construct collection data
        self.data = self._construct_collection_data(examples)

        # Separate completed and pending tasks
        self.completed_results = []
        self.pending_data = []
        self._filter_completed_tasks()

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
                    example = example.copy()
                    example["example_id"] = example_id
                    example["rollout_id"] = rollout_id
                    collection_data.append({
                        "lm": self.lm,
                        "system_prompt_path": f"data/{self.task_id}/prompts/{self.prompt_name}.md",
                        "example": example,
                        "workspace": workspace,
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

    def _filter_completed_tasks(self):
        """
        Filter out already-completed tasks and load their results.
        Only applies to agentic mode when resume=True.
        """
        import copy

        if not self.resume or not self.is_agentic:
            self.pending_data = self.data
            self.completed_results = []
            return

        for args in self.data:
            workspace_path = Path(args["workspace"])
            existing_traces = list(workspace_path.glob("trace*.md")) if workspace_path.exists() else []

            if existing_traces:
                # Load existing result if available
                trace_json_files = list(workspace_path.glob("trace*.json"))
                if trace_json_files:
                    try:
                        with open(trace_json_files[0], 'r') as f:
                            existing_data = json.load(f)
                            result = copy.deepcopy(args["example"])
                            result["run_result"] = existing_data
                            self.completed_results.append(result)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not load existing trace from {workspace_path}: {e}")
                        self.pending_data.append(args)
                else:
                    # Has .md but no .json - re-run
                    self.pending_data.append(args)
            else:
                self.pending_data.append(args)

        print(f"📊 Found {len(self.completed_results)} completed tasks, {len(self.pending_data)} pending")

    def get_pending_args(self) -> List[dict]:
        """
        Get all pending task arguments.

        Returns:
            List of argument dictionaries for tasks that need to be run
        """
        return self.pending_data

    def get_completed_results(self) -> List[dict]:
        """
        Get results from already-completed tasks.

        Returns:
            List of completed results loaded from existing trace files
        """
        return self.completed_results

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
