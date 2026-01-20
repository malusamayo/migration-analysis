"""
Module for comparing agent trajectories using LLM analysis.

This module provides DSPy-based components and functions for comparing
multiple trajectory rollouts and analyzing behavioral differences.
"""

import json
import dspy
from pathlib import Path
from typing import List, Dict, Any

from .trajectory_loader import (
    load_trajectory_trace,
    load_and_convert_trajectories,
    combine_trajectories_markdown,
    extract_task_description
)
from ..utils import LM_DICT, batch_inference


class TrajectoryComparison(dspy.Signature):
    """Compare multiple agent trajectories and provide analysis of behavioral differences.
 For each difference, specify: 1) Which trajectories exhibit the behavior, 2) What the behavioral difference is (e.g., different tool usage, reasoning approach, error handling), 3) Impact on task completion.
 """

    trajectories = dspy.InputField(
        desc="Markdown-formatted trajectories from different rollouts, separated by '---TRAJECTORY---'"
    )
    task_description = dspy.InputField(
        desc="The original task/request that the agent was trying to complete"
    )
    comparison = dspy.OutputField(
        desc="A list of behavioral differences between trajectories."
    )

class CompareTrajectories(dspy.Module):
    """Module for comparing agent trajectories."""

    def __init__(self):
        super().__init__()
        self.compare = dspy.ChainOfThought(TrajectoryComparison)

    def forward(self, trajectories: str, task_description: str, config):
        """Compare trajectories and return analysis."""
        return self.compare(
            trajectories=trajectories,
            task_description=task_description,
            config=config
        )

class CrossModelTrajectorySetComparison(dspy.Signature):
    """Compare sets of trajectories from two different models on the same task.

    Analyze behavioral patterns, common strategies, and systematic differences between the two model sets.
    Focus on: 1) Consistent behavioral differences across rollouts, 2) Success/failure patterns,
    3) Different problem-solving approaches, 4) Tool usage patterns, 5) Error handling strategies.
    """

    trajectories_set_a = dspy.InputField(
        desc="Markdown-formatted trajectories from first model across multiple rollouts, separated by '---TRAJECTORY---'"
    )
    trajectories_set_b = dspy.InputField(
        desc="Markdown-formatted trajectories from second model across multiple rollouts, separated by '---TRAJECTORY---'"
    )
    task_description = dspy.InputField(
        desc="The original task/request that both models attempted"
    )
    comparison = dspy.OutputField(
        desc="Systematic analysis of differences between the two model sets, including success rates, common patterns, and behavioral differences"
    )


class CompareModelTrajectorysets(dspy.Module):
    """Module for comparing trajectory sets from two different models."""

    def __init__(self):
        super().__init__()
        self.compare = dspy.ChainOfThought(CrossModelTrajectorySetComparison)

    def forward(
        self,
        trajectories_set_a: str,
        trajectories_set_b: str,
        task_description: str,
    ):
        """Compare trajectory sets from two models and return analysis."""
        return self.compare(
            trajectories_set_a=trajectories_set_a,
            trajectories_set_b=trajectories_set_b,
            task_description=task_description,
        )


def compare_rollout_trajectories(
    trace_paths: List[Path],
    model_name: str = "gemini-2.5-flash",
    output_path: Path = None,
    random_seed: int = 0,
    task_id: str = None,
) -> Dict[str, Any]:
    """
    Compare multiple trajectory rollouts using LLM analysis.

    Args:
        trace_paths: List of paths to trace JSON files
        model_name: Name of the language model to use for comparison
        output_path: Optional path to save comparison results
        random_seed: Random seed for reproducibility in LLM calls
        task_id: Optional task identifier (e.g., "webtest", "webgen")

    Returns:
        Dictionary containing comparison results
    """
    # Load all trajectories
    print(f"Loading {len(trace_paths)} trajectory traces...")
    trajectories_data = load_and_convert_trajectories(trace_paths)

    # Extract task description from first trajectory
    task_description = extract_task_description(trajectories_data[0]["trace_data"], task_id=task_id)

    # Combine trajectories with separators
    trajectories_text = combine_trajectories_markdown(trajectories_data)

    # Get LLM for comparison
    lm = LM_DICT[model_name]

    print(f"Comparing trajectories using {model_name} (seed={random_seed})...")
    with dspy.context(lm=lm):
        comparator = CompareTrajectories()
        result = comparator(
            trajectories=trajectories_text,
            task_description=task_description,
            config={"rollout_id": random_seed},
        )

    # Prepare output
    comparison_result = {
        "model": model_name,
        "num_trajectories": len(trace_paths),
        "trajectory_files": [str(p) for p in trace_paths],
        "task_description": task_description,
        "comparison_analysis": result.comparison,
        "trajectories_markdown": trajectories_text
    }

    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        if output_path.suffix == ".json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
        # Save as Markdown
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Trajectory Comparison Report\n\n")
                f.write(f"**Model Used:** {model_name}\n")
                f.write(f"**Number of Trajectories:** {len(trace_paths)}\n\n")
                f.write(f"## Task Description\n\n{task_description}\n\n")
                f.write(f"## Comparison Analysis\n\n{result.comparison}\n\n")
                f.write(f"---\n\n## Detailed Trajectories\n\n{trajectories_text}\n")

        print(f"Comparison saved to: {output_path}")

    return comparison_result


def _compare_single_example(
    example_id: int,
    base_dir: Path,
    output_dir: Path,
    comparison_model: str,
    random_seed: int = 0,
    task_id: str = None,
) -> Dict[str, Any]:
    """
    Helper function to compare trajectories for a single example.
    Designed to be used with batch_inference for parallel processing.

    Args:
        example_id: Example identifier
        base_dir: Base directory containing workspace directories
        output_dir: Directory to save comparison results
        comparison_model: Model to use for comparison
        random_seed: Random seed for reproducibility in LLM calls
        task_id: Optional task identifier (e.g., "webtest", "webgen")

    Returns:
        Dictionary containing comparison result or error
    """
    # Find all workspace directories for this example
    workspace_pattern = f"example{example_id}_rollout*"
    workspace_dirs = sorted(base_dir.glob(workspace_pattern))

    if not workspace_dirs:
        return {
            "example_id": example_id,
            "error": "No workspace directories found"
        }

    # Collect all trace files
    trace_files = []
    for workspace in workspace_dirs:
        traces = list(workspace.glob("trace_*.json"))
        trace_files.extend(traces)

    if len(trace_files) < 2:
        return {
            "example_id": example_id,
            "error": f"Found {len(trace_files)} trace(s), need at least 2 for comparison"
        }

    output_path = output_dir / f"example{example_id}_comparison.md"

    try:
        # Load trace data to extract task description

        first_trace_data = load_trajectory_trace(trace_files[0])
        task_description = extract_task_description(first_trace_data, task_id=task_id)

        # Get rollout IDs from workspace directories
        rollout_ids = []
        for trace_file in trace_files:
            # Extract rollout ID from path (e.g., example0_rollout2/trace_xyz.json -> "rollout2")
            workspace_name = trace_file.parent.name
            if "_rollout" in workspace_name:
                rollout_id = workspace_name.split("_rollout")[1]
                rollout_ids.append(f"rollout{rollout_id}")

        result = compare_rollout_trajectories(
            trace_paths=trace_files,
            model_name=comparison_model,
            output_path=output_path,
            random_seed=random_seed,
            task_id=task_id,
        )

        return {
            "example_id": example_id,
            "num_trajectories": len(trace_files),
            "comparison_model": comparison_model,
            "task_description": task_description,
            "comparison": result["comparison_analysis"],  # Extract comparison text from result dict
            "output_path": str(output_path),
            "rollout_ids": rollout_ids
        }
    except Exception as e:
        return {
            "example_id": example_id,
            "error": str(e)
        }

def compare_examples_batch(
    task_id: str,
    model_name: str,
    prompt_name: str = "default",
    num_examples: int = 10,
    comparison_model: str = "gemini-2.5-flash",
    output_dir: Path = None,
    agentic: bool = True,
    max_workers: int = 8,
    rollout_version: str = "v0",
    random_seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Compare trajectories for multiple examples independently using parallel batch inference.

    Args:
        task_id: Task identifier (e.g., "webgen", "webtest")
        model_name: Model used for generation
        prompt_name: Prompt template name (default: "default")
        num_examples: Number of examples to compare (default: 10)
        comparison_model: Model to use for comparison (default: "gemini-2.5-flash")
        output_dir: Directory to save comparison results (default: results/{task_id}/{model_name}_{prompt_name}/comparisons)
        agentic: Whether the runs used agentic execution (default: True)
        max_workers: Maximum number of parallel workers (default: 8)
        rollout_version: Rollout version identifier (e.g., "v0", "v1")
        random_seed: Random seed for reproducibility in LLM calls (default: 0)

    Returns:
        List of comparison results for each example
    """
    # Determine base directory based on execution mode
    if agentic:
        base_dir = Path(f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}")
    else:
        base_dir = Path(f"results/{task_id}/{model_name}_{prompt_name}/rollouts/{rollout_version}")

    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    if output_dir is None:
        output_dir = Path(f"results/{task_id}/{model_name}_{prompt_name}/comparisons/{rollout_version}_seed{random_seed}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Starting batch comparison for {num_examples} examples using {max_workers} workers (seed={random_seed})")
    print(f"{'='*80}\n")

    # Prepare arguments for batch inference
    args_list = [
        {
            "example_id": example_id,
            "base_dir": base_dir,
            "output_dir": output_dir,
            "comparison_model": comparison_model,
            "random_seed": random_seed,
            "task_id": task_id,
        }
        for example_id in range(num_examples)
    ]

    # Run comparisons in parallel using batch inference
    results = batch_inference(
        program=_compare_single_example,
        args_list=args_list,
        use_process=False,  # Use threads since we're doing I/O and API calls
        max_workers=max_workers
    )

    # Print summary for each result
    successful = 0
    failed = 0
    for result in results:
        example_id = result["example_id"]
        if "error" in result:
            print(f"⚠️  Example {example_id}: {result['error']}")
            failed += 1
        else:
            print(f"✅ Example {example_id}: Compared {result['num_trajectories']} trajectories -> {result['output_path']}")
            successful += 1

    # Save summary of all comparisons
    summary_path = output_dir / "batch_comparison_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "task_id": task_id,
            "model_name": model_name,
            "prompt_name": prompt_name,
            "comparison_model": comparison_model,
            "num_examples_processed": len(results),
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Batch comparison completed!")
    print(f"Total examples: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed/Skipped: {failed}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}")

    return results


def _compare_single_example_trajectory_sets(
    path_a: str,
    path_b: str,
    example_id: int,
    comparison_model: str = "gemini-2.5-flash",
    output_path: Path = None,
    max_rollouts: int = None,
    label_a: str = None,
    label_b: str = None,
    task_id: str = None,
) -> Dict[str, Any]:
    """
    Compare trajectory rollout sets from two arbitrary paths.

    This is a generic function that can compare any two sets of rollouts,
    whether from different models, prompts, skill versions, etc.

    Args:
        path_a: Path to first rollout directory (e.g., "results/webtest/model_default/rollouts/v0")
        path_b: Path to second rollout directory (e.g., "results/webtest/model_default/rollouts/v1")
        example_id: Example identifier (e.g., 0 for example0)
        comparison_model: Model to use for comparison analysis (default: "gemini-2.5-flash")
        output_path: Optional path to save comparison results
        max_rollouts: Maximum number of rollouts to include per set (default: all)
        label_a: Optional label for first set (default: extracted from path)
        label_b: Optional label for second set (default: extracted from path)
        task_id: Optional task identifier (e.g., "webtest", "webgen")

    Returns:
        Dictionary containing comparison results with trajectory sets

    Example:
        >>> result = compare_trajectory_sets(
        ...     path_a="results/webtest/qwen3-coder-30b-a3b_default/rollouts/v0",
        ...     path_b="results/webtest/qwen3-coder-30b-a3b_default/rollouts/v1",
        ...     example_id=0,
        ...     output_path=Path("results/webtest/qwen3-coder-30b-a3b_default/comparisons/v0_vs_v1.md")
        ... )
    """
    from pathlib import Path

    path_a = Path(path_a)
    path_b = Path(path_b)

    # Extract labels from paths if not provided
    if label_a is None:
        label_a = str(path_a)
    if label_b is None:
        label_b = str(path_b)

    def load_trajectories_from_path(path: Path, label: str) -> List[Dict[str, Any]]:
        """Load all rollout trajectories from a given path."""
        if not path.exists():
            print(f"⚠️  Warning: Directory not found: {path}")
            return []

        # Find all rollout directories for this example
        rollout_dirs = sorted(path.glob(f"example{example_id}_rollout*"))

        if max_rollouts:
            rollout_dirs = rollout_dirs[:max_rollouts]

        trajectories = []
        for rollout_dir in rollout_dirs:
            # Find trace file in the workspace
            traces = list(rollout_dir.glob("trace_*.json"))

            if not traces:
                print(f"⚠️  Warning: No trace file found in {rollout_dir}")
                continue

            if len(traces) > 1:
                print(f"⚠️  Warning: Multiple trace files found in {rollout_dir}, using first one")

            # Extract rollout_id from directory name
            rollout_id = rollout_dir.name.split("_rollout")[1]

            trajectories.append({
                "trace_path": traces[0],
                "rollout_id": rollout_id,
                "rollout_dir": rollout_dir,
            })

        return trajectories

    print(f"\n{'='*80}")
    print(f"Comparing trajectory sets on example{example_id}")
    print(f"Set A: {label_a}")
    print(f"Set B: {label_b}")
    print(f"{'='*80}\n")

    # Load trajectories from both paths
    print(f"Loading trajectories from {label_a}...")
    trajectories_a = load_trajectories_from_path(path_a, label_a)
    print(f"  Found {len(trajectories_a)} rollout(s)")

    print(f"Loading trajectories from {label_b}...")
    trajectories_b = load_trajectories_from_path(path_b, label_b)
    print(f"  Found {len(trajectories_b)} rollout(s)")

    if not trajectories_a:
        return {
            "example_id": example_id,
            "error": f"No trajectories found in {path_a}"
        }
    if not trajectories_b:
        return {
            "example_id": example_id,
            "error": f"No trajectories found in {path_b}"
        }

    try:
        def convert_and_combine_trajectories(
            trajectory_list: List[Dict[str, Any]],
            label: str
        ) -> tuple[List[Dict[str, Any]], str]:
            """Convert trajectories to markdown and combine them."""
            print(f"\nConverting {len(trajectory_list)} trajectories from {label}...")
            trace_paths = [t["trace_path"] for t in trajectory_list]
            traj_data = load_and_convert_trajectories(trace_paths)

            # Add rollout_id to each trajectory
            for i, traj in enumerate(traj_data):
                traj["rollout_id"] = trajectory_list[i]["rollout_id"]

            # Combine trajectories into markdown text
            text_parts = []
            for traj in traj_data:
                rollout_label = f"Rollout {traj['rollout_id']}"
                text_parts.append(
                    f"---TRAJECTORY---\n\n**{rollout_label}**\n\n{traj['markdown']}"
                )
            combined_text = "\n\n".join(text_parts)

            return traj_data, combined_text

        # Load and convert trajectories for both sets
        traj_data_a, trajectories_set_a = convert_and_combine_trajectories(
            trajectories_a, label_a
        )
        traj_data_b, trajectories_set_b = convert_and_combine_trajectories(
            trajectories_b, label_b
        )

        # Extract task description from first trajectory
        task_description = extract_task_description(traj_data_a[0]["trace_data"], task_id=task_id)

        # Get LLM for comparison
        lm = LM_DICT[comparison_model]

        print(f"\nComparing trajectory sets using {comparison_model}...")
        with dspy.context(lm=lm):
            comparator = CompareModelTrajectorysets()
            result = comparator(
                trajectories_set_a=trajectories_set_a,
                trajectories_set_b=trajectories_set_b,
                task_description=task_description,
            )

        # Prepare output
        total_trajectories = len(traj_data_a) + len(traj_data_b)

        comparison_result = {
            "example_id": example_id,
            "num_trajectories": total_trajectories,  # For compatibility with compare_examples_batch
            "set_a": {
                "label": label_a,
                "path": str(path_a),
                "num_rollouts": len(traj_data_a),
                "rollout_ids": [t["rollout_id"] for t in traj_data_a],
            },
            "set_b": {
                "label": label_b,
                "path": str(path_b),
                "num_rollouts": len(traj_data_b),
                "rollout_ids": [t["rollout_id"] for t in traj_data_b],
            },
            "comparison_model": comparison_model,
            "task_description": task_description,
            "comparison": result.comparison,
        }

        print(f"\n✅ Comparison complete")

        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Add output_path to result (for compatibility with compare_examples_batch)
            comparison_result["output_path"] = str(output_path)

            # Save as JSON
            if output_path.suffix == ".json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            # Save as Markdown
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Trajectory Set Comparison\n\n")
                    f.write(f"**Example:** example{example_id}\n\n")
                    f.write(f"## Sets Compared\n\n")
                    f.write(f"**Set A:** {label_a}\n")
                    f.write(f"  - Path: {path_a}\n")
                    f.write(f"  - Rollouts: {len(traj_data_a)} ({', '.join([t['rollout_id'] for t in traj_data_a])})\n\n")
                    f.write(f"**Set B:** {label_b}\n")
                    f.write(f"  - Path: {path_b}\n")
                    f.write(f"  - Rollouts: {len(traj_data_b)} ({', '.join([t['rollout_id'] for t in traj_data_b])})\n\n")
                    f.write(f"**Comparison Model:** {comparison_model}\n\n")
                    f.write(f"## Task Description\n\n{task_description}\n\n")
                    f.write(f"## Comparison Analysis\n\n{result.comparison}\n\n")
                    f.write(f"---\n\n## Set A Trajectories\n\n{trajectories_set_a}\n\n")
                    f.write(f"---\n\n## Set B Trajectories\n\n{trajectories_set_b}\n")

            print(f"✅ Comparison saved to: {output_path}")

        return comparison_result

    except Exception as e:
        return {
            "example_id": example_id,
            "error": str(e)
        }


def compare_trajectory_sets_batch(
    path_a: str,
    path_b: str,
    num_examples: int,
    comparison_model: str = "gemini-2.5-flash",
    output_dir: Path = None,
    max_rollouts: int = None,
    label_a: str = None,
    label_b: str = None,
    max_workers: int = 8,
    task_id: str = None,
    model_name: str = None,
    prompt_name: str = None,
) -> List[Dict[str, Any]]:
    """
    Compare trajectory rollout sets from two arbitrary paths across multiple examples.

    This is a batch processing version of _compare_single_example_trajectory_sets that
    processes multiple examples in parallel.

    Args:
        path_a: Path to first rollout directory (e.g., "results/webtest/model_default/rollouts/v0")
        path_b: Path to second rollout directory (e.g., "results/webtest/model_default/rollouts/v1")
        num_examples: Number of examples to compare
        comparison_model: Model to use for comparison analysis (default: "gemini-2.5-flash")
        output_dir: Directory to save comparison results (auto-generated if not specified)
        max_rollouts: Maximum number of rollouts to include per set (default: all)
        label_a: Optional label for first set (default: extracted from path)
        label_b: Optional label for second set (default: extracted from path)
        max_workers: Maximum number of parallel workers (default: 8)

    Returns:
        List of comparison results for each example

    Example:
        >>> results = compare_trajectory_sets_batch(
        ...     path_a="results/webtest/qwen3-coder-30b-a3b_default/rollouts/v0",
        ...     path_b="results/webtest/qwen3-coder-30b-a3b_default/rollouts/v1",
        ...     num_examples=10,
        ... )
    """
    from pathlib import Path

    path_a = Path(path_a)
    path_b = Path(path_b)

    # Auto-generate labels from paths if not provided
    if label_a is None:
        label_a = path_a.name
    if label_b is None:
        label_b = path_b.name

    # Auto-generate output directory if not provided

    print(f"\n{'='*80}")
    print(f"BATCH TRAJECTORY SET COMPARISON")
    print(f"{'='*80}")
    print(f"Set A: {label_a}")
    print(f"  Path: {path_a}")
    print(f"Set B: {label_b}")
    print(f"  Path: {path_b}")
    print(f"Examples: {num_examples}")
    print(f"Comparison Model: {comparison_model}")
    print(f"Max Workers: {max_workers}")
    if max_rollouts:
        print(f"Max Rollouts per Set: {max_rollouts}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}\n")

    # Prepare arguments for batch inference
    args_list = []
    for example_id in range(num_examples):
        output_filename = f"example{example_id}_{label_a}_vs_{label_b}.md"
        output_path = output_dir / output_filename

        args_list.append({
            "path_a": str(path_a),
            "path_b": str(path_b),
            "example_id": example_id,
            "comparison_model": comparison_model,
            "output_path": output_path,
            "max_rollouts": max_rollouts,
            "label_a": label_a,
            "label_b": label_b,
            "task_id": task_id,
        })

    # Run comparisons in parallel using batch inference
    results = batch_inference(
        program=_compare_single_example_trajectory_sets,
        args_list=args_list,
        use_process=False,  # Use threads since we're doing I/O and API calls
        max_workers=max_workers
    )

    # Print summary for each result
    successful = 0
    failed = 0
    for result in results:
        example_id = result["example_id"]
        if "error" in result:
            print(f"⚠️  Example {example_id}: {result.get('error', 'Unknown error')}")
            failed += 1
        else:
            set_a = result.get("set_a", {})
            set_b = result.get("set_b", {})
            print(f"✅ Example {example_id}: {set_a.get('num_rollouts', 0)} vs {set_b.get('num_rollouts', 0)} rollouts")
            successful += 1

    # Save summary of all comparisons
    summary_path = output_dir / "batch_comparison_summary.json"
    summary_data = {
        "comparison_model": comparison_model,
        "num_examples_processed": len(results),
        "results": results
    }

    # Add optional task metadata fields (for compatibility with compare_examples_batch)
    if task_id:
        summary_data["task_id"] = task_id
    if model_name:
        summary_data["model_name"] = model_name
    if prompt_name:
        summary_data["prompt_name"] = prompt_name

    # Add trajectory set comparison specific fields
    summary_data.update({
        "path_a": str(path_a),
        "path_b": str(path_b),
        "label_a": label_a,
        "label_b": label_b,
        "successful": successful,
        "failed": failed,
        "max_rollouts": max_rollouts,
    })

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"BATCH COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Total examples: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed/Skipped: {failed}")
    print(f"Summary saved to: {summary_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")

    return results


class AggregateAnalysis(dspy.Signature):
    """Aggregate multiple trajectory comparison analyses to identify common patterns.

    Analyze all comparison results together to extract:
    1) Recurring behavioral patterns (found in 50%+ of examples)
    2) Common failure modes and their frequency
    3) Universal improvements that would benefit most examples
    4) Outlier cases that differ from the pattern
    """

    comparison_summaries = dspy.InputField(
        desc="List of comparison analyses from multiple examples, each containing behavioral differences"
    )
    task_description = dspy.InputField(
        desc="The original task/request that the agent was trying to complete"
    )
    aggregated_patterns = dspy.OutputField(
        desc="List of common behavioral patterns found across examples. Each pattern should include: 1) Description of the pattern, 2) Frequency (number/percentage of examples), 3) Impact on task success, 4) Example IDs where observed"
    )
    recommended_improvements = dspy.OutputField(
        desc="List of universal improvements that would address the common patterns. Each improvement should be actionable and specific."
    )


class AggregateComparisons(dspy.Module):
    """Module for aggregating multiple comparison analyses."""

    def __init__(self):
        super().__init__()
        self.aggregate = dspy.ChainOfThought(AggregateAnalysis)

    def forward(self, comparison_summaries: str, task_description: str, config):
        """Aggregate comparisons and identify common patterns."""
        return self.aggregate(
            comparison_summaries=comparison_summaries,
            task_description=task_description,
            config=config,
        )


def aggregate_comparison_analyses(
    comparison_dir: Path,
    model_name: str = "gemini-2.5-flash",
    output_path: Path = None,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    Aggregate multiple comparison analyses to identify common patterns.

    This function loads all individual comparison results from a directory,
    uses an LLM to synthesize common patterns, and generates a unified analysis
    that can be used for creating universal patches.

    Args:
        comparison_dir: Directory containing comparison result files
        model_name: Model to use for aggregation (default: "gemini-2.5-flash")
        output_path: Optional path to save aggregated analysis
        random_seed: Random seed for reproducibility in LLM calls (default: 0)

    Returns:
        Dictionary containing aggregated analysis results
    """
    comparison_dir = Path(comparison_dir)

    # Load the batch comparison summary
    summary_file = comparison_dir / "batch_comparison_summary.json"
    if not summary_file.exists():
        raise ValueError(f"Batch comparison summary not found: {summary_file}")

    with open(summary_file, 'r', encoding='utf-8') as f:
        batch_summary = json.load(f)

    # Extract comparison analyses from all examples
    comparison_summaries = []
    successful_examples = []

    for result in batch_summary.get("results", []):
        if "error" in result:
            continue

        example_id = result.get("example_id")
        comparison = result.get("comparison", "")

        if comparison:
            successful_examples.append(example_id)
            comparison_summaries.append({
                "example_id": example_id,
                "analysis": comparison
            })

    if not comparison_summaries:
        raise ValueError("No successful comparison analyses found")

    # Extract task description from first result (it's at the top level now)
    task_description = None
    for result in batch_summary["results"]:
        if "error" not in result and result.get("task_description"):
            task_description = result["task_description"]
            break

    if not task_description:
        task_description = "Task description not found"

    # Format summaries for aggregation
    formatted_summaries = []
    for summary in comparison_summaries:
        example_id = summary["example_id"]
        analysis = summary["analysis"]
        if isinstance(analysis, list):
            analysis_text = "\n".join(f"- {item}" for item in analysis)
        else:
            analysis_text = str(analysis)
        formatted_summaries.append(f"## Example {example_id}\n{analysis_text}")

    summaries_text = "\n\n".join(formatted_summaries)

    # Get LLM for aggregation
    lm = LM_DICT[model_name]

    print(f"Aggregating {len(comparison_summaries)} comparison analyses using {model_name} (seed={random_seed})...")
    with dspy.context(lm=lm):
        aggregator = AggregateComparisons()
        result = aggregator(
            comparison_summaries=summaries_text,
            task_description=task_description,
            config={"rollout_id": random_seed},
        )

    # Prepare output
    aggregated_result = {
        "model": model_name,
        "num_examples_analyzed": len(comparison_summaries),
        "example_ids": successful_examples,
        "task_description": task_description,
        "common_patterns": result.aggregated_patterns,
        "recommended_improvements": result.recommended_improvements,
        "comparison_summaries": comparison_summaries
    }

    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        if output_path.suffix == ".json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_result, f, indent=2, ensure_ascii=False)
        # Save as Markdown
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Aggregated Trajectory Analysis\n\n")
                f.write(f"**Analysis Model:** {model_name}\n")
                f.write(f"**Examples Analyzed:** {len(comparison_summaries)}\n")
                f.write(f"**Example IDs:** {', '.join(map(str, successful_examples))}\n\n")

                f.write(f"## Task Description\n\n{task_description}\n\n")

                f.write(f"## Common Patterns Across Examples\n\n")

                # Handle aggregated_patterns as either list or string
                if isinstance(result.aggregated_patterns, list):
                    for i, pattern in enumerate(result.aggregated_patterns, 1):
                        f.write(f"### Pattern {i}\n{pattern}\n\n")
                else:
                    f.write(f"{result.aggregated_patterns}\n\n")

                f.write(f"## Recommended Improvements\n\n")

                # Handle recommended_improvements as either list or string
                if isinstance(result.recommended_improvements, list):
                    for i, improvement in enumerate(result.recommended_improvements, 1):
                        f.write(f"{i}. {improvement}\n")
                else:
                    f.write(f"{result.recommended_improvements}\n")

        print(f"Aggregated analysis saved to: {output_path}")

    return aggregated_result
