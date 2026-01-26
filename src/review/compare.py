"""
Module for comparing agent trajectories using LLM analysis.

This module provides DSPy-based components and functions for comparing
multiple trajectory rollouts and analyzing behavioral differences.
"""

import json
import dspy
from pathlib import Path
from typing import List, Dict, Any

from .trajectory_utils import (
    load_and_convert_trajectories,
    combine_trajectories_markdown,
    extract_task_description,
    load_eval_results,
)
from ..utils import LM_DICT, batch_inference


# =============================================================================
# Core DSPy Modules
# =============================================================================

class TrajectoryComparison(dspy.Signature):
    """Compare multiple agent trajectories and provide analysis of behavioral differences.

    Each trajectory can come from different prompts, models, or configurations.
    For each difference, specify:
    1) Which trajectories (by annotation/label) exhibit the behavior
    2) What the behavioral difference is (e.g., different tool usage, reasoning approach, error handling)
    3) Impact on task completion
    """

    trajectories = dspy.InputField(
        desc="Markdown-formatted trajectories with annotations (model, prompt, etc.), separated by '---TRAJECTORY---'"
    )
    task_description = dspy.InputField(
        desc="The original task/request that the agent was trying to complete"
    )
    comparison = dspy.OutputField(
        desc="A markdown list of behavioral differences between trajectories, indicating which trajectories exhibit each behavior"
    )
    insight_extracted = dspy.OutputField(
        desc="A markdown list of short, actionable insights extracted from the comparison"
    )


class CompareTrajectories(dspy.Module):
    """Unified module for comparing agent trajectories.

    This module can compare trajectories from any source (different models, prompts,
    skill versions, etc.). Each trajectory is annotated with metadata to identify its source.
    """

    def __init__(self):
        super().__init__()
        self.compare = dspy.ChainOfThought(TrajectoryComparison)

    def forward(self, trajectories: str, task_description: str, config=None):
        """Compare trajectories and return analysis.

        Args:
            trajectories: Markdown-formatted trajectories with annotations
            task_description: The original task description
            config: Optional configuration dict (e.g., for random seed)

        Returns:
            DSPy prediction with comparison analysis
        """
        if config is None:
            config = {}
        return self.compare(
            trajectories=trajectories,
            task_description=task_description,
            config=config
        )


# =============================================================================
# Helper Functions for Trajectory Comparison
# =============================================================================

def generate_comparison_output_dir(
    paths: List[str],
    random_seed: int = 0,
) -> Path:
    """
    Generate output directory path for comparison results based on paths.

    The output directory is created under the first path's subdirectory:
    - Path format: results/{task_id}/{model_prompt}/rollouts/{version}
    - Output format: results/{task_id}/{model_prompt}/comparisons/...

    Args:
        paths: List of rollout directory paths
        random_seed: Random seed used for comparison (default: 0)

    Returns:
        Path object for the output directory

    Example:
        >>> paths = ["results/webtest/qwen3-coder-30b-a3b_static/rollouts/v0",
        ...          "results/webtest/gemini-3-flash-preview_static/rollouts/v1"]
        >>> generate_comparison_output_dir(paths, random_seed=42)
        Path('results/webtest/qwen3-coder-30b-a3b_static/comparisons/v0_vs_gemini-3-flash-preview_static-v1_seed42')
    """
    assert paths, "At least one path must be specified"

    first_path = Path(paths[0])

    # Check if parent is 'rollouts' directory
    if first_path.parent.name != "rollouts":
        raise ValueError(
            f"Expected 'rollouts' directory in path, got '{first_path.parent.name}'. "
            f"Path should be in format: results/{{task_id}}/{{model_prompt}}/rollouts/{{version}}"
        )

    # Navigate up from rollouts/version to get base directory
    # e.g., results/webtest/qwen3-coder-30b-a3b_static/rollouts/v0
    #    -> results/webtest/qwen3-coder-30b-a3b_static
    base_dir = first_path.parent.parent
    rollout_id = first_path.name

    # Create comparison subdirectory name
    if len(paths) > 1:
        # Include model+prompt+rollout info from other paths being compared
        other_path_info = []
        for p in paths[1:]:
            path_obj = Path(p)
            model_prompt = path_obj.parent.parent.name  # e.g., "gemini-3-flash-preview_static"
            rollout_version = path_obj.name  # e.g., "v0"
            other_path_info.append(f"{model_prompt}-{rollout_version}")

        paths_str = "_".join(other_path_info[:2])  # Use first 2 other paths
        if len(other_path_info) > 2:
            paths_str += f"_and_{len(other_path_info)-2}_more"
        output_dir = base_dir / "comparisons" / f"{rollout_id}_vs_{paths_str}_seed{random_seed}"
    else:
        # Single path comparison
        output_dir = base_dir / "comparisons" / f"{rollout_id}_seed{random_seed}"

    return output_dir


def collect_trace_files(
    task_id: str,
    example_id: int,
    paths: List[str],
    max_traces: int = 3,
) -> List[Dict[str, Any]]:
    """
    Collect trace files with maximum score spread across arbitrary rollout paths.

    This function finds all trace files for a given example across different paths
    (which can represent different models, prompts, rollout versions, or any combination),
    loads their scores, and selects traces with maximum score spread.

    Args:
        task_id: Task identifier (e.g., "webgen", "webtest")
        example_id: Example identifier (e.g., 0 for example0)
        paths: List of rollout directory paths (e.g., ["results/webgen/gpt-4_default/rollouts/v0",
               "results/webgen/claude-3_v1/rollouts/v0"])
        max_traces: Maximum number of traces to return (default: 3)

    Returns:
        List of dictionaries containing trace information with maximum score spread.
        Each dict contains:
        - trace_path: Path to the trace file
        - score_info: Score dictionary (score, pass_rate, success)
        - rollout_id: Rollout identifier
        - path: Source rollout path
        - annotation: Annotation dict with path info
    """
    all_traces = []

    # Iterate through all provided paths
    for path_str in paths:
        base_dir = Path(path_str)

        if not base_dir.exists():
            print(f"⚠️  Warning: Directory not found: {base_dir}")
            continue

        # Load eval results for this path
        eval_results_path = base_dir / "eval_results.yaml"
        eval_results = load_eval_results(eval_results_path)

        # Find all rollout directories for this example
        rollout_pattern = f"example{example_id}_rollout*"
        rollout_dirs = sorted(base_dir.glob(rollout_pattern))

        for rollout_dir in rollout_dirs:
            # Find trace file in the workspace
            traces = list(rollout_dir.glob("trace_*.json"))

            if not traces:
                continue

            if len(traces) > 1:
                print(f"⚠️  Warning: Multiple trace files in {rollout_dir}, using first")

            trace_file = traces[0]

            # Extract rollout ID from directory name
            rollout_id = rollout_dir.name.split("_rollout")[1] if "_rollout" in rollout_dir.name else "0"

            # Extract model and prompt from path
            # Path format: results/{task_id}/{model}_{prompt}/rollouts/{version}
            path_parts = Path(path_str).parts
            if len(path_parts) >= 3:
                model_prompt = path_parts[2]  # e.g., "qwen3-coder-30b-a3b_static"
                if "_" in model_prompt:
                    # Split on last underscore to handle models with underscores
                    parts = model_prompt.rsplit("_", 1)
                    model = parts[0]
                    prompt = parts[1]
                else:
                    model = model_prompt
                    prompt = "unknown"
            else:
                model = "unknown"
                prompt = "unknown"

            eval_result = eval_results.get(str(rollout_dir), {})

            # Add to collection
            all_traces.append({
                "trace_path": trace_file,
                "path": path_str,
                "score_info": eval_result if eval_result else None,
                "annotation": {
                    "model": model,
                    "prompt": prompt,
                    "score": eval_result.get("score"),
                    "feedback": eval_result.get("feedback"),
                }
            })

    if not all_traces:
        raise ValueError(f"No traces found for example{example_id} across given models/prompts")

    # Filter traces with valid scores
    traces_with_scores = [t for t in all_traces if t["score_info"] is not None]

    if not traces_with_scores:
        print(f"⚠️  Warning: No traces with scores found, returning first {max_traces} traces")
        return all_traces[:max_traces]

    # Sort by score
    traces_with_scores.sort(key=lambda t: t["score_info"]["score"])

    # Select traces with maximum spread
    if len(traces_with_scores) <= max_traces:
        selected_traces = traces_with_scores
    else:
        # Strategy: select min, max, and middle traces for maximum spread
        selected_traces = []

        # Always include lowest score
        selected_traces.append(traces_with_scores[0])

        # Always include highest score
        selected_traces.append(traces_with_scores[-1])

        # Select middle trace(s) to maximize spread
        remaining = max_traces - 2
        if remaining > 0:
            # Distribute remaining selections evenly
            step = len(traces_with_scores) // (remaining + 1)
            for i in range(1, remaining + 1):
                idx = min(i * step, len(traces_with_scores) - 2)
                selected_traces.append(traces_with_scores[idx])

        # Sort by score again for consistent ordering
        selected_traces.sort(key=lambda t: t["score_info"]["score"])

    print(f"\n{'='*80}")
    print(f"Selected {len(selected_traces)} traces for example{example_id}:")

    return selected_traces


def compare_rollout_trajectories(
    selected_traces: List[Dict[str, Any]],
    model_name: str = "gemini-2.5-flash",
    output_path: Path = None,
    random_seed: int = 0,
    task_id: str = None,
) -> Dict[str, Any]:
    """
    Compare multiple trajectory rollouts using LLM analysis.

    Args:
        selected_traces: List of trace info dicts from collect_trace_files, each containing:
            - trace_path: Path to trace file
            - score_info: Score dictionary
            - rollout_id: Rollout identifier
            - model: Model name
            - prompt: Prompt name
            - annotation: Annotation dict
        model_name: Name of the language model to use for comparison
        output_path: Optional path to save comparison results
        random_seed: Random seed for reproducibility in LLM calls
        task_id: Optional task identifier (e.g., "webtest", "webgen")

    Returns:
        Dictionary containing comparison results
    """
    # Extract information from selected traces
    trace_paths = [t["trace_path"] for t in selected_traces]
    annotations = [t["annotation"] for t in selected_traces]

    # Load all trajectories
    print(f"Loading {len(trace_paths)} trajectory traces...")
    trajectories_data = load_and_convert_trajectories(trace_paths)

    # Extract task description from first trajectory
    task_description = extract_task_description(trajectories_data[0]["trace_data"], task_id=task_id)

    if annotations:
        for i, traj in enumerate(trajectories_data):
            if i < len(annotations):
                traj["annotations"] = annotations[i]

    # Combine trajectories with separators
    trajectories_text = combine_trajectories_markdown(
        trajectories_data,
        annotations=annotations
    )

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
        "insight_extracted": result.insight_extracted,
        "trajectories_markdown": trajectories_text
    }

    if annotations:
        comparison_result["annotations"] = annotations

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
                f.write(f"**Number of Trajectories:** {len(trace_paths)}\n")
                f.write(f"\n## Task Description\n\n{task_description}\n\n")
                f.write(f"## Comparison Analysis\n\n{result.comparison}\n\n")
                f.write(f"## Insight Extracted\n\n{result.insight_extracted}\n\n")
                f.write(f"---\n\n## Detailed Trajectories\n\n{trajectories_text}\n")

        print(f"Comparison saved to: {output_path}")

    return comparison_result


def compare_example(
    task_id: str,
    example_id: int,
    paths: List[str],
    max_traces: int = 3,
    comparison_model: str = "gemini-2.5-flash",
    output_path: Path = None,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    Unified function to compare trajectories for a single example.

    This function:
    1. Collects traces with maximum score spread using collect_trace_files
    2. Compares them using compare_rollout_trajectories

    Args:
        task_id: Task identifier (e.g., "webgen", "webtest")
        example_id: Example identifier (e.g., 0 for example0)
        paths: List of rollout directory paths (e.g., ["results/webgen/gpt-4_default/rollouts/v0",
               "results/webgen/claude-3_v1/rollouts/v0"])
        max_traces: Maximum number of traces to compare (default: 3)
        comparison_model: Model to use for comparison (default: "gemini-2.5-flash")
        output_path: Optional path to save comparison results
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing comparison results or error

    Example:
        # Compare across different paths
        result = compare_example(
            task_id="webgen",
            example_id=0,
            paths=[
                "results/webgen/gpt-4_default/rollouts/v0",
                "results/webgen/claude-3_v1/rollouts/v0"
            ],
            max_traces=3
        )
    """
    try:
        # Collect traces with maximum score spread
        selected_traces = collect_trace_files(
            task_id=task_id,
            example_id=example_id,
            paths=paths,
            max_traces=max_traces
        )

        # Compare the selected trajectories
        result = compare_rollout_trajectories(
            selected_traces=selected_traces,
            model_name=comparison_model,
            output_path=output_path,
            random_seed=random_seed,
            task_id=task_id,
        )

        # Add example_id and summary info for batch compatibility
        return {
            "example_id": example_id,
            "num_trajectories": len(selected_traces),
            "comparison_model": comparison_model,
            "task_description": result["task_description"],
            "comparison": result["comparison_analysis"],
            "insight_extracted": result["insight_extracted"],
            "output_path": str(output_path) if output_path else None,
            "annotations": result["annotations"],
        }

    except Exception as e:
        return {
            "example_id": example_id,
            "error": str(e)
        }


# =============================================================================
# Batch Processing Functions
# =============================================================================

def compare_examples_batch(
    task_id: str,
    paths: List[str],
    output_dir: Path,
    num_examples: int = 10,
    comparison_model: str = "gemini-2.5-flash",
    max_workers: int = 8,
    random_seed: int = 0,
    max_traces: int = 3,
) -> List[Dict[str, Any]]:
    """
    Compare trajectories for multiple examples using parallel batch inference.

    This function uses the unified compare_example function for each example.

    Args:
        task_id: Task identifier (e.g., "webgen", "webtest")
        paths: List of rollout directory paths (e.g., ["results/webgen/gpt-4_default/rollouts/v0",
               "results/webgen/claude-3_v1/rollouts/v0"])
        num_examples: Number of examples to compare (default: 10)
        comparison_model: Model to use for comparison (default: "gemini-2.5-flash")
        output_dir: Directory to save comparison results
        max_workers: Maximum number of parallel workers (default: 8)
        random_seed: Random seed for reproducibility in LLM calls (default: 0)
        max_traces: Maximum traces per example (default: 3)

    Returns:
        List of comparison results for each example
    """
    # Set defaults
    assert paths, "At least one path must be specified"

    print(f"\n{'='*80}")
    print(f"Starting batch comparison for {num_examples} examples using {max_workers} workers")
    print(f"Paths: {paths}")
    print(f"Seed: {random_seed}")
    print(f"{'='*80}\n")

    # Prepare arguments for batch inference
    args_list = [
        {
            "task_id": task_id,
            "example_id": example_id,
            "paths": paths,
            "max_traces": max_traces,
            "comparison_model": comparison_model,
            "output_path": output_dir / f"example{example_id}_comparison.md",
            "random_seed": random_seed,
        }
        for example_id in range(num_examples)
    ]

    # Run comparisons in parallel
    results = batch_inference(
        program=compare_example,
        args_list=args_list,
        use_process=False,
        max_workers=max_workers
    )

    # Print summary
    successful = sum(1 for r in results if "error" not in r)
    failed = sum(1 for r in results if "error" in r)

    for result in results:
        example_id = result["example_id"]
        if "error" in result:
            print(f"⚠️  Example {example_id}: {result['error']}")
        else:
            print(f"✅ Example {example_id}: Compared {result['num_trajectories']} trajectories -> {result['output_path']}")

    # Save summary
    summary_path = output_dir / "batch_comparison_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "task_id": task_id,
            "paths": paths,
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