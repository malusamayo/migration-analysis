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

    def forward(self, trajectories: str, task_description: str):
        """Compare trajectories and return analysis."""
        return self.compare(
            trajectories=trajectories,
            task_description=task_description
        )


def compare_rollout_trajectories(
    trace_paths: List[Path],
    model_name: str = "gemini-2.5-flash",
    output_path: Path = None
) -> Dict[str, Any]:
    """
    Compare multiple trajectory rollouts using LLM analysis.

    Args:
        trace_paths: List of paths to trace JSON files
        model_name: Name of the language model to use for comparison
        output_path: Optional path to save comparison results

    Returns:
        Dictionary containing comparison results
    """
    # Load all trajectories
    print(f"Loading {len(trace_paths)} trajectory traces...")
    trajectories_data = load_and_convert_trajectories(trace_paths)

    # Extract task description from first trajectory
    task_description = extract_task_description(trajectories_data[0]["trace_data"])

    # Combine trajectories with separators
    trajectories_text = combine_trajectories_markdown(trajectories_data)

    # Get LLM for comparison
    lm = LM_DICT[model_name]

    print(f"Comparing trajectories using {model_name}...")
    with dspy.context(lm=lm):
        comparator = CompareTrajectories()
        result = comparator(
            trajectories=trajectories_text,
            task_description=task_description
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
    comparison_model: str
) -> Dict[str, Any]:
    """
    Helper function to compare trajectories for a single example.
    Designed to be used with batch_inference for parallel processing.

    Args:
        example_id: Example identifier
        base_dir: Base directory containing workspace directories
        output_dir: Directory to save comparison results
        comparison_model: Model to use for comparison

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
        result = compare_rollout_trajectories(
            trace_paths=trace_files,
            model_name=comparison_model,
            output_path=output_path
        )
        return {
            "example_id": example_id,
            "num_trajectories": len(trace_files),
            "comparison": result,
            "output_path": str(output_path)
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
    max_workers: int = 8
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

    Returns:
        List of comparison results for each example
    """
    # Determine base directory based on execution mode
    if agentic:
        base_dir = Path(f"results/{task_id}/{model_name}_{prompt_name}_agentic_workspace")
    else:
        base_dir = Path(f"results/{task_id}/{model_name}_{prompt_name}_workspace")

    if not base_dir.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    if output_dir is None:
        output_dir = Path(f"results/{task_id}/{model_name}_{prompt_name}/comparisons")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Starting batch comparison for {num_examples} examples using {max_workers} workers")
    print(f"{'='*80}\n")

    # Prepare arguments for batch inference
    args_list = [
        {
            "example_id": example_id,
            "base_dir": base_dir,
            "output_dir": output_dir,
            "comparison_model": comparison_model
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

    def forward(self, comparison_summaries: str, task_description: str):
        """Aggregate comparisons and identify common patterns."""
        return self.aggregate(
            comparison_summaries=comparison_summaries,
            task_description=task_description
        )


def aggregate_comparison_analyses(
    comparison_dir: Path,
    model_name: str = "gemini-2.5-flash",
    output_path: Path = None
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
        comparison = result.get("comparison", {})
        comparison_analysis = comparison.get("comparison_analysis", [])

        if comparison_analysis:
            successful_examples.append(example_id)
            comparison_summaries.append({
                "example_id": example_id,
                "analysis": comparison_analysis
            })

    if not comparison_summaries:
        raise ValueError("No successful comparison analyses found")

    # Extract task description from first comparison
    first_comparison = batch_summary["results"][0].get("comparison", {})
    task_description = first_comparison.get("task_description", "Task description not found")

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

    print(f"Aggregating {len(comparison_summaries)} comparison analyses using {model_name}...")
    with dspy.context(lm=lm):
        aggregator = AggregateComparisons()
        result = aggregator(
            comparison_summaries=summaries_text,
            task_description=task_description,
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
