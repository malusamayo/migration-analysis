"""Utility functions for analyzing evaluation results."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np


def build_summary_dataframe(
    result_paths: List[Union[str, Path]],
    include_trajectories: bool = True
) -> pd.DataFrame:
    """
    Build a summary dataframe from evaluation result files.

    Args:
        result_paths: List of paths to evaluation result files (JSON or YAML)
        include_trajectories: Whether to include trajectory metrics (default: True)

    Returns:
        DataFrame with aggregated metrics across all result files, including:
            - prompt_name: Extracted from path (task name)
            - model_name: Extracted from path
            - rollout_id: Extracted from path
            - num_examples: Number of examples
            - avg_score: Average score across all examples
            - avg_cost: Average accumulated cost
            - avg_tokens: Average tokens per turn
            - avg_steps: Average number of steps
            - scores_by_rollout: Scores grouped by rollout
            - raw_data: List of raw evaluation results

    Example:
        >>> paths = [
        ...     "./results/webtest/qwen3-coder-30b-a3b_static/rollouts/v0/eval_results.yaml",
        ...     "./results/webtest/gemini-3-flash-preview_static/rollouts/v0/eval_results.yaml"
        ... ]
        >>> df = build_summary_dataframe(paths)
        >>> print(df[['model_name', 'avg_score', 'avg_cost']])
    """
    summary_rows = []

    for path in result_paths:
        path = Path(path)

        # Load results
        with open(path) as f:
            if path.suffix in ['.yaml', '.yml']:
                import yaml
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        if not data:
            continue

        # Extract metadata from path
        prompt_name = _extract_prompt_name(path)
        model_name = _extract_model_name(path)
        rollout_id = _extract_rollout_id(path)

        # Calculate basic metrics
        scores = [e['score'] for e in data]
        avg_score = np.mean(scores)

        row = {
            'prompt_name': prompt_name,
            'model_name': model_name,
            'rollout_id': rollout_id,
            'result_path': str(path),
            'num_examples': len(data),
            'avg_score': avg_score,
            'scores': scores,
        }

        # Add trajectory metrics if requested
        if include_trajectories:
            traj_metrics = _extract_trajectory_metrics(data)
            row.update(traj_metrics)

        # Calculate scores by rollout if structure allows
        try:
            scores_array = np.array(scores)
            if len(scores_array) % 3 == 0:  # Assuming 3 rollouts per example
                rollout_scores = scores_array.reshape(-1, 3).mean(axis=0)
                row['score_rollout_0'] = rollout_scores[0]
                row['score_rollout_1'] = rollout_scores[1]
                row['score_rollout_2'] = rollout_scores[2]
        except Exception:
            pass

        row['raw_data'] = data
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _extract_model_name(path: Path) -> str:
    """Extract model name from result path."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == 'results' and i + 2 < len(parts):
            # Typically: results/{task}/{model_name}/...
            return parts[i + 2]
    return path.parent.parent.name


def _extract_prompt_name(path: Path) -> str:
    """Extract prompt name (task name) from result path."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == 'results' and i + 1 < len(parts):
            # Typically: results/{task}/{model_name}/...
            return parts[i + 1]
    return path.parent.parent.parent.name


def _extract_rollout_id(path: Path) -> str:
    """Extract rollout ID from result path."""
    parts = path.parts
    for i, part in enumerate(parts):
        if part == 'rollouts' and i + 1 < len(parts):
            # Typically: .../rollouts/{rollout_id}/eval_results.yaml
            return parts[i + 1]
    return None


def _extract_trajectory_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract trajectory metrics from evaluation data."""
    trajs = []
    missing_traces = []

    for e in data:
        workspace_dir = e.get('workspace_dir', '')
        if not workspace_dir or not os.path.exists(workspace_dir):
            missing_traces.append(workspace_dir)
            continue

        # Find trace file
        trace_file = None
        for file in os.listdir(workspace_dir):
            if file.startswith("trace_") and file.endswith(".json"):
                trace_file = os.path.join(workspace_dir, file)
                break

        if trace_file:
            with open(trace_file) as f:
                trajs.append(json.load(f))
        else:
            missing_traces.append(workspace_dir)

    if not trajs:
        return {
            'avg_cost': None,
            'avg_tokens_per_turn': None,
            'avg_steps': None,
            'num_missing_traces': len(missing_traces),
        }

    # Extract metrics
    metrics_df = pd.json_normalize([t.get('metrics', {}) for t in trajs])

    result = {
        'num_trajectories': len(trajs),
        'num_missing_traces': len(missing_traces),
    }

    # Safely extract metrics
    if 'accumulated_cost' in metrics_df.columns:
        result['avg_cost'] = metrics_df['accumulated_cost'].mean()
        result['total_cost'] = metrics_df['accumulated_cost'].sum()

    if 'accumulated_token_usage.per_turn_token' in metrics_df.columns:
        result['avg_tokens_per_turn'] = metrics_df['accumulated_token_usage.per_turn_token'].mean()

    # Extract token usage details
    if 'accumulated_token_usage.prompt_tokens' in metrics_df.columns:
        result['avg_prompt_tokens'] = metrics_df['accumulated_token_usage.prompt_tokens'].mean()
        result['total_prompt_tokens'] = metrics_df['accumulated_token_usage.prompt_tokens'].sum()

    if 'accumulated_token_usage.completion_tokens' in metrics_df.columns:
        result['avg_completion_tokens'] = metrics_df['accumulated_token_usage.completion_tokens'].mean()
        result['total_completion_tokens'] = metrics_df['accumulated_token_usage.completion_tokens'].sum()

    # Calculate total tokens if both prompt and completion are available
    if 'avg_prompt_tokens' in result and 'avg_completion_tokens' in result:
        result['avg_total_tokens'] = result['avg_prompt_tokens'] + result['avg_completion_tokens']
        result['total_total_tokens'] = result['total_prompt_tokens'] + result['total_completion_tokens']

    if 'costs' in metrics_df.columns:
        result['avg_steps'] = metrics_df['costs'].map(len).mean()

    # Calculate agent steps
    num_steps = [
        len([e for e in t.get('events', []) if e.get('source') == 'agent'])
        for t in trajs
    ]
    result['avg_agent_steps'] = np.mean(num_steps) if num_steps else None

    return result


def compare_models(
    result_paths: List[Union[str, Path]],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        result_paths: List of paths to evaluation result files
        metrics: List of metrics to include (default: all available)

    Returns:
        DataFrame with models as rows and metrics as columns

    Example:
        >>> paths = [
        ...     "./results/webtest/qwen3-coder-30b-a3b_static/rollouts/v0/eval_results.yaml",
        ...     "./results/webtest/gemini-3-flash-preview_static/rollouts/v0/eval_results.yaml"
        ... ]
        >>> comparison = compare_models(paths, metrics=['avg_score', 'avg_cost', 'avg_steps'])
        >>> print(comparison)
    """
    summary_df = build_summary_dataframe(result_paths)

    if metrics is None:
        # Default metrics to compare
        metrics = [
            'num_examples', 'avg_score', 'avg_cost', 'total_cost',
            'avg_prompt_tokens', 'avg_completion_tokens', 'avg_total_tokens',
            'avg_tokens_per_turn', 'avg_steps', 'avg_agent_steps'
        ]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in summary_df.columns]

    comparison = summary_df[['model_name'] + available_metrics].copy()
    comparison = comparison.set_index('model_name')

    return comparison


def load_scores_detailed(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and analyze scores from a single evaluation result file.

    This is a convenience function that provides detailed analysis of a single result file,
    similar to the load_scores function in your notebook.

    Args:
        path: Path to evaluation result file

    Returns:
        Dictionary with detailed analysis including raw data and metrics

    Example:
        >>> result = load_scores_detailed("./results/webtest/qwen3-coder-30b-a3b_static/rollouts/v0/eval_results.yaml")
        >>> print(f"Avg score: {result['avg_score']:.3f}")
        >>> print(f"Avg cost: {result['avg_cost']:.3f}")
    """
    path = Path(path)

    with open(path) as f:
        if path.suffix in ['.yaml', '.yml']:
            import yaml
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    scores = [e['score'] for e in data]

    # Extract trajectory data
    traj_metrics = _extract_trajectory_metrics(data)

    # Build workspace to result mapping
    data_dict = {e['workspace_dir']: e for e in data}

    result = {
        'path': str(path),
        'prompt_name': _extract_prompt_name(path),
        'model_name': _extract_model_name(path),
        'rollout_id': _extract_rollout_id(path),
        'num_examples': len(data),
        'avg_score': np.mean(scores),
        'scores': scores,
        'data_dict': data_dict,
        'raw_data': data,
    }

    # Add trajectory metrics
    result.update(traj_metrics)

    # Calculate rollout scores if applicable
    try:
        scores_array = np.array(scores)
        if len(scores_array) % 3 == 0:
            rollout_scores = scores_array.reshape(-1, 3).mean(axis=0)
            result['scores_by_rollout'] = rollout_scores.tolist()
    except Exception:
        pass

    # Print summary
    print(f"Model: {result['model_name']}")
    print(f"Examples: {result['num_examples']}")
    print(f"Avg score: {result['avg_score']:.3f}")

    if result.get('avg_cost'):
        print(f"Avg cost: ${result['avg_cost']:.3f}")
    if result.get('total_cost'):
        print(f"Total cost: ${result['total_cost']:.3f}")

    if result.get('avg_prompt_tokens'):
        print(f"Avg prompt tokens: {result['avg_prompt_tokens']:.0f}")
    if result.get('avg_completion_tokens'):
        print(f"Avg completion tokens: {result['avg_completion_tokens']:.0f}")
    if result.get('avg_total_tokens'):
        print(f"Avg total tokens: {result['avg_total_tokens']:.0f}")

    if result.get('avg_tokens_per_turn'):
        print(f"Avg tokens/turn: {result['avg_tokens_per_turn']:.1f}")
    if result.get('avg_steps'):
        print(f"Avg steps: {result['avg_steps']:.1f}")

    if 'scores_by_rollout' in result:
        print(f"Scores by rollout: {result['scores_by_rollout']}")

    if result.get('num_missing_traces', 0) > 0:
        print(f"⚠️  Missing traces: {result['num_missing_traces']}")

    return result
