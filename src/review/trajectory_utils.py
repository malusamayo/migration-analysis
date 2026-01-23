"""
Utilities for loading and converting trajectory traces.

This module provides functions for loading trajectory trace JSON files
and converting them to markdown format for analysis.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


def convert_json_to_markdown(json_data: Dict[str, Any]) -> str:
    """
    Convert JSON evaluation trace to markdown format.

    Keeps essential information like action, reasoning, and observation.

    Args:
        json_data: Dictionary containing evaluation trace data

    Returns:
        Formatted markdown string
    """
    markdown_lines = []

    assert "events" in json_data and json_data["events"]
    markdown_lines.append("\n---\n\n### Agent Execution Trace\n")

    action_count = 0
    for event in json_data["events"]:
        event_kind = event.get("kind", "Unknown")

        # Skip system prompt events as they're just setup
        if event_kind == "SystemPromptEvent":
            continue

        # Handle MessageEvent (user and assistant messages)
        if event_kind == "MessageEvent":
            if "llm_message" in event and event["llm_message"]:
                msg = event["llm_message"]
                role = msg.get("role", "")

                if msg.get("content"):
                    content_text = ""
                    if isinstance(msg["content"], list):
                        for item in msg["content"]:
                            if isinstance(item, dict) and "text" in item:
                                content_text = item["text"]
                                break
                    else:
                        content_text = str(msg["content"])

                    if content_text:
                        if role == "user":
                            markdown_lines.append(f"\n#### User Request\n")
                            markdown_lines.append(f"{content_text}\n")
                        elif role == "assistant":
                            action_count += 1
                            markdown_lines.append(f"\n#### Step {action_count}\n")

                            # Add reasoning if available
                            if "reasoning_content" in msg and msg["reasoning_content"]:
                                markdown_lines.append(f"**Reasoning:**\n")
                                markdown_lines.append(f"{msg['reasoning_content']}\n")

                            # Add the assistant's message
                            markdown_lines.append(f"**Response:**\n")
                            markdown_lines.append(f"{content_text}\n")

        # Handle ActionEvent (agent actions with reasoning)
        elif event_kind == "ActionEvent":
            action_count += 1
            markdown_lines.append(f"\n#### Step {action_count}\n")

            # Extract and add thought content if present
            thought_text = ""
            if "thought" in event and event["thought"]:
                if isinstance(event["thought"], list):
                    for item in event["thought"]:
                        if isinstance(item, dict) and "text" in item:
                            thought_text = item["text"]
                            break
                elif isinstance(event["thought"], str):
                    thought_text = event["thought"]

            if thought_text:
                markdown_lines.append(f"**Thought:**\n")
                markdown_lines.append(f"{thought_text}\n")

            # Add reasoning content
            if "reasoning_content" in event and event["reasoning_content"]:
                markdown_lines.append(f"**Reasoning:**\n")
                markdown_lines.append(f"{event['reasoning_content']}\n")

            # Add action details
            if "action" in event and event["action"]:
                action = event["action"]
                action_kind = action.get("kind", "Unknown")
                markdown_lines.append(f"**Action:** `{action_kind}`\n")

                # Add specific action details based on type
                if action_kind == "FileEditorAction":
                    command = action.get("command", "")
                    path = action.get("path", "")
                    markdown_lines.append(f"- Command: `{command}`")
                    markdown_lines.append(f"- Path: `{path}`")

                elif action_kind == "TerminalAction":
                    command = action.get("command", "")
                    markdown_lines.append(f"- Command: `{command}`")

                elif action_kind == "BrowserAction":
                    if "url" in action:
                        markdown_lines.append(f"- URL: `{action['url']}`")

                elif action_kind == "ThinkAction":
                    if "thought" in action:
                        markdown_lines.append(f"- Thought: `{action['thought']}`")

                markdown_lines.append("")

        # Handle ObservationEvent (results from actions)
        elif event_kind == "ObservationEvent":
            if "observation" in event and event["observation"]:
                obs = event["observation"]
                obs_kind = obs.get("kind", "Unknown")

                markdown_lines.append(f"**Observation:**\n")

                # Handle FileEditorObservation
                if obs_kind == "FileEditorObservation":
                    # Extract the text content from the content array
                    if "content" in obs and isinstance(obs["content"], list):
                        for item in obs["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content = item.get("text", "")
                                # Truncate if too long
                                if len(text_content) > 1000:
                                    text_content = text_content[:1000] + "...\n[Output truncated]"
                                markdown_lines.append("```")
                                markdown_lines.append(text_content)
                                markdown_lines.append("```\n")
                                break
                    else:
                        # Fallback to string representation
                        obs_str = str(obs)
                        if len(obs_str) > 500:
                            obs_str = obs_str[:500] + "...\n[Output truncated]"
                        markdown_lines.append("```")
                        markdown_lines.append(obs_str)
                        markdown_lines.append("```\n")

                # Handle TerminalObservation
                elif obs_kind == "TerminalObservation":
                    # Extract the text content from the content array
                    if "content" in obs and isinstance(obs["content"], list):
                        for item in obs["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content = item.get("text", "")
                                # Truncate if too long
                                if len(text_content) > 1000:
                                    text_content = text_content[:1000] + "...\n[Output truncated]"
                                markdown_lines.append("```")
                                markdown_lines.append(text_content)
                                markdown_lines.append("```")
                                break

                    # Add metadata if present (exit code, error status, etc.)
                    metadata_parts = []
                    if "is_error" in obs:
                        metadata_parts.append(f"Error: {obs['is_error']}")
                    if "exit_code" in obs:
                        metadata_parts.append(f"Exit code: {obs['exit_code']}")
                    if "timeout" in obs:
                        metadata_parts.append(f"Timeout: {obs['timeout']}")

                    if metadata_parts:
                        markdown_lines.append(f"*{' | '.join(metadata_parts)}*\n")
                    else:
                        markdown_lines.append("")

                # Handle other observation types (fallback)
                else:
                    # Extract text content if available
                    text_content = None
                    if "content" in obs and isinstance(obs["content"], list):
                        for item in obs["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content = item.get("text", "")
                                break

                    if text_content:
                        # Truncate if too long
                        if len(text_content) > 1000:
                            text_content = text_content[:1000] + "...\n[Output truncated]"
                        markdown_lines.append("```")
                        markdown_lines.append(text_content)
                        markdown_lines.append("```\n")
                    else:
                        # Fallback to string representation
                        obs_str = str(obs)
                        if len(obs_str) > 500:
                            obs_str = obs_str[:500] + "...\n[Output truncated]"
                        markdown_lines.append("```")
                        markdown_lines.append(obs_str)
                        markdown_lines.append("```\n")

        # Handle AgentErrorEvent (errors from tool validation or execution)
        elif event_kind == "AgentErrorEvent":
            error_msg = event.get("error", "Unknown error")
            tool_name = event.get("tool_name", "Unknown tool")

            markdown_lines.append(f"**Error ({tool_name}):**\n")
            markdown_lines.append("```")
            markdown_lines.append(error_msg)
            markdown_lines.append("```\n")

        # Handle any uncaptured event types
        else:
            assert False, f"Unhandled event kind: {event_kind}"

    return "\n".join(markdown_lines)

def load_trajectory_trace(trace_path: Path) -> Dict[str, Any]:
    """Load a trajectory trace JSON file."""
    with open(trace_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_eval_results(eval_results_path: Path) -> Dict[str, Any]:
    """
    Load evaluation results from a YAML file.

    Args:
        eval_results_path: Path to eval_results.yaml file

    Returns:
        Dictionary of evaluation results keyed by workspace directory
    """
    if not eval_results_path.exists():
        return []

    try:
        with open(eval_results_path, 'r', encoding='utf-8') as f:
            results = yaml.safe_load(f)
            results_dict = {res['workspace_dir']: res for res in results} if results else {}
            return results_dict
    except Exception as e:
        print(f"Warning: Failed to load eval results from {eval_results_path}: {e}")
        return {}


def extract_task_description(trace_data: Dict[str, Any], task_id: Optional[str] = None) -> str:
    """
    Extract the task description.

    If task_id is provided, reads from data/{task_id}/metadata.yaml.
    Otherwise, falls back to extracting from trace data's SystemPromptEvent.

    Args:
        trace_data: Dictionary containing evaluation trace data
        task_id: Optional task identifier (e.g., "webtest", "webgen")

    Returns:
        Task description string
    """
    # If task_id is provided, read from metadata.yaml
    if task_id:
        metadata_path = Path(f"data/{task_id}/metadata.yaml")
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = yaml.safe_load(f)
                    if metadata and "description" in metadata:
                        return metadata["description"]
            except Exception as e:
                print(f"Warning: Failed to load metadata from {metadata_path}: {e}")

    # Fallback: Look for the SystemPromptEvent which contains the agent's instructions
    if "events" in trace_data:
        for event in trace_data["events"]:
            if event.get("kind") == "SystemPromptEvent":
                system_prompt = event.get("system_prompt", {})
                if isinstance(system_prompt, dict) and "text" in system_prompt:
                    return system_prompt["text"]
                elif isinstance(system_prompt, str):
                    return system_prompt

    return "System instructions not found in trace data"


def load_and_convert_trajectories(trace_paths: list[Path]) -> list[Dict[str, Any]]:
    """
    Load multiple trajectory traces and convert them to markdown.

    Args:
        trace_paths: List of paths to trace JSON files

    Returns:
        List of dictionaries containing path, trace_data, and markdown for each trajectory
    """
    trajectories_data = []
    for path in trace_paths:
        trace_data = load_trajectory_trace(path)
        trajectories_data.append({
            "path": str(path),
            "trace_data": trace_data,
            "markdown": convert_json_to_markdown(trace_data)
        })
    return trajectories_data


def combine_trajectories_markdown(
    trajectories_data: list[Dict[str, Any]],
    annotations: List[Dict[str, str]] = None,
) -> str:
    """
    Combine multiple trajectory markdowns with separators.

    Args:
        trajectories_data: List of trajectory data dictionaries
        scores: Optional list of score dictionaries for each trajectory
        rollout_ids: Optional list of rollout IDs for each trajectory
        annotations: Optional list of annotation dicts for each trajectory (e.g., {"model": "gpt-4", "prompt": "v1"})

    Returns:
        Combined markdown text with trajectory separators
    """
    combined_trajectories = []
    metadata_entries = []
    for i, traj in enumerate(trajectories_data, 1):
        # Build header with trajectory number or rollout ID
        header = f"## Trajectory {i}"

        # Add annotation information if available
        annotation_text = ""
        if annotations and i-1 < len(annotations):
            ann = annotations[i-1]
            ann_parts = [f"{k}: {v}" for k, v in ann.items()]
            annotation_text = f"\n**Metadata:** {', '.join(ann_parts)}"
            metadata_entries.append(f"- Trajectory {i}: {', '.join(ann_parts)}")

        combined_trajectories.append(
            f"{header}{annotation_text}\n\n{traj['markdown']}"
        )
    combined_trajectories_text = "\n\n---TRAJECTORY---\n\n".join(combined_trajectories)
    if metadata_entries:
        metadata_section = "\n---\n\n## Trajectory Metadata\n" + "\n".join(metadata_entries) + "\n\n"
        combined_trajectories_text = combined_trajectories_text + metadata_section 
    return combined_trajectories_text