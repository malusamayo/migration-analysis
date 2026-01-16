"""
Utilities for loading and converting trajectory traces.

This module provides functions for loading trajectory trace JSON files
and converting them to markdown format for analysis.
"""

import json
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
    markdown_lines.append("\n---\n\n## Agent Execution Trace\n")

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
                            markdown_lines.append(f"\n### User Request\n")
                            markdown_lines.append(f"{content_text}\n")
                        elif role == "assistant":
                            action_count += 1
                            markdown_lines.append(f"\n### Step {action_count}\n")

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
            markdown_lines.append(f"\n### Step {action_count}\n")

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

                markdown_lines.append("")

        # Handle ObservationEvent (results from actions)
        elif event_kind == "ObservationEvent":
            if "observation" in event and event["observation"]:
                obs = event["observation"]
                # Only show first 500 chars of observation to keep it concise
                obs_str = str(obs)
                if len(obs_str) > 500:
                    obs_str = obs_str[:500] + "...\n[Output truncated]"

                markdown_lines.append(f"**Observation:**\n")
                markdown_lines.append("```")
                markdown_lines.append(obs_str)
                markdown_lines.append("```\n")

        # Handle any uncaptured event types
        else:
            assert False, f"Unhandled event kind: {event_kind}"

    return "\n".join(markdown_lines)

def load_trajectory_trace(trace_path: Path) -> Dict[str, Any]:
    """Load a trajectory trace JSON file."""
    with open(trace_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_task_description(trace_data: Dict[str, Any]) -> str:
    """Extract the system instructions from trace data."""
    # Look for the SystemPromptEvent which contains the agent's instructions
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


def combine_trajectories_markdown(trajectories_data: list[Dict[str, Any]]) -> str:
    """
    Combine multiple trajectory markdowns with separators.

    Args:
        trajectories_data: List of trajectory data dictionaries

    Returns:
        Combined markdown text with trajectory separators
    """
    combined_trajectories = []
    for i, traj in enumerate(trajectories_data, 1):
        combined_trajectories.append(
            f"## Trajectory {i}\n**File:** {traj['path']}\n\n{traj['markdown']}"
        )
    return "\n\n---TRAJECTORY---\n\n".join(combined_trajectories)
