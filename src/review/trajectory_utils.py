"""
Utilities for loading and converting trajectory traces.

This module provides functions for loading trajectory trace JSON files
and converting them to markdown format for analysis.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


def _extract_text_from_content(content: Any) -> str:
    """Extract the first text block from message/observation content."""
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "text" in item:
                return item["text"]
        return ""
    if content is None:
        return ""
    return str(content)


def _truncate_text(text: str, limit: int) -> str:
    """Truncate long content for markdown readability."""
    if len(text) > limit:
        return text[:limit] + "...\n[Output truncated]"
    return text


def _append_subagent_trace_markdown(
    markdown_lines: list[str],
    agent_id: str,
    payload: Dict[str, Any],
    *,
    heading_level: int,
) -> None:
    """Append one subagent trace using headings nested under a parent step."""
    heading = "#" * heading_level
    nested_heading = "#" * min(heading_level + 1, 6)

    markdown_lines.append(f"\n{heading} Delegated Subagent `{agent_id}`\n")

    subagent_events = payload.get("events", [])
    if subagent_events:
        _append_event_stream_markdown(
            markdown_lines,
            subagent_events,
            user_heading=f"{nested_heading} User Request",
            step_heading_template=f"{nested_heading} Step {{step}}",
        )
    else:
        markdown_lines.append("\n_No subagent events recorded._\n")


def _append_event_stream_markdown(
    markdown_lines: list[str],
    events: list[Dict[str, Any]],
    *,
    user_heading: str = "#### User Request",
    step_heading_template: str = "#### Step {step}",
    subagents: Optional[Dict[str, Any]] = None,
    rendered_subagents: Optional[set[str]] = None,
) -> None:
    """Append a sequence of events to markdown_lines."""
    action_count = 0
    pending_delegate_agents: list[str] = []

    for event in events:
        event_kind = event.get("kind", "Unknown")

        if event_kind == "SystemPromptEvent":
            continue

        if event_kind == "MessageEvent":
            if "llm_message" in event and event["llm_message"]:
                msg = event["llm_message"]
                role = msg.get("role", "")
                content_text = _extract_text_from_content(msg.get("content"))

                if content_text:
                    if role == "user":
                        markdown_lines.append(f"\n{user_heading}\n")
                        markdown_lines.append(f"{content_text}\n")
                    elif role == "assistant":
                        action_count += 1
                        markdown_lines.append(
                            f"\n{step_heading_template.format(step=action_count)}\n"
                        )

                        if "reasoning_content" in msg and msg["reasoning_content"]:
                            markdown_lines.append("**Reasoning:**\n")
                            markdown_lines.append(f"{msg['reasoning_content']}\n")

                        markdown_lines.append("**Response:**\n")
                        markdown_lines.append(f"{content_text}\n")

        elif event_kind == "ActionEvent":
            action_count += 1
            markdown_lines.append(
                f"\n{step_heading_template.format(step=action_count)}\n"
            )

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
                markdown_lines.append("**Thought:**\n")
                markdown_lines.append(f"{thought_text}\n")

            if "reasoning_content" in event and event["reasoning_content"]:
                markdown_lines.append("**Reasoning:**\n")
                markdown_lines.append(f"{event['reasoning_content']}\n")

            if "action" in event and event["action"]:
                action = event["action"]
                action_kind = action.get("kind", "Unknown")
                markdown_lines.append(f"**Action:** `{action_kind}`\n")

                if action_kind == "FileEditorAction":
                    command = action.get("command", "")
                    path = action.get("path", "")
                    markdown_lines.append(f"- Command: `{command}`")
                    markdown_lines.append(f"- Path: `{path}`")
                    if command == "str_replace":
                        if "old_str" in action and action["old_str"]:
                            markdown_lines.append(
                                f"- Old content:\n```\n{action['old_str']}\n```"
                            )
                        if "new_str" in action and action["new_str"]:
                            markdown_lines.append(
                                f"- New content:\n```\n{action['new_str']}\n```"
                            )
                    elif command == "create" and action.get("file_text"):
                        content = _truncate_text(action["file_text"], 1000)
                        markdown_lines.append(f"- File content:\n```\n{content}\n```")

                elif action_kind == "TerminalAction":
                    command = action.get("command", "")
                    markdown_lines.append(f"- Command: `{command}`")

                elif action_kind == "BrowserAction":
                    if "url" in action:
                        markdown_lines.append(f"- URL: `{action['url']}`")

                elif action_kind == "ThinkAction":
                    if "thought" in action:
                        thought = _truncate_text(action["thought"], 2000)
                        markdown_lines.append(f"- Thought:\n```\n{thought}\n```")

                elif action_kind == "MCPToolAction":
                    tool_name = event.get("tool_name", "")
                    if tool_name:
                        markdown_lines.append(f"- Tool: `{tool_name}`")
                    data = action.get("data", {})
                    if data:
                        data_str = json.dumps(data, indent=2)
                        markdown_lines.append(f"- Arguments:\n```json\n{data_str}\n```")

                elif action_kind == "FinishAction":
                    message = action.get("message", "")
                    if message:
                        markdown_lines.append(f"- Message: {message}")

                elif action_kind == "DelegateAction":
                    command = action.get("command", "")
                    if command:
                        markdown_lines.append(f"- Command: `{command}`")
                    ids = action.get("ids") or []
                    if ids:
                        markdown_lines.append(
                            f"- Agent IDs: {', '.join(f'`{agent_id}`' for agent_id in ids)}"
                        )
                    tasks = action.get("tasks") or {}
                    if tasks:
                        for agent_id, task in tasks.items():
                            markdown_lines.append(
                                f"- Task for `{agent_id}`: {task}"
                            )
                    if command == "delegate" and tasks:
                        pending_delegate_agents = [
                            agent_id
                            for agent_id in tasks
                            if subagents
                            and agent_id in subagents
                            and (
                                rendered_subagents is None
                                or agent_id not in rendered_subagents
                            )
                        ]
                    else:
                        pending_delegate_agents = []

                markdown_lines.append("")

        elif event_kind == "ObservationEvent":
            if "observation" in event and event["observation"]:
                obs = event["observation"]
                obs_kind = obs.get("kind", "Unknown")

                markdown_lines.append("**Observation:**\n")

                if obs_kind == "MCPToolObservation":
                    if "content" in obs and isinstance(obs["content"], list):
                        for item in obs["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                if text.startswith("[Tool ") and text.endswith(
                                    " executed.]"
                                ):
                                    continue
                                text = _truncate_text(text, 1000)
                                markdown_lines.append("```")
                                markdown_lines.append(text)
                                markdown_lines.append("```\n")

                elif obs_kind == "FileEditorObservation":
                    if "content" in obs and isinstance(obs["content"], list):
                        for item in obs["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content = _truncate_text(item.get("text", ""), 1000)
                                markdown_lines.append("```")
                                markdown_lines.append(text_content)
                                markdown_lines.append("```\n")
                                break
                    else:
                        obs_str = _truncate_text(str(obs), 500)
                        markdown_lines.append("```")
                        markdown_lines.append(obs_str)
                        markdown_lines.append("```\n")

                elif obs_kind == "TerminalObservation":
                    if "content" in obs and isinstance(obs["content"], list):
                        for item in obs["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content = _truncate_text(item.get("text", ""), 1000)
                                markdown_lines.append("```")
                                markdown_lines.append(text_content)
                                markdown_lines.append("```")
                                break

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

                else:
                    text_content = None
                    if "content" in obs and isinstance(obs["content"], list):
                        for item in obs["content"]:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_content = item.get("text", "")
                                break

                    if text_content:
                        text_content = _truncate_text(text_content, 1000)
                        markdown_lines.append("```")
                        markdown_lines.append(text_content)
                        markdown_lines.append("```\n")
                    else:
                        obs_str = _truncate_text(str(obs), 500)
                        markdown_lines.append("```")
                        markdown_lines.append(obs_str)
                        markdown_lines.append("```\n")

                if pending_delegate_agents:
                    for agent_id in pending_delegate_agents:
                        if subagents is None or agent_id not in subagents:
                            continue
                        _append_subagent_trace_markdown(
                            markdown_lines,
                            agent_id,
                            subagents[agent_id],
                            heading_level=5,
                        )
                        if rendered_subagents is not None:
                            rendered_subagents.add(agent_id)
                    pending_delegate_agents = []

        elif event_kind == "AgentErrorEvent":
            error_msg = event.get("error", "Unknown error")
            tool_name = event.get("tool_name", "Unknown tool")

            markdown_lines.append(f"**Error ({tool_name}):**\n")
            markdown_lines.append("```")
            markdown_lines.append(error_msg)
            markdown_lines.append("```\n")

        else:
            assert False, f"Unhandled event kind: {event_kind}"


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
    subagents = json_data.get("subagents", {})
    rendered_subagents: set[str] = set()
    _append_event_stream_markdown(
        markdown_lines,
        json_data["events"],
        subagents=subagents,
        rendered_subagents=rendered_subagents,
    )

    remaining_subagents = [
        (agent_id, payload)
        for agent_id, payload in subagents.items()
        if agent_id not in rendered_subagents
    ]
    if remaining_subagents:
        markdown_lines.append("\n---\n\n### Additional Subagent Traces\n")
        for agent_id, payload in remaining_subagents:
            _append_subagent_trace_markdown(
                markdown_lines,
                agent_id,
                payload,
                heading_level=4,
            )

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
