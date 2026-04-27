import json
import re
from pathlib import Path
from typing import Any


def _extract_text_blocks(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    text_blocks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            text_blocks.append(text)
    if not text_blocks:
        return None
    return "".join(text_blocks)


def _extract_user_prompt(trace_payload: Any) -> str:
    events = trace_payload if isinstance(trace_payload, list) else trace_payload["events"]
    for event in events:
        llm_message = event.get("llm_message")
        if not isinstance(llm_message, dict):
            continue
        if llm_message.get("role") != "user":
            continue
        prompt = _extract_text_blocks(llm_message.get("content"))
        if prompt is not None:
            return prompt
    raise ValueError("Could not find user prompt in teacher trajectory payload")


class TeacherTrajectoryStore:
    def __init__(self, trajectory_dir: str):
        self.trajectory_dir = Path(trajectory_dir)
        if not self.trajectory_dir.is_dir():
            raise ValueError(f"Teacher trajectory directory does not exist: {trajectory_dir}")
        self._by_dataset_index: dict[int, Path] = {}
        self._prompt_by_dataset_index: dict[int, str] = {}
        self._by_prompt: dict[str, list[Path]] = {}
        self._index()

    def _index(self) -> None:
        for logs_dir in sorted(self.trajectory_dir.iterdir()):
            if not logs_dir.is_dir():
                continue
            match = re.fullmatch(r"example(\d+)_rollout\d+_logs", logs_dir.name)
            if match is None:
                continue

            dataset_index = int(match.group(1))
            trace_path = self._select_trace_path(logs_dir)
            prompt = _extract_user_prompt(json.loads(trace_path.read_text()))
            self._by_dataset_index.setdefault(dataset_index, trace_path)
            self._prompt_by_dataset_index.setdefault(dataset_index, prompt)
            prompt_matches = self._by_prompt.setdefault(prompt, [])
            prompt_matches.append(trace_path)

    def _select_trace_path(self, logs_dir: Path) -> Path:
        raw_traces = sorted(logs_dir.glob("raw_trace_*.json"))
        if raw_traces:
            return raw_traces[0]

        traces = sorted(logs_dir.glob("trace_*.json"))
        if traces:
            return traces[0]

        raise ValueError(f"No trace JSON found in teacher trajectory logs dir: {logs_dir}")

    def resolve(self, example: dict[str, Any]) -> str | None:
        dataset_index = example.get("_dataset_index")
        prompt = example.get("prompt")

        if isinstance(dataset_index, int):
            trace_path = self._by_dataset_index.get(dataset_index)
            if trace_path is not None:
                teacher_prompt = self._prompt_by_dataset_index[dataset_index]
                if not isinstance(prompt, str) or prompt == teacher_prompt:
                    return str(trace_path)

        if not isinstance(prompt, str):
            return None
        prompt_matches = self._by_prompt.get(prompt, [])
        if not prompt_matches:
            return None
        if len(prompt_matches) > 1:
            raise ValueError(
                "Multiple teacher trajectories matched the same prompt; "
                f"cannot disambiguate example with dataset index {dataset_index}"
            )
        return str(prompt_matches[0])
