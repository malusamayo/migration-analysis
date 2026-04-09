import copy
import difflib
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

from openhands.sdk import Agent


def execute_agent_candidate(
    code: str,
    base_dir: str,
    lm_model: str,
    seed_prompt: str,
) -> Agent:
    """Execute candidate code and return the constructed Agent."""
    namespace = {}
    exec(code, namespace)
    return namespace["build_agent"](base_dir, lm_model, seed_prompt)


def extract_workspace_scripts(code: str) -> dict[str, str]:
    """Execute candidate code and extract optional workspace scripts."""
    namespace = {}
    exec(code, namespace)
    fn = namespace.get("get_workspace_scripts")
    if fn is None:
        return {}
    return fn()


def _validate_worker(code: str, lm_model: str, seed_prompt: str) -> tuple[bool, str]:
    """Subprocess worker for validate_agent_candidate."""
    import tempfile

    from openhands.sdk import Agent as _Agent
    from openhands.sdk import Conversation as _Conversation

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            namespace = {}
            exec(code, namespace)
            agent = namespace["build_agent"](tmp_dir, lm_model, seed_prompt)
            if not isinstance(agent, _Agent):
                return False, f"build_agent returned {type(agent).__name__}, expected Agent"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

        conversation = None
        try:
            conversation = _Conversation(agent=agent, workspace=tmp_dir)
            conversation.send_message("test")
        except Exception as e:
            return False, f"{type(e).__name__} during conversation init: {e}"
        finally:
            if conversation is not None:
                try:
                    conversation.close()
                except Exception:
                    pass

        return True, ""


def validate_agent_candidate(
    code: str,
    lm_model: str,
    seed_prompt: str,
) -> tuple[bool, str]:
    """Validate candidate code by compiling, executing, and building the Agent."""
    try:
        compile(code, "agent.py", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        future = pool.submit(_validate_worker, code, lm_model, seed_prompt)
        return future.result()


def format_eval_feedback(output: dict, score: float) -> str:
    """Build rich evaluation feedback from an eval result dict."""
    parts = [f"Score: {score}"]

    feedback = output.get("feedback")
    if feedback:
        parts.append(feedback)

    if "error" in output and output["error"]:
        parts.append(f"Error: {output['error']}")

    test_output = output.get("test_output")
    if test_output:
        for fr in test_output:
            if fr.get("status") in ("failed", "error", "timeout"):
                file_line = f"[{fr['status'].upper()}] {fr.get('file', '?')}"
                stderr = fr.get("stderr", "").strip()
                stdout = fr.get("stdout", "").strip()
                if stderr:
                    file_line += f"\n{stderr[:500]}"
                elif stdout:
                    file_line += f"\n{stdout[:500]}"
                parts.append(file_line)

    return "\n".join(parts)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def empty_cost_bucket() -> dict:
    return {
        "accumulated_cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }


def add_to_cost_bucket(bucket: dict, metrics: dict) -> None:
    bucket["accumulated_cost"] += metrics.get("accumulated_cost", 0.0)
    usage = metrics.get("accumulated_token_usage") or {}
    bucket["prompt_tokens"] += usage.get("prompt_tokens", 0)
    bucket["completion_tokens"] += usage.get("completion_tokens", 0)
    bucket["cache_read_tokens"] += usage.get("cache_read_tokens", 0)
    bucket["cache_write_tokens"] += usage.get("cache_write_tokens", 0)
    bucket["reasoning_tokens"] += usage.get("reasoning_tokens", 0)


def extract_dspy_cost(lm, history_len_before: int) -> Optional[dict]:
    """Extract cost incurred by a dspy LM since history_len_before."""
    if lm is None or not hasattr(lm, "history"):
        return None
    new_calls = lm.history[history_len_before:]
    if not new_calls:
        return None
    cost = sum(c.get("cost", 0.0) or 0.0 for c in new_calls)
    prompt_tokens = 0
    completion_tokens = 0
    for c in new_calls:
        usage = c.get("usage") or {}
        prompt_tokens += usage.get("prompt_tokens", 0) or 0
        completion_tokens += usage.get("completion_tokens", 0) or 0
    return {
        "accumulated_cost": cost,
        "accumulated_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        },
    }


def build_code_diff(old_text: str, new_text: str, max_lines: int = 120) -> str:
    diff_lines = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile="current.py",
            tofile="proposed.py",
            lineterm="",
        )
    )
    if not diff_lines:
        return "(no code changes)"
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines] + ["... diff truncated ..."]
    return "\n".join(diff_lines)


def summarize_score_changes(
    batch: Sequence[dict[str, Any]],
    before_scores: Sequence[float],
    after_scores: Sequence[float],
    limit: int = 3,
) -> list[dict[str, Any]]:
    changes = []
    for example, before, after in zip(batch, before_scores, after_scores):
        delta = after - before
        if abs(delta) < 1e-9:
            continue
        changes.append(
            {
                "task_input": example.get("prompt", "")[:300],
                "before": before,
                "after": after,
                "delta": delta,
            }
        )

    changes.sort(key=lambda item: (item["delta"], abs(item["delta"])))
    return changes[:limit]


def resolve_markdown_path(markdown_path: str, docs_dir: str) -> str:
    """Resolve a markdown path from absolute, cwd-relative, or docs-relative input."""
    candidate = Path(markdown_path)
    if candidate.is_absolute():
        resolved = candidate
    else:
        cwd_relative = Path.cwd() / candidate
        docs_relative = Path(docs_dir) / candidate
        if cwd_relative.exists():
            resolved = cwd_relative
        else:
            resolved = docs_relative
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Markdown file not found: {markdown_path}")
    return str(resolved)


def clone_example(example: dict) -> dict:
    return copy.deepcopy(example)
