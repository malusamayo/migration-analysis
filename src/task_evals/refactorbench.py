"""Evaluation for the refactorbench task."""

import subprocess
import tempfile

_PYTHON311 = subprocess.check_output(["uv", "python", "find", "3.11"], text=True, stderr=subprocess.DEVNULL).strip()
from pathlib import Path
from typing import Any, Dict, Optional

import dspy


class ShortenMessageSignature(dspy.Signature):
    """Shorten a failure message to a single line within a max length."""

    message = dspy.InputField(desc="Error or failure message to shorten.")
    max_length = dspy.InputField(desc="Maximum length in characters.")
    shortened = dspy.OutputField(desc="Shortened single-line message within max_length.")


class ShortenMessage(dspy.Module):
    """DSPy module to shorten error messages."""

    def __init__(self):
        super().__init__()
        self.shorten = dspy.Predict(ShortenMessageSignature)

    def forward(self, message: str, max_length: int):
        return self.shorten(message=message, max_length=max_length)


def _fallback_shorten(message: str, max_length: int = 220) -> str:
    """Return a deterministic single-line fallback summary."""

    lines = [line.strip() for line in message.splitlines() if line.strip()]
    priority_fragments = []
    for line in lines:
        if line.startswith("FAIL:"):
            priority_fragments.append(line)
        elif "AssertionError:" in line:
            priority_fragments.append(line)
        elif line.startswith("AssertionError"):
            priority_fragments.append(line)
        elif line.startswith("E   "):
            priority_fragments.append(line[4:])
        elif "FAILED (" in line:
            priority_fragments.append(line)

    if priority_fragments:
        collapsed = " | ".join(priority_fragments)
    else:
        collapsed = " ".join(message.split())

    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[: max_length - 3] + "..."


def _summarize_failure(
    lm: Optional[dspy.LM],
    stdout: str,
    stderr: str,
    returncode: Optional[int] = None,
    max_length: int = 220,
) -> str:
    """Summarize a failed evaluation script run."""

    parts = []
    if returncode is not None:
        parts.append(f"Return code: {returncode}")
    if stderr.strip():
        parts.append(stderr.strip())
    if stdout.strip():
        parts.append(stdout.strip())

    combined = "\n".join(parts).strip()
    if not combined:
        return "Evaluation script failed with no stdout/stderr."

    if lm is None:
        return _fallback_shorten(combined, max_length=max_length)

    try:
        with dspy.context(lm=lm):
            predictor = ShortenMessage()
            shortened = predictor(message=combined, max_length=max_length).shortened
        return _fallback_shorten(shortened, max_length=max_length)
    except Exception:
        return _fallback_shorten(combined, max_length=max_length)



def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    workspace_path = Path(workspace_dir)
    example_id = example["id"]
    eval_script = example.get("eval_script", "")

    if not workspace_path.exists():
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "success": False,
            "feedback": "Workspace directory does not exist.",
            "stdout": "",
            "stderr": "",
        }

    if not eval_script:
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "success": False,
            "feedback": "Evaluation script missing from dataset example.",
            "stdout": "",
            "stderr": "",
        }

    try:
        with tempfile.TemporaryDirectory(
            dir=workspace_path,
            prefix=".refactorbench_eval_",
        ) as eval_dir:
            eval_dir_path = Path(eval_dir)
            local_test_path = eval_dir_path / "eval_script.py"
            local_test_path.write_text(eval_script, encoding="utf-8")

            result = subprocess.run(
                [_PYTHON311, str(local_test_path.name)],
                cwd=eval_dir_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired as exc:
        feedback = _summarize_failure(
            lm,
            exc.stdout or "",
            exc.stderr or f"Evaluation script timed out after {timeout}s.",
        )
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "repo_name": example["repo_name"],
            "score": 0.0,
            "success": False,
            "feedback": feedback,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }

    success = result.returncode == 0
    feedback = (
        "Evaluation script passed."
        if success
        else _summarize_failure(lm, result.stdout, result.stderr, result.returncode)
    )
    return {
        "workspace_dir": workspace_dir,
        "example_id": example_id,
        "repo_name": example["repo_name"],
        "score": 1.0 if success else 0.0,
        "success": success,
        "feedback": feedback,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
