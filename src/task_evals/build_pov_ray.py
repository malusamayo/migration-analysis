"""
Evaluation functions for build-pov-ray task.

After the agent run, test.sh is executed inside the same Docker container
(via runner.py post-agent hook), which writes reward.txt and ctrf.json to
workspace_dir/verifier_logs/. This module reads those results.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import dspy


def run_single_instance_eval(
    lm: dspy.LM,
    workspace_dir: str,
    example: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Read build-pov-ray verifier results from workspace_dir/verifier_logs/.

    The verifier (test.sh) runs inside the Docker container after the agent
    finishes and writes:
      - reward.txt: "1" (pass) or "0" (fail)
      - ctrf.json:  pytest CTRF report with per-test details

    Args:
        lm: DSPy LM (unused, kept for interface compatibility)
        workspace_dir: Path to the workspace directory
        example: Optional example dict (unused)

    Returns:
        Dict with score (0.0 or 1.0), workspace_dir, and optional ctrf details.
    """
    workspace_path = Path(workspace_dir)
    verifier_logs = workspace_path / "verifier_logs"
    reward_file = verifier_logs / "reward.txt"
    ctrf_file = verifier_logs / "ctrf.json"

    if not reward_file.exists():
        return {
            "workspace_dir": str(workspace_dir),
            "score": 0.0,
            "error": "Verifier did not run (reward.txt missing). Agent may have timed out or test.sh was not executed.",
        }

    score = float(reward_file.read_text().strip())

    result = {
        "workspace_dir": str(workspace_dir),
        "score": score,
    }

    if ctrf_file.exists():
        with open(ctrf_file) as f:
            result["ctrf"] = json.load(f)

    return result
