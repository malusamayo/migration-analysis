"""Workspace setup for the replicatorbench task.

Directory layout after setup:
    workspace_dir/
        hypothesis.txt          ← hypothesis being tested (orientation only, no numbers)
        replication_data/       ← original analysis scripts (.R, .do) and data files
    log_dir/
        paper.pdf               ← original paper (answer key, NOT visible to agent)
        groundtruth/
            expected_post_registration.json    ← primary ground truth
            expected_post_registration_2.json  ← alternative variant (if present)

The paper is intentionally withheld from the workspace: the agent must reproduce
the numerical result by running/translating the analysis scripts, not by reading
the paper's reported values.
"""
import json
import re
import shutil
from pathlib import Path


def _extract_hypotheses(initial_details: str) -> str:
    """Return only the [HYPOTHESES] / [HYPOTHESIS] section of initial_details.

    Section headers are assumed to be [ALL-CAPS WORD] at the start of a line,
    so inline brackets like '[artemisinin]' are not treated as section breaks.
    """
    m = re.search(
        r"\[HYPOTHES[EI]S\]\s*(.*?)(?=\n\s*\[[A-Z]+\]|\Z)",
        initial_details,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return initial_details.strip()


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    workspace = Path(workspace_dir)
    data_root = Path(example["data_root"])
    log_path = Path(log_dir)

    # Hypothesis only — no claim numbers
    hypothesis_text = _extract_hypotheses(example["initial_details"])
    (workspace / "hypothesis.txt").write_text(hypothesis_text, encoding="utf-8")

    # Replication scripts and data
    repl_dir = workspace / "replication_data"
    repl_dir.mkdir(exist_ok=True)
    for rel_path in example.get("replication_data_files", []):
        src = data_root / rel_path
        if src.exists():
            shutil.copy2(src, repl_dir / src.name)

    # Paper goes to log_dir (answer key, invisible to agent)
    src_pdf = data_root / example["original_paper_pdf"]
    if src_pdf.exists():
        shutil.copy2(src_pdf, log_path / "paper.pdf")

    # Ground truth
    gt_dir = log_path / "groundtruth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    gt = example.get("expected_post_registration")
    if gt:
        (gt_dir / "expected_post_registration.json").write_text(
            json.dumps(gt, indent=2), encoding="utf-8"
        )

    gt2 = example.get("expected_post_registration_2")
    if gt2:
        (gt_dir / "expected_post_registration_2.json").write_text(
            json.dumps(gt2, indent=2), encoding="utf-8"
        )
