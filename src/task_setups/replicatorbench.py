"""Workspace setup for the replicatorbench task.

The redesigned task is closer to the paper's staged workflow:
    workspace_dir/
        task_context.json       ← mode, available resources, required artifacts
        initial_details.txt     ← claim/hypothesis hints
        original_paper.pdf      ← primary extraction source
        replication_data/       ← data files and, optionally, native code
    log_dir/
        groundtruth/            ← expected extraction refs and human documents

Code access is controlled by the REPLICATORBENCH_CODE_ACCESS environment
variable:
    - "code_and_data" (default): include native replication code and data
    - "data_only": include data files but withhold native analysis code
"""
import json
import os
import shutil
from pathlib import Path


CODE_FILE_EXTENSIONS = {
    ".r",
    ".R",
    ".do",
    ".py",
    ".ipynb",
    ".jl",
    ".m",
    ".sas",
    ".stata",
}


def _code_access_mode() -> str:
    mode = os.getenv("REPLICATORBENCH_CODE_ACCESS", "code_and_data").strip().lower()
    if mode not in {"code_and_data", "data_only"}:
        return "code_and_data"
    return mode


def _copy_tree_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _should_copy_replication_file(path: Path, mode: str) -> bool:
    if mode == "code_and_data":
        return True
    return path.suffix not in CODE_FILE_EXTENSIONS


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    workspace = Path(workspace_dir)
    data_root = Path(example["data_root"])
    study_id = str(example["study_id"])
    study_root = data_root / study_id
    input_root = study_root / "input"
    gt_root = study_root / "gt"
    log_path = Path(log_dir)
    mode = _code_access_mode()

    workspace.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    # Context files visible to the agent.
    initial_details_path = input_root / "initial_details.txt"
    if initial_details_path.exists():
        shutil.copy2(initial_details_path, workspace / "initial_details.txt")
    else:
        (workspace / "initial_details.txt").write_text(
            example.get("initial_details", ""),
            encoding="utf-8",
        )

    paper_path = input_root / "original_paper.pdf"
    if paper_path.exists():
        shutil.copy2(paper_path, workspace / "original_paper.pdf")

    repl_dir = workspace / "replication_data"
    repl_dir.mkdir(exist_ok=True)
    for rel_path in example.get("replication_data_files", []):
        src = data_root / rel_path
        if src.exists() and _should_copy_replication_file(src, mode):
            shutil.copy2(src, repl_dir / src.name)

    task_context = {
        "study_id": study_id,
        "code_access_mode": mode,
        "available_resources": {
            "paper": (workspace / "original_paper.pdf").exists(),
            "initial_details": True,
            "replication_data_files": sorted(p.name for p in repl_dir.iterdir()),
        },
        "required_outputs": [
            "post_registration.json",
            "replication_info.json",
            "execution_results.json",
            "interpret_results.json",
        ],
        "notes": [
            "Use initial_details.txt for the claim statement and hypotheses fields.",
            "Use original_paper.pdf for all other extraction fields.",
            "In data_only mode, native replication code is intentionally withheld.",
            "Ground-truth preregistration and report documents exist only for evaluation.",
        ],
    }
    (workspace / "task_context.json").write_text(
        json.dumps(task_context, indent=2),
        encoding="utf-8",
    )

    # Preserve raw ground-truth files for evaluation and sanity checks.
    gt_dir = log_path / "groundtruth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    if gt_root.exists():
        for item in gt_root.iterdir():
            if item.is_file() or item.is_symlink():
                _copy_tree_file(item, gt_dir / item.name)
