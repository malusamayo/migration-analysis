"""Workspace setup for the RefactorBench task."""

import json
import shutil
from pathlib import Path


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    workspace = Path(workspace_dir)
    log_path = Path(log_dir)
    repo_root = Path(example["repo_path"])

    shutil.copytree(repo_root, workspace, dirs_exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    task_context = {
        "task_id": example["id"],
        "repo_name": example["repo_name"],
        "problem_statement": example["problem_statement"],
        "problem_path": example["problem_path"],
    }
    (workspace / "task_context.json").write_text(
        json.dumps(task_context, indent=2),
        encoding="utf-8",
    )
