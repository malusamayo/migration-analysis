import argparse
import json
from pathlib import Path

from build_triplet_coverage import build_curated_run_eval_rows
from build_triplet_coverage import load_summary


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def build_run_meta(row: dict) -> dict:
    linkage = row["linkage"]
    return {
        "index": row["batch_index"],
        "task_id": row["task_id"],
        "model_name": row["model_name"],
        "prompt_name": row["prompt_name"],
        "max_examples": row["max_examples"],
        "n_responses": row["expected_trials"],
        "rollout_version": row["rollout_version"],
        "effective_rollout_version": row["effective_rollout_version"],
        "agent_file": row["agent_file"],
        "run_dir": linkage["run_dir"],
        "run_config_path": linkage["run_config_path"],
        "collect_log_path": linkage["collect_log_path"],
        "evaluate_log_path": linkage["evaluate_log_path"],
        "source_batch_dir": row["source_batch_dir"],
    }


def build_payload(summary_path: Path) -> dict:
    rows = build_curated_run_eval_rows(load_summary(summary_path))
    return {
        "description": "Curated run/eval inputs used by the replication package.",
        "runs": [
            {
                "batch_dir": row["batch_dir"],
                "run_meta": build_run_meta(row),
                "launch_meta": row["launch"],
            }
            for row in rows
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=PACKAGE_DIR / "outputs" / "data" / "summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_DIR / "data" / "curated_run_eval_inputs.json",
    )
    args = parser.parse_args()

    payload = build_payload(args.summary_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(len(payload["runs"]))


if __name__ == "__main__":
    main()
