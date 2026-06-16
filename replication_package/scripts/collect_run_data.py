import argparse
import csv
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PACKAGE_DIR.parent
WORKSPACE_RE = re.compile(r"example(?P<example_id>\d+)_rollout(?P<rollout_id>\d+)$")
SPLIT_WEIGHTS = {"train": 2, "val": 2, "test": 6}
SUMMARY_SPLITS = ("full", "train", "val", "test")
TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)
LATENCY_FIELDS = (
    "end_to_end_latency_sec",
    "model_inference_latency_sec",
)
LATENCY_AVG_FIELDS = (
    "avg_end_to_end_latency_sec",
    "avg_model_inference_latency_sec",
)
COUNT_FIELDS = ("model_inference_calls",)
SUMMARY_VALUE_FIELDS = (
    "score",
    "cost",
    *TOKEN_FIELDS,
    *LATENCY_FIELDS,
    *LATENCY_AVG_FIELDS,
    *COUNT_FIELDS,
)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_DIR / path


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_batch_dirs(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [resolve_repo_path(Path(line.strip())) for line in lines if line.strip()]


def parse_workspace_key(workspace_dir: str) -> tuple[int, int]:
    match = WORKSPACE_RE.search(workspace_dir)
    if match is None:
        raise ValueError(f"Could not parse example/rollout ids from workspace_dir={workspace_dir!r}")
    return int(match.group("example_id")), int(match.group("rollout_id"))


def compute_split_sizes(max_examples: int) -> dict[str, int]:
    total_weight = sum(SPLIT_WEIGHTS.values())
    train_examples = max_examples * SPLIT_WEIGHTS["train"] // total_weight
    val_examples = max_examples * SPLIT_WEIGHTS["val"] // total_weight
    test_examples = max_examples - train_examples - val_examples
    return {
        "train": train_examples,
        "val": val_examples,
        "test": test_examples,
    }


def compute_split_ranges(max_examples: int) -> dict[str, tuple[int, int]]:
    split_sizes = compute_split_sizes(max_examples)
    train_end = split_sizes["train"]
    val_end = train_end + split_sizes["val"]
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, max_examples),
    }


def split_for_example(example_id: int, max_examples: int) -> str:
    for split_name, (start, end) in compute_split_ranges(max_examples).items():
        if start <= example_id < end:
            return split_name
    raise ValueError(f"example_id={example_id} is out of range for max_examples={max_examples}")


def make_missing_summary(expected_trials: int) -> dict[str, Any]:
    summary = {
        "examples": None,
        "records": None,
        "score": None,
        "cost": None,
        "trial_scores": [None] * expected_trials,
        "trial_costs": [None] * expected_trials,
    }
    for token_field in TOKEN_FIELDS:
        summary[token_field] = None
        summary[f"trial_{token_field}"] = [None] * expected_trials
    for latency_field in LATENCY_FIELDS:
        summary[latency_field] = None
        summary[f"trial_{latency_field}"] = [None] * expected_trials
    for latency_avg_field in LATENCY_AVG_FIELDS:
        summary[latency_avg_field] = None
        summary[f"trial_{latency_avg_field}"] = [None] * expected_trials
    for count_field in COUNT_FIELDS:
        summary[count_field] = None
        summary[f"trial_{count_field}"] = [None] * expected_trials
    return summary


def event_timestamp(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(event["timestamp"])


def compute_end_to_end_latency(events: list[dict[str, Any]]) -> float:
    return (event_timestamp(events[-1]) - event_timestamp(events[0])).total_seconds()


def compute_model_inference_latency(response_latencies: list[dict[str, Any]]) -> float:
    return sum(float(response_latency["latency"]) for response_latency in response_latencies)


def summarize_records(records: list[dict[str, Any]], expected_trials: int) -> dict[str, Any]:
    by_rollout: dict[int, list[dict[str, Any]]] = defaultdict(list)
    example_ids = set()
    totals = {token_field: 0 for token_field in TOKEN_FIELDS}

    for record in records:
        rollout_id = int(record["rollout_id"])
        by_rollout[rollout_id].append(record)
        example_ids.add(int(record["example_id"]))
        for token_field in TOKEN_FIELDS:
            totals[token_field] += int(record[token_field])

    rollout_ids = sorted(by_rollout)
    if rollout_ids != list(range(expected_trials)):
        raise ValueError(f"Expected rollout ids 0..{expected_trials - 1}, got {rollout_ids}")
    rollout_sizes = {rollout_id: len(by_rollout[rollout_id]) for rollout_id in rollout_ids}
    if len(set(rollout_sizes.values())) != 1:
        raise ValueError(f"Mismatched record counts across rollouts: {rollout_sizes}")

    total_score = sum(float(record["score"]) for record in records)
    total_cost = sum(float(record["cost"]) for record in records)
    total_end_to_end_latency = sum(float(record["end_to_end_latency_sec"]) for record in records)
    total_model_inference_latency = sum(
        float(record["model_inference_latency_sec"]) for record in records
    )
    total_model_inference_calls = sum(int(record["model_inference_calls"]) for record in records)

    summary = {
        "examples": len(example_ids),
        "records": len(records),
        "score": total_score / len(records),
        "cost": total_cost,
        "end_to_end_latency_sec": total_end_to_end_latency,
        "model_inference_latency_sec": total_model_inference_latency,
        "avg_end_to_end_latency_sec": total_end_to_end_latency / len(records),
        "avg_model_inference_latency_sec": total_model_inference_latency / len(records),
        "model_inference_calls": total_model_inference_calls,
        "trial_scores": [],
        "trial_costs": [],
        "trial_end_to_end_latency_sec": [],
        "trial_model_inference_latency_sec": [],
        "trial_avg_end_to_end_latency_sec": [],
        "trial_avg_model_inference_latency_sec": [],
        "trial_model_inference_calls": [],
    }

    for rollout_id in rollout_ids:
        rollout_records = by_rollout[rollout_id]
        summary["trial_scores"].append(
            sum(float(record["score"]) for record in rollout_records) / len(rollout_records)
        )
        summary["trial_costs"].append(sum(float(record["cost"]) for record in rollout_records))
        trial_end_to_end_latency = sum(
            float(record["end_to_end_latency_sec"]) for record in rollout_records
        )
        trial_model_inference_latency = sum(
            float(record["model_inference_latency_sec"]) for record in rollout_records
        )
        summary["trial_end_to_end_latency_sec"].append(trial_end_to_end_latency)
        summary["trial_model_inference_latency_sec"].append(trial_model_inference_latency)
        summary["trial_avg_end_to_end_latency_sec"].append(
            trial_end_to_end_latency / len(rollout_records)
        )
        summary["trial_avg_model_inference_latency_sec"].append(
            trial_model_inference_latency / len(rollout_records)
        )
        summary["trial_model_inference_calls"].append(
            sum(int(record["model_inference_calls"]) for record in rollout_records)
        )

    for token_field in TOKEN_FIELDS:
        summary[token_field] = totals[token_field]
        summary[f"trial_{token_field}"] = [
            sum(int(record[token_field]) for record in by_rollout[rollout_id])
            for rollout_id in rollout_ids
        ]

    return summary


def build_workspace_records(
    run_results_path: Path,
    eval_results_path: Path,
    max_examples: int,
) -> list[dict[str, Any]]:
    run_rows = load_json(run_results_path)
    eval_rows = load_yaml(eval_results_path)
    eval_by_key = {
        parse_workspace_key(str(row["workspace_dir"])): row
        for row in eval_rows
    }

    records = []
    for run_row in run_rows:
        example_id = int(run_row["example_id"])
        rollout_id = int(run_row["rollout_id"])
        eval_row = eval_by_key[(example_id, rollout_id)]
        metrics = run_row["run_result"]["metrics"]
        events = run_row["run_result"]["events"]
        response_latencies = metrics["response_latencies"]
        token_usage = metrics["accumulated_token_usage"]
        records.append(
            {
                "example_id": example_id,
                "rollout_id": rollout_id,
                "split": split_for_example(example_id, max_examples),
                "workspace_dir": str(eval_row["workspace_dir"]),
                "conversation_id": run_row["run_result"]["conversation_id"],
                "score": None if eval_row.get("score") is None else float(eval_row["score"]),
                "cost": float(metrics["accumulated_cost"]),
                "end_to_end_latency_sec": compute_end_to_end_latency(events),
                "model_inference_latency_sec": compute_model_inference_latency(response_latencies),
                "model_inference_calls": len(response_latencies),
                "prompt_tokens": int(token_usage["prompt_tokens"]),
                "completion_tokens": int(token_usage["completion_tokens"]),
                "cache_read_tokens": int(token_usage["cache_read_tokens"]),
                "cache_write_tokens": int(token_usage["cache_write_tokens"]),
                "reasoning_tokens": int(token_usage["reasoning_tokens"]),
                "success": eval_row.get("success"),
                "tests_passed": eval_row.get("tests_passed"),
                "tests_failed": eval_row.get("tests_failed"),
                "tests_total": eval_row.get("tests_total"),
                "error": run_row["run_result"]["error"],
            }
        )

    records.sort(key=lambda record: (record["example_id"], record["rollout_id"]))
    return records


def launch_results_by_index(batch_dir: Path) -> dict[int, dict[str, Any]]:
    launch_results_path = batch_dir / "launch_results.json"
    if not launch_results_path.exists():
        return {}
    launch_rows = load_json(launch_results_path)
    return {int(row["index"]): row for row in launch_rows}


def summarize_run(batch_dir: Path, run_meta: dict[str, Any], launch_meta: dict[str, Any] | None) -> dict[str, Any]:
    max_examples = int(run_meta["max_examples"])
    expected_trials = int(run_meta["n_responses"])
    run_dir = resolve_repo_path(Path(run_meta["run_dir"]))
    run_results_path = run_dir / "run.json"
    eval_results_path = run_dir / "eval_results.yaml"
    collect_log_path = resolve_repo_path(Path(run_meta["collect_log_path"]))
    evaluate_log_path = resolve_repo_path(Path(run_meta["evaluate_log_path"]))
    run_config_path = resolve_repo_path(Path(run_meta["run_config_path"]))

    base_summary = {
        "batch_name": batch_dir.name,
        "batch_dir": str(batch_dir.relative_to(REPO_DIR)),
        "batch_json_path": str((batch_dir / "batch.json").relative_to(REPO_DIR)),
        "launch_results_path": str((batch_dir / "launch_results.json").relative_to(REPO_DIR))
        if (batch_dir / "launch_results.json").exists()
        else None,
        "source_batch_dir": run_meta.get("source_batch_dir"),
        "batch_index": int(run_meta["index"]),
        "task_id": run_meta["task_id"],
        "model_name": run_meta["model_name"],
        "prompt_name": run_meta["prompt_name"],
        "max_examples": max_examples,
        "expected_trials": expected_trials,
        "found_trials": None,
        "rollout_version": run_meta["rollout_version"],
        "effective_rollout_version": run_meta.get("effective_rollout_version", run_dir.name),
        "agent_file": run_meta.get("agent_file"),
        "linkage": {
            "run_dir": str(run_dir.relative_to(REPO_DIR)),
            "run_results_path": str(run_results_path.relative_to(REPO_DIR)),
            "eval_results_path": str(eval_results_path.relative_to(REPO_DIR)),
            "run_config_path": str(run_config_path.relative_to(REPO_DIR)),
            "collect_log_path": str(collect_log_path.relative_to(REPO_DIR)),
            "evaluate_log_path": str(evaluate_log_path.relative_to(REPO_DIR)),
            "run_dir_exists": run_dir.exists(),
            "run_results_path_exists": run_results_path.exists(),
            "eval_results_path_exists": eval_results_path.exists(),
            "run_config_path_exists": run_config_path.exists(),
            "collect_log_path_exists": collect_log_path.exists(),
            "evaluate_log_path_exists": evaluate_log_path.exists(),
        },
        "launch": launch_meta,
        "split_sizes": compute_split_sizes(max_examples),
    }

    if not run_results_path.exists() or not eval_results_path.exists():
        missing = [
            str(path.relative_to(REPO_DIR))
            for path in [run_results_path, eval_results_path]
            if not path.exists()
        ]
        return {
            **base_summary,
            "status": "missing_artifacts",
            "error": "Missing required artifacts: " + ", ".join(missing),
            "full": make_missing_summary(expected_trials),
            "train": make_missing_summary(expected_trials),
            "val": make_missing_summary(expected_trials),
            "test": make_missing_summary(expected_trials),
            "workspace_records": [],
        }

    workspace_records = build_workspace_records(run_results_path, eval_results_path, max_examples)
    invalid_score_records = [record for record in workspace_records if record["score"] is None]
    if invalid_score_records:
        return {
            **base_summary,
            "status": "invalid_eval_results",
            "error": f"Found {len(invalid_score_records)} workspace rows with null score",
            "found_trials": len({record["rollout_id"] for record in workspace_records}),
            "full": make_missing_summary(expected_trials),
            "train": make_missing_summary(expected_trials),
            "val": make_missing_summary(expected_trials),
            "test": make_missing_summary(expected_trials),
            "workspace_records": workspace_records,
        }

    split_records = {
        "full": workspace_records,
        "train": [record for record in workspace_records if record["split"] == "train"],
        "val": [record for record in workspace_records if record["split"] == "val"],
        "test": [record for record in workspace_records if record["split"] == "test"],
    }
    split_summaries = {
        split_name: summarize_records(split_records[split_name], expected_trials)
        for split_name in SUMMARY_SPLITS
    }

    return {
        **base_summary,
        "status": "ok",
        "error": None,
        "found_trials": len({record["rollout_id"] for record in workspace_records}),
        "full": split_summaries["full"],
        "train": split_summaries["train"],
        "val": split_summaries["val"],
        "test": split_summaries["test"],
        "workspace_records": workspace_records,
    }


def flatten_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "batch_name": summary["batch_name"],
        "batch_dir": summary["batch_dir"],
        "batch_json_path": summary["batch_json_path"],
        "launch_results_path": summary["launch_results_path"] or "",
        "source_batch_dir": summary["source_batch_dir"] or "",
        "batch_index": summary["batch_index"],
        "task_id": summary["task_id"],
        "model_name": summary["model_name"],
        "prompt_name": summary["prompt_name"],
        "max_examples": summary["max_examples"],
        "expected_trials": summary["expected_trials"],
        "found_trials": summary["found_trials"],
        "rollout_version": summary["rollout_version"],
        "effective_rollout_version": summary["effective_rollout_version"],
        "agent_file": summary["agent_file"] or "",
        "status": summary["status"],
        "error": summary["error"] or "",
        "launch_status": "" if summary["launch"] is None else summary["launch"].get("status", ""),
        "launch_return_code": "" if summary["launch"] is None else summary["launch"].get("return_code", ""),
        "run_dir": summary["linkage"]["run_dir"],
        "run_results_path": summary["linkage"]["run_results_path"],
        "eval_results_path": summary["linkage"]["eval_results_path"],
        "run_config_path": summary["linkage"]["run_config_path"],
        "collect_log_path": summary["linkage"]["collect_log_path"],
        "evaluate_log_path": summary["linkage"]["evaluate_log_path"],
        "collect_log_exists": summary["linkage"]["collect_log_path_exists"],
        "evaluate_log_exists": summary["linkage"]["evaluate_log_path_exists"],
        "eval_score": summary["full"]["score"],
    }

    for split_name in SUMMARY_SPLITS:
        split_summary = summary[split_name]
        row[f"{split_name}_examples"] = split_summary["examples"]
        row[f"{split_name}_records"] = split_summary["records"]
        for value_field in SUMMARY_VALUE_FIELDS:
            row[f"{split_name}_{value_field}"] = split_summary[value_field]
        for trial_index in range(summary["expected_trials"]):
            row[f"{split_name}_trial_{trial_index}_score"] = split_summary["trial_scores"][trial_index]
            row[f"{split_name}_trial_{trial_index}_cost"] = split_summary["trial_costs"][trial_index]
            for token_field in TOKEN_FIELDS:
                row[f"{split_name}_trial_{trial_index}_{token_field}"] = split_summary[
                    f"trial_{token_field}"
                ][trial_index]
            for latency_field in LATENCY_FIELDS:
                row[f"{split_name}_trial_{trial_index}_{latency_field}"] = split_summary[
                    f"trial_{latency_field}"
                ][trial_index]
            for latency_avg_field in LATENCY_AVG_FIELDS:
                row[f"{split_name}_trial_{trial_index}_{latency_avg_field}"] = split_summary[
                    f"trial_{latency_avg_field}"
                ][trial_index]
            for count_field in COUNT_FIELDS:
                row[f"{split_name}_trial_{trial_index}_{count_field}"] = split_summary[
                    f"trial_{count_field}"
                ][trial_index]

    return row


def build_record_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in summary["workspace_records"]:
        rows.append(
            {
                "batch_name": summary["batch_name"],
                "batch_dir": summary["batch_dir"],
                "batch_index": summary["batch_index"],
                "task_id": summary["task_id"],
                "model_name": summary["model_name"],
                "prompt_name": summary["prompt_name"],
                "status": summary["status"],
                "rollout_version": summary["rollout_version"],
                "effective_rollout_version": summary["effective_rollout_version"],
                "run_dir": summary["linkage"]["run_dir"],
                "run_results_path": summary["linkage"]["run_results_path"],
                "eval_results_path": summary["linkage"]["eval_results_path"],
                "run_config_path": summary["linkage"]["run_config_path"],
                "collect_log_path": summary["linkage"]["collect_log_path"],
                "evaluate_log_path": summary["linkage"]["evaluate_log_path"],
                **record,
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summarize_run_job(job: tuple[Path, dict[str, Any], dict[str, Any] | None]) -> dict[str, Any]:
    batch_dir, run_meta, launch_meta = job
    return summarize_run(batch_dir, run_meta, launch_meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-batches",
        type=Path,
        default=PACKAGE_DIR / "data" / "source_batch_dirs.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs" / "data")
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dirs = load_batch_dirs(args.source_batches)

    run_jobs = []
    for batch_dir in batch_dirs:
        batch_meta = load_json(batch_dir / "batch.json")
        launch_by_index = launch_results_by_index(batch_dir)
        for run_meta in batch_meta["runs"]:
            run_jobs.append((batch_dir, run_meta, launch_by_index.get(int(run_meta["index"]))))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        summaries = list(executor.map(summarize_run_job, run_jobs))

    summary_rows = [flatten_summary_row(summary) for summary in summaries]
    record_rows = []
    for summary in summaries:
        record_rows.extend(build_record_rows(summary))

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch_dirs": [str(batch_dir.relative_to(REPO_DIR)) for batch_dir in batch_dirs],
        "split_weights": SPLIT_WEIGHTS,
        "rows": summaries,
    }
    summary_json_path = output_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    write_tsv(output_dir / "summary.tsv", summary_rows)
    write_tsv(output_dir / "records.tsv", record_rows)

    print(summary_json_path)
    print(output_dir / "summary.tsv")
    print(output_dir / "records.tsv")


if __name__ == "__main__":
    main()
