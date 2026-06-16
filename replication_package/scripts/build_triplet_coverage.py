import argparse
import csv
import json
import re
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
FINAL_TABLE_TASKS = [
    "attendance_payroll_audit_s2l_high",
    "budget_approval_s2l_high",
    "woocommerce_stock_alert_s2l",
    "machine_operating_s2l",
    "webtest",
    "webarena",
    "refactorbench",
]
FINAL_TABLE_OPTIMIZED_MODELS = [
    "gemma-4-26b-a4b",
    "ministral-3-8b",
    "qwen3-coder-30b-a3b",
]
FINAL_TABLE_BASELINE_MODELS = [
    "gemini-3.1-pro-preview",
    "gemma-4-26b-a4b",
    "ministral-3-8b",
    "qwen3-coder-30b-a3b",
]
EXTRA_VARIANT_TASKS = [
    "attendance_payroll_audit_s2l_medium",
    "attendance_payroll_audit_s2l_extra_high",
    "budget_approval_s2l_medium",
    "budget_approval_s2l_extra_high",
]
EXTRA_VARIANT_MODEL = "qwen3-coder-30b-a3b"
SEED_PATTERN = re.compile(r"^gepa_seed(\d+)_best$")


def load_summary(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def optimized_seed(row: dict) -> int | None:
    match = SEED_PATTERN.fullmatch(row["rollout_version"])
    if match:
        return int(match.group(1))
    return None


def optimized_coverage_group(row: dict) -> str:
    if row["task_id"] in FINAL_TABLE_TASKS and row["model_name"] in FINAL_TABLE_OPTIMIZED_MODELS:
        return "main_final_seed_pool"
    if row["task_id"] in EXTRA_VARIANT_TASKS and row["model_name"] == EXTRA_VARIANT_MODEL:
        return "extra_variant_seed_pool"
    return "unused_candidate"


def build_triplet_rows(summary_rows: list[dict]) -> list[dict]:
    triplet_rows = []
    for row in summary_rows:
        seed = optimized_seed(row)
        if seed is not None:
            group = optimized_coverage_group(row)
            triplet_rows.append(
                {
                    "task_id": row["task_id"],
                    "model_name": row["model_name"],
                    "seed": seed,
                    "coverage_group": group,
                    "included_in_curated_run_eval_archive": group
                    in {"main_final_seed_pool", "extra_variant_seed_pool"},
                    "rollout_version": row["rollout_version"],
                    "batch_dir": row["batch_dir"],
                    "batch_index": row["batch_index"],
                    "run_dir": row["linkage"]["run_dir"],
                    "run_results_path": row["linkage"]["run_results_path"],
                    "eval_results_path": row["linkage"]["eval_results_path"],
                    "full_records": row["full"]["records"],
                    "test_records": row["test"]["records"],
                    "val_score": row["val"]["score"],
                    "test_score": row["test"]["score"],
                }
            )
    return sorted(
        triplet_rows,
        key=lambda item: (
            item["coverage_group"],
            item["task_id"],
            item["model_name"],
            item["seed"],
        ),
    )


def is_curated_baseline(row: dict) -> bool:
    return (
        row["rollout_version"].startswith("baseline")
        and row["task_id"] in FINAL_TABLE_TASKS
        and row["model_name"] in FINAL_TABLE_BASELINE_MODELS
    )


def is_curated_optimized(row: dict) -> bool:
    seed = optimized_seed(row)
    return seed is not None and optimized_coverage_group(row) in {
        "main_final_seed_pool",
        "extra_variant_seed_pool",
    }


def build_curated_run_eval_rows(summary_rows: list[dict]) -> list[dict]:
    return sorted(
        [row for row in summary_rows if is_curated_baseline(row) or is_curated_optimized(row)],
        key=lambda row: (
            row["task_id"],
            row["model_name"],
            row["rollout_version"],
            row["batch_index"],
        ),
    )


def write_rows(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=PACKAGE_DIR / "outputs" / "data" / "summary.json")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PACKAGE_DIR / "outputs" / "data" / "included_task_model_seed_triplets.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PACKAGE_DIR / "outputs" / "data" / "included_task_model_seed_triplets.json",
    )
    args = parser.parse_args()

    rows = build_triplet_rows(load_summary(args.summary_json))
    write_rows(rows, args.output_csv, args.output_json)

    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[row["coverage_group"]] = group_counts.get(row["coverage_group"], 0) + 1
    print(args.output_csv)
    print(args.output_json)
    print(json.dumps(group_counts, sort_keys=True))


if __name__ == "__main__":
    main()
