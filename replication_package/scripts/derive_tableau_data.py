import argparse
import json
import re
from pathlib import Path

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
LATENCY_COLUMN = "avg_end_to_end_latency_sec"
TASK_ORDER = [
    "attendance_payroll_audit_s2l_high",
    "budget_approval_s2l_high",
    "woocommerce_stock_alert_s2l",
    "machine_operating_s2l",
    "webtest",
    "webarena",
    "refactorbench",
]
TASK_LABELS = {
    "attendance_payroll_audit_s2l_high": "Attn.",
    "budget_approval_s2l_high": "Budget",
    "woocommerce_stock_alert_s2l": "Stock",
    "machine_operating_s2l": "Anom.",
    "webtest": "Playwr.",
    "webarena": "Web Mgmt.",
    "refactorbench": "Refact.",
}
MODEL_ORDER = [
    "gemini-3.1-pro-preview",
    "gemma-4-26b-a4b",
    "ministral-3-8b",
    "qwen3-coder-30b-a3b",
]
MODEL_PRICES = {
    "qwen3-coder-30b-a3b": {
        "input_price_per_million": 0.15,
        "output_price_per_million": 0.6,
    },
    "ministral-3-8b": {
        "input_price_per_million": 0.15,
        "output_price_per_million": 0.15,
    },
    "gemini-3.1-pro-preview": {
        "input_price_per_million": 2,
        "output_price_per_million": 12,
    },
    "gemma-4-26b-a4b": {
        "input_price_per_million": 0.15,
        "output_price_per_million": 0.6,
    },
}


def load_summary(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as handle:
        summary = json.load(handle)
    return pd.json_normalize(summary["rows"])


def select_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["status"] == "ok"].copy()
    df["is_optimized"] = df["rollout_version"].str.endswith("best")
    best_indices = df.groupby(["task_id", "model_name", "is_optimized"])["val.score"].idxmax()
    test_cols = [col for col in df.columns if col.startswith("test.")]
    test_col_mapping = {col: col.replace("test.", "") for col in test_cols}
    df_test = df.loc[
        best_indices,
        [
            "task_id",
            "model_name",
            "rollout_version",
            "is_optimized",
            "val.score",
            *test_cols,
        ],
    ].reset_index(drop=True)
    return df_test.rename(columns=test_col_mapping)


def add_costs(df_test: pd.DataFrame) -> pd.DataFrame:
    df_test = df_test.copy()
    df_test["input_price_per_million"] = df_test["model_name"].map(
        lambda model_name: MODEL_PRICES[model_name]["input_price_per_million"]
    )
    df_test["output_price_per_million"] = df_test["model_name"].map(
        lambda model_name: MODEL_PRICES[model_name]["output_price_per_million"]
    )
    df_test["estimated_cost"] = (
        df_test["prompt_tokens"] * df_test["input_price_per_million"] / 1e6
        + df_test["completion_tokens"] * df_test["output_price_per_million"] / 1e6
    )
    df_test["estimated_cost_per_query"] = df_test["estimated_cost"] / df_test["records"]
    return df_test


def normalize_rollout_label(rollout_version: str) -> str:
    if re.match(r"^gepa_seed\d+_best$", rollout_version):
        return "+ optimized harness"
    if rollout_version.startswith("baseline"):
        return ""
    return rollout_version


def build_selected_rows(summary_path: Path) -> pd.DataFrame:
    df = load_summary(summary_path)
    df_test = add_costs(select_test_rows(df))
    return (
        df_test[df_test["task_id"].isin(TASK_ORDER)]
        .assign(
            task_sort=lambda frame: pd.Categorical(
                frame["task_id"], categories=TASK_ORDER, ordered=True
            ),
            model_sort=lambda frame: pd.Categorical(
                frame["model_name"], categories=MODEL_ORDER, ordered=True
            ),
            task_label=lambda frame: frame["task_id"].map(TASK_LABELS),
            rollout_label=lambda frame: frame["rollout_version"].map(normalize_rollout_label),
            score_pct=lambda frame: frame["score"] * 100,
            latency_sec=lambda frame: frame[LATENCY_COLUMN],
        )
        .sort_values(["model_sort", "rollout_label", "task_sort"])
        .reset_index(drop=True)
    )


def format_score_cost(score: float, cost: float) -> str:
    return f"{score:.1f} / \\${cost:.3f}"


def build_tableau_data(selected_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, rollout_label), group in selected_rows.groupby(
        ["model_name", "rollout_label"], sort=False
    ):
        by_task = group.set_index("task_label")
        row = {
            "model_name": model_name,
            "rollout_label": rollout_label,
        }
        rounded_scores = []
        costs = []
        for task_id in TASK_ORDER:
            task_label = TASK_LABELS[task_id]
            score = float(by_task.loc[task_label, "score_pct"])
            cost = float(by_task.loc[task_label, "estimated_cost_per_query"])
            rounded_scores.append(round(score, 1))
            costs.append(cost)
            row[task_label] = format_score_cost(score, cost)
        row["Avg."] = format_score_cost(sum(rounded_scores) / len(rounded_scores), sum(costs) / len(costs))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=PACKAGE_DIR / "outputs" / "data" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs" / "data")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = build_selected_rows(args.summary_json)
    tableau_data = build_tableau_data(selected_rows)
    selected_rows.to_csv(args.output_dir / "selected_test_rows.csv", index=False)
    tableau_data.to_csv(args.output_dir / "tableau_data.csv", index=False)
    selected_rows.to_json(args.output_dir / "selected_test_rows.json", orient="records", indent=2)
    tableau_data.to_json(args.output_dir / "tableau_data.json", orient="records", indent=2)
    print(args.output_dir / "selected_test_rows.csv")
    print(args.output_dir / "tableau_data.csv")


if __name__ == "__main__":
    main()
