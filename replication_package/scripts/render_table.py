import argparse
import json
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
    "webarena": "Web.",
    "refactorbench": "Refact.",
}
MODEL_ORDER = [
    "gemini-3.1-pro-preview",
    "gemma-4-26b-a4b",
    "qwen3-coder-30b-a3b",
    "ministral-3-8b",
]
MODEL_GROUPS = {
    "Large Language Models": ["gemini-3.1-pro-preview"],
    "Small Language Models": [
        "gemma-4-26b-a4b",
        "qwen3-coder-30b-a3b",
        "ministral-3-8b",
    ],
}
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
    df = df.copy()
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


def prepare_table_rows(df_test: pd.DataFrame) -> pd.DataFrame:
    return (
        df_test[df_test["task_id"].isin(TASK_ORDER)]
        .assign(
            task_sort=lambda frame: pd.Categorical(
                frame["task_id"], categories=TASK_ORDER, ordered=True
            ),
            model_sort=lambda frame: pd.Categorical(
                frame["model_name"], categories=MODEL_ORDER, ordered=True
            ),
            task_id=lambda frame: frame["task_id"].map(TASK_LABELS),
            rollout_version=lambda frame: frame["rollout_version"]
            .str.replace(r"^gepa_seed\d+_best$", "optimized", regex=True)
            .str.replace(r"^baseline.*", "baseline", regex=True),
            score_pct=lambda frame: frame["score"] * 100,
            latency_sec=lambda frame: frame[LATENCY_COLUMN],
        )
        .sort_values(["model_sort", "rollout_version", "task_sort"])
        .drop(columns=["task_sort", "model_sort"])
    )


def build_wide_table(df_sorted: pd.DataFrame) -> pd.DataFrame:
    task_labels = [TASK_LABELS[task_id] for task_id in TASK_ORDER]
    wide = (
        df_sorted.assign(
            rollout_label=lambda frame: frame["rollout_version"]
            .map({"baseline": "", "optimized": "+ optimized harness"})
            .fillna(frame["rollout_version"])
        )
        .pivot_table(
            index=["model_name", "rollout_label"],
            columns="task_id",
            values=["score_pct", "estimated_cost_per_query", "latency_sec"],
            aggfunc="first",
        )
        .reindex(columns=task_labels, level=1)
    )

    for metric in ["score_pct", "estimated_cost_per_query", "latency_sec"]:
        wide[(metric, "Avg.")] = wide[metric].mean(axis=1)

    wide = wide.reset_index()
    wide[("model_sort", "")] = pd.Categorical(
        wide[("model_name", "")], categories=MODEL_ORDER, ordered=True
    )
    wide[("rollout_sort", "")] = wide[("rollout_label", "")].map(
        {"": 0, "+ optimized harness": 1}
    )
    return wide.sort_values([("model_sort", ""), ("rollout_sort", "")]).drop(
        columns=[("model_sort", ""), ("rollout_sort", "")]
    )


def format_latency(seconds: float) -> str:
    return f"{seconds:.0f}s"


def format_latex_cell(score: float, cost: float, latency: float) -> str:
    return f"\\cell{{{score:.1f}}}{{\\${cost:.3f}}}{{{format_latency(latency)}}}"


def format_markdown_cell(score: float, cost: float, latency: float) -> str:
    return f"{score:.1f} / ${cost:.3f} / {format_latency(latency)}"


def format_model_label(model_name: str, rollout_label: str) -> str:
    if rollout_label == "":
        return f"\\texttt{{{model_name}}}"
    return "{\\textit{{+ optimized harness}}}"


def format_latex_row(row: pd.Series, columns: list[str]) -> str:
    label = format_model_label(str(row[("model_name", "")]), str(row[("rollout_label", "")]))
    cells = [
        format_latex_cell(
            float(row[("score_pct", column)]),
            float(row[("estimated_cost_per_query", column)]),
            float(row[("latency_sec", column)]),
        )
        for column in columns
    ]
    return f"  {label}\n    & " + "\n    & ".join(cells) + " \\\\"


def render_latex_table(wide: pd.DataFrame) -> str:
    columns = [TASK_LABELS[task_id] for task_id in TASK_ORDER] + ["Avg."]
    lines = [
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.25cm}rrrrrrrr@{}}",
        r"  \toprule",
        "  "
        + " & ".join(
            [
                r"\textbf{Model}",
                *[f"\\textbf{{{column}}}" for column in columns],
            ]
        )
        + r" \\",
        r"  \midrule",
    ]

    group_index = 0
    for group_name, model_names in MODEL_GROUPS.items():
        if group_index > 0:
            lines.append(r"  \midrule")
            lines.append("")
        lines.append(
            "  "
            + rf"\multicolumn{{9}}{{@{{}}l@{{}}}}{{\footnotesize\textit{{\textcolor{{adaptgray}}{{{group_name}}}}}}} \\"
        )
        for model_index, model_name in enumerate(model_names):
            model_rows = wide[wide[("model_name", "")].astype(str) == model_name]
            for _, row in model_rows.iterrows():
                lines.append("")
                lines.append(format_latex_row(row, columns))
            if model_index < len(model_names) - 1:
                lines.append(r"  \midrule")
        group_index += 1

    lines.extend(["", r"  \bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def render_markdown_table(wide: pd.DataFrame) -> str:
    columns = [TASK_LABELS[task_id] for task_id in TASK_ORDER] + ["Avg."]
    rows = []
    for _, row in wide.iterrows():
        model_name = str(row[("model_name", "")])
        rollout_label = str(row[("rollout_label", "")])
        rows.append(
            {
                "Model": model_name if rollout_label == "" else f"{model_name} + optimized harness",
                **{
                    column: format_markdown_cell(
                        float(row[("score_pct", column)]),
                        float(row[("estimated_cost_per_query", column)]),
                        float(row[("latency_sec", column)]),
                    )
                    for column in columns
                },
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def write_outputs(wide: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latex = render_latex_table(wide)
    markdown = render_markdown_table(wide)
    (output_dir / "results_table.tex").write_text(latex + "\n", encoding="utf-8")
    (output_dir / "results_table.md").write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, default=PACKAGE_DIR / "outputs" / "data" / "summary.json")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs" / "tables")
    args = parser.parse_args()

    df = load_summary(args.summary_json)
    df_test = add_costs(select_test_rows(df))
    df_sorted = prepare_table_rows(df_test)
    wide = build_wide_table(df_sorted)
    write_outputs(wide, args.output_dir)
    print(args.output_dir / "results_table.tex")
    print(args.output_dir / "results_table.md")


if __name__ == "__main__":
    main()
