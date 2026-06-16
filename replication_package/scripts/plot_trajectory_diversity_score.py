import argparse
import re
from pathlib import Path

import altair as alt
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
TASK_COLUMN_MAP = {
    "attendance_payroll_audit_s2l_high": "Attn.",
    "budget_approval_s2l_high": "Budget",
    "woocommerce_stock_alert_s2l": "Stock",
    "machine_operating_s2l": "Anom.",
    "webtest": "Playwr.",
    "webarena": "Web Mgmt.",
    "refactorbench": "Refact.",
}
TASK_LABELS = {
    "attendance_payroll_audit_s2l_high": "Attendance",
    "budget_approval_s2l_high": "Budget",
    "woocommerce_stock_alert_s2l": "Stock",
    "machine_operating_s2l": "Anomaly",
    "webtest": "Playwright",
    "webarena": "WebArena",
    "refactorbench": "Refactor",
}


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def parse_score(value: str) -> float:
    return float(re.match(r"\s*([0-9.]+)\s*/", value).group(1))


def build_plot_data(
    metrics_path: Path,
    tableau_data_path: Path,
    diversity_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diversity = read_dataframe(metrics_path)
    performance = read_dataframe(tableau_data_path)
    optimized = performance[performance["rollout_label"] == "+ optimized harness"]

    rows = []
    for task, score_column in TASK_COLUMN_MAP.items():
        trajectory_diversity = diversity.loc[diversity["task"] == task, diversity_column].item()
        for _, model_row in optimized.iterrows():
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "model_name": model_row["model_name"],
                    "trajectory_diversity": trajectory_diversity,
                    "optimized_score": parse_score(model_row[score_column]),
                }
            )

    model_points = pd.DataFrame(rows)
    task_means = (
        model_points.groupby(["task", "task_label", "trajectory_diversity"], as_index=False)
        .agg(
            mean_optimized_score=("optimized_score", "mean"),
            std_optimized_score=("optimized_score", "std"),
        )
        .sort_values("trajectory_diversity")
    )
    return model_points, task_means


def build_task_diversity_panel(task_means: pd.DataFrame, show_labels: bool) -> alt.Chart:
    x_min = task_means["trajectory_diversity"].min()
    x_max = task_means["trajectory_diversity"].max()
    y_min = task_means["mean_optimized_score"].min()
    y_max = task_means["mean_optimized_score"].max()
    x_padding = max((x_max - x_min) * 0.08, 0.015)
    y_padding = max((y_max - y_min) * 0.12, 4)

    base = alt.Chart(task_means).encode(
        x=alt.X(
            "trajectory_diversity:Q",
            title="Task diversity",
            scale=alt.Scale(domain=[x_min - x_padding, x_max + x_padding], nice=False),
            axis=alt.Axis(
                grid=True,
                labelFontSize=14,
                titleFontSize=17,
                titleFontWeight="bold",
                format=".2f",
                tickCount=5,
            ),
        ),
        y=alt.Y(
            "mean_optimized_score:Q",
            title="Avg. optimized score",
            scale=alt.Scale(domain=[y_min - y_padding, y_max + y_padding], nice=False),
            axis=alt.Axis(
                grid=True,
                labelFontSize=14,
                titleFontSize=17,
                titleFontWeight="bold",
                tickCount=5,
            ),
        ),
    )

    line = base.transform_regression(
        "trajectory_diversity",
        "mean_optimized_score",
    ).mark_line(color="#2b6f8f", strokeWidth=4)
    points = base.mark_circle(size=160, color="#2b6f8f", opacity=0.72).encode(
        tooltip=[
            alt.Tooltip("task_label:N", title="Task"),
            alt.Tooltip("trajectory_diversity:Q", title="Edit distance", format=".3f"),
            alt.Tooltip("mean_optimized_score:Q", title="Mean score", format=".1f"),
        ],
    )

    chart = line + points
    if show_labels:
        labels = base.mark_text(
            align="left",
            baseline="middle",
            dx=7,
            fontSize=9,
            color="#111111",
        ).encode(text="task_label:N")
        chart += labels
    return chart.properties(width=200, height=150)


def build_template_panel(template_data: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(template_data).encode(
        x=alt.X(
            "number_of_task_templates:Q",
            title="Number of task templates",
            scale=alt.Scale(domain=[0, 22], nice=False),
            axis=alt.Axis(
                values=[3, 8, 20],
                grid=True,
                labelFontSize=14,
                titleFontSize=17,
                titleFontWeight="bold",
                format="d",
            ),
        ),
        y=alt.Y(
            "avg_optimized_score:Q",
            title="",
            scale=alt.Scale(domain=[60, 95], nice=False),
            axis=alt.Axis(
                grid=True,
                labelFontSize=14,
                titleFontSize=17,
                titleFontWeight="bold",
                tickCount=4,
            ),
        ),
    )
    line = base.mark_line(color="#2b6f8f", strokeWidth=4)
    points = base.mark_circle(size=160, color="#2b6f8f", opacity=0.72)
    return (line + points).properties(width=200, height=150)


def plot(
    task_means: pd.DataFrame,
    template_data: pd.DataFrame,
    output_path: Path,
    show_labels: bool,
) -> None:
    charts = [
        build_task_diversity_panel(task_means, show_labels),
        build_template_panel(template_data),
    ]
    chart = (
        alt.hconcat(*charts, spacing=24)
        .resolve_scale(y="shared")
        .configure_axis(
            gridColor="#d8d8d8",
            gridWidth=1.2,
            domainColor="#8a8a8a",
            domainWidth=1.4,
            tickColor="#8a8a8a",
            labelColor="#111111",
            titleColor="#111111",
        )
        .configure_view(stroke=None)
        .configure(background="white")
    )
    chart.save(output_path)
    chart.save(output_path.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=PACKAGE_DIR / "outputs" / "data" / "task_diversity_metrics.json")
    parser.add_argument("--tableau-data", type=Path, default=PACKAGE_DIR / "outputs" / "data" / "tableau_data.json")
    parser.add_argument("--template-data", type=Path, default=PACKAGE_DIR / "data" / "task_template_scores.json")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs" / "figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_points, task_means = build_plot_data(
        args.metrics,
        args.tableau_data,
        "tool_sequence_mean_edit_distance",
    )
    template_data = read_dataframe(args.template_data)
    model_points.to_csv(args.output_dir / "trajectory_diversity_model_points.csv", index=False)
    task_means.to_csv(args.output_dir / "trajectory_diversity_task_means.csv", index=False)

    output_path = args.output_dir / "trajectory_diversity_side_by_side.png"
    plot(task_means, template_data, output_path, show_labels=False)
    labeled_output_path = args.output_dir / "trajectory_diversity_side_by_side_labeled.png"
    plot(task_means, template_data, labeled_output_path, show_labels=True)
    print(output_path)
    print(output_path.with_suffix(".pdf"))
    print(labeled_output_path)
    print(labeled_output_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
