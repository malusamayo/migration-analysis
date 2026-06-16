import argparse
from pathlib import Path

import altair as alt
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
MODEL_DOMAIN = [
    "ministral-3-8b",
    "qwen3-coder-30b-a3b",
    "gemma-4-26b-a4b",
    "gemini-3.1-pro-preview",
]


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def parse_score(value: str) -> float:
    return float(value.split("/", 1)[0].strip())


def build_plot_data(tableau_data_path: Path, intelligence_path: Path) -> pd.DataFrame:
    performance = read_dataframe(tableau_data_path).fillna("")
    intelligence = read_dataframe(intelligence_path)
    rows = []
    for _, model_row in intelligence.iterrows():
        model_name = model_row["model_name"]
        base_row = performance[
            (performance["model_name"] == model_name)
            & (performance["rollout_label"] == "")
        ]
        optimized_row = performance[
            (performance["model_name"] == model_name)
            & (performance["rollout_label"] == "+ optimized harness")
        ]
        row = {
            "model_name": model_name,
            "intelligence_index": model_row["intelligence_index"],
            "base_harness": parse_score(base_row.iloc[0]["Avg."]),
        }
        if not optimized_row.empty:
            row["optimized_harness"] = parse_score(optimized_row.iloc[0]["Avg."])
        else:
            row["optimized_harness"] = None
        rows.append(row)

    data = pd.DataFrame(rows)
    long_data = data.melt(
        id_vars=["model_name", "intelligence_index"],
        value_vars=["base_harness", "optimized_harness"],
        var_name="harness",
        value_name="avg_performance",
    ).dropna()
    long_data["harness"] = long_data["harness"].replace(
        {
            "base_harness": "Base harness",
            "optimized_harness": "Optimized harness",
        }
    )
    return long_data


def build_chart(data: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(data).encode(
        x=alt.X(
            "intelligence_index:Q",
            title="Artificial Analysis intelligence index",
            scale=alt.Scale(domain=[10, 60]),
        ),
        y=alt.Y(
            "avg_performance:Q",
            title="Avg. performance",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "harness:N",
            title="Harness",
            scale=alt.Scale(
                domain=["Base harness", "Optimized harness"],
                range=["#1f77b4", "#ff7f0e"],
            ),
            legend=alt.Legend(orient="right", symbolType="circle"),
        ),
        shape=alt.Shape(
            "model_name:N",
            title="Model",
            scale=alt.Scale(
                domain=MODEL_DOMAIN,
                range=["circle", "square", "triangle", "diamond"],
            ),
            legend=alt.Legend(orient="right"),
        ),
        tooltip=[
            alt.Tooltip("model_name:N", title="Model"),
            alt.Tooltip("harness:N", title="Harness"),
            alt.Tooltip("intelligence_index:Q", title="Intelligence"),
            alt.Tooltip("avg_performance:Q", title="Performance", format=".1f"),
        ],
    )

    trend_lines = base.mark_line(strokeWidth=1.5, opacity=0.6).encode(
        detail="harness:N",
        order=alt.Order("intelligence_index:Q"),
    )
    points = base.mark_point(filled=True, size=90, stroke="#ffffff", strokeWidth=0.8)

    return (
        alt.layer(trend_lines, points)
        .properties(width=200, height=100)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=10, titleFontSize=11, orient="right")
        .configure_view(stroke=None)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tableau-data",
        type=Path,
        default=PACKAGE_DIR / "outputs" / "data" / "tableau_data.json",
    )
    parser.add_argument(
        "--intelligence-index",
        type=Path,
        default=PACKAGE_DIR / "data" / "model_intelligence_index.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs" / "figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_plot_data(args.tableau_data, args.intelligence_index)
    data.to_csv(args.output_dir / "intelligence_vs_performance_data.csv", index=False)
    chart = build_chart(data)
    output_path = args.output_dir / "intelligence_vs_performance.png"
    chart.save(output_path, scale_factor=2)
    chart.save(output_path.with_suffix(".pdf"))
    print(output_path)
    print(output_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
