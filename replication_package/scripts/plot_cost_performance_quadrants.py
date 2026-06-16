import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


PACKAGE_DIR = Path(__file__).resolve().parents[1]
POINT_SPECS = [
    ("gemma-4-26b-a4b", "+ optimized harness", "#6f9bb2"),
    ("gemma-4-26b-a4b", "", "#a6a6a6"),
    ("gemini-3.1-pro-preview", "", "#d97b7d"),
]


def read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def parse_score_and_cost(value: str) -> tuple[float, float]:
    match = re.match(r"\s*([0-9.]+)\s*/\s*\\?\$([0-9.]+)", value)
    return float(match.group(1)), float(match.group(2))


def build_plot_data(tableau_data_path: Path) -> pd.DataFrame:
    source = read_dataframe(tableau_data_path).fillna("")
    rows = []
    for model_name, rollout_label, color in POINT_SPECS:
        row = source[
            (source["model_name"] == model_name)
            & (source["rollout_label"] == rollout_label)
        ].iloc[0]
        score, cost = parse_score_and_cost(row["Budget"])
        rows.append(
            {
                "model_name": model_name,
                "rollout_label": rollout_label,
                "score": score,
                "cost": cost,
                "color": color,
            }
        )
    return pd.DataFrame(rows)


def currency(value: float, _position: int) -> str:
    return f"${value:0.3f}"


def plot(data: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=150)
    ax.scatter(
        data["cost"],
        data["score"],
        s=130,
        c=data["color"],
        edgecolors="none",
        alpha=0.95,
    )

    ax.set_xlim(-0.012, 0.24)
    ax.set_ylim(0, 105)
    ax.set_xticks([0.0, 0.12, 0.24])
    ax.set_yticks([0, 50, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(currency))

    ax.set_xlabel("Cost", fontsize=15, fontweight="bold")
    ax.set_ylabel("Performance", fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=13, length=5, color="#777777")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#777777")

    ax.grid(False)
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tableau-data", type=Path, default=PACKAGE_DIR / "outputs" / "data" / "tableau_data.json")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs" / "figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = build_plot_data(args.tableau_data)
    output_path = args.output_dir / "cost_performance_quadrants.png"
    data.to_csv(args.output_dir / "cost_performance_quadrants_data.csv", index=False)
    plot(data, output_path)
    print(output_path)
    print(output_path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
