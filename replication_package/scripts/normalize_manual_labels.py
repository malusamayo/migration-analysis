import argparse
from pathlib import Path

import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parents[1]
FAILURE_MODE_COLUMNS = [
    "tool-use",
    "instruction-following",
    "knowledge",
    "long-context",
    "planning",
]
STRATEGY_COLUMNS = [
    "context",
    "tool",
    "orchestration",
]


def parse_task_model_seed(value: str) -> tuple[str, str, str]:
    task, model, seed = [part.strip() for part in value.split(" / ")]
    return task, model, seed


def normalize_labels(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, sep="\t").fillna("")
    rows = []
    for _, raw_row in raw.iterrows():
        task, model, seed = parse_task_model_seed(raw_row.iloc[0])
        values = [str(value).strip() for value in raw_row.iloc[1:].tolist()]
        labels = {value for value in values if value}
        row = {
            "task": task,
            "model": model,
            "seed": seed,
        }
        row.update({f"failure_mode_{column}": column in labels for column in FAILURE_MODE_COLUMNS})
        row.update({f"strategy_{column}": column in labels for column in STRATEGY_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-labels", type=Path, default=PACKAGE_DIR / "data" / "manual_labels_raw.txt")
    parser.add_argument("--output", type=Path, default=PACKAGE_DIR / "outputs" / "data" / "manual_labels.csv")
    args = parser.parse_args()

    labels = normalize_labels(args.raw_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)
    labels.to_json(args.output.with_suffix(".json"), orient="records", indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
