import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PACKAGE_DIR.parent
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
EXAMPLE_RE = re.compile(r"example(\d+)_rollout(\d+)_logs")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    path: Path
    requested_path: Path


TASKS = [
    TaskSpec(
        "attendance_payroll_audit_s2l_high",
        Path("results/attendance_payroll_audit_s2l_high/gemini-3.1-pro-preview_default/rollouts/baseline"),
        Path("results/attendance_payroll_audit_s2l_high/gemini-3.1-pro-preview_default/rollouts/baseline"),
    ),
    TaskSpec(
        "budget_approval_s2l_high",
        Path("results/budget_approval_s2l_high/gemini-3.1-pro-preview_default/rollouts/baseline"),
        Path("results/budget_approval_s2l_high/gemini-3.1-pro-preview_default/rollouts/baseline"),
    ),
    TaskSpec(
        "machine_operating_s2l",
        Path("results/machine_operating_s2l/gemini-3.1-pro-preview_default/rollouts/baseline"),
        Path("results/machine_operating_s2l/gemini-3.1-pro-preview_default/rollouts/baseline"),
    ),
    TaskSpec(
        "refactorbench",
        Path("results/refactorbench/gemini-3.1-pro-preview_default/rollouts/baseline"),
        Path("results/refactorbench/gemini-3.1-pro-preview_default/rollouts/baseline"),
    ),
    TaskSpec(
        "webtest",
        Path("results/webtest/gemini-3.1-pro-preview_static/rollouts/baseline"),
        Path("results/webtest/gemini-3.1-pro-preview_static/rollouts/baseline"),
    ),
    TaskSpec(
        "webarena",
        Path("results/webarena/gemini-3.1-pro-preview_shopping_admin/rollouts/collect_all"),
        Path("results/webarena/gemini-3.1-pro-preview_static/rollouts/baseline"),
    ),
    TaskSpec(
        "woocommerce_stock_alert_s2l",
        Path("results/woocommerce_stock_alert_s2l/gemini-3.1-pro-preview_default/rollouts/baseline"),
        Path("results/woocommerce_stock_alert_s2l/gemini-3.1-pro-preview_default/rollouts/baseline"),
    ),
]


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_DIR / path


def trace_files(path: Path) -> list[Path]:
    resolved_path = resolve_repo_path(path)
    files = sorted(resolved_path.glob("example*_rollout*_logs/trace_*.json"))
    if not files:
        files = sorted(resolved_path.glob("example*_rollout*/trace_*.json"))
    if not files:
        files = sorted(resolved_path.glob("**/trace_*.json"))
    return files


def content_text(content: list[dict]) -> str:
    return "\n".join(item["text"] for item in content if item["type"] == "text")


def extract_trace(path: Path, task: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data["events"]
    user_events = [event for event in events if event.get("source") == "user" and event.get("llm_message")]
    prompt = content_text(user_events[0]["llm_message"]["content"])
    system_events = [event for event in events if event.get("system_prompt")]
    system_prompt = system_events[0]["system_prompt"]["text"]
    action_tools = [
        event["tool_name"]
        for event in events
        if event.get("kind") == "ActionEvent" and event.get("tool_name")
    ]
    reasoning_chars = sum(len(event.get("reasoning_content") or "") for event in events)
    match = EXAMPLE_RE.search(str(path))
    example_id = int(match.group(1))
    rollout_id = int(match.group(2))
    return {
        "task": task,
        "example_id": example_id,
        "rollout_id": rollout_id,
        "trace_path": str(path.relative_to(REPO_DIR)),
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "system_prompt": system_prompt,
        "tool_sequence": action_tools,
        "tool_sequence_key": " -> ".join(action_tools),
        "num_actions": len(action_tools),
        "reasoning_chars": reasoning_chars,
        "error": data["error"],
    }


def tokenize(texts: list[str]) -> list[str]:
    tokens = []
    for text in texts:
        tokens.extend(token.lower() for token in TOKEN_RE.findall(text))
    return tokens


def normalized_entropy(items: list[str]) -> float:
    counts = Counter(items)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(len(counts))


def distinct_ngram(tokens: list[str], n: int) -> float:
    grams = list(zip(*(tokens[i:] for i in range(n))))
    return len(set(grams)) / len(grams)


def mean_pairwise_distance(similarity: np.ndarray) -> float:
    n = similarity.shape[0]
    distances = 1 - similarity[np.triu_indices(n, k=1)]
    return float(distances.mean())


def mean_nearest_neighbor_distance(similarity: np.ndarray) -> float:
    masked = similarity.copy()
    np.fill_diagonal(masked, -np.inf)
    return float((1 - masked.max(axis=1)).mean())


def text_similarity_metrics(texts: list[str], analyzer: str, ngram_range: tuple[int, int], prefix: str) -> dict:
    matrix = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range).fit_transform(texts)
    similarity = cosine_similarity(matrix)
    return {
        f"{prefix}_mean_pairwise_distance": mean_pairwise_distance(similarity),
        f"{prefix}_mean_nearest_neighbor_distance": mean_nearest_neighbor_distance(similarity),
    }


def sequence_distance(seq_a: list[str], seq_b: list[str]) -> float:
    return 1 - SequenceMatcher(a=seq_a, b=seq_b, autojunk=False).ratio()


def normalized_edit_distance(seq_a: list[str], seq_b: list[str]) -> float:
    previous = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        current = [i]
        for j, item_b in enumerate(seq_b, start=1):
            substitution_cost = int(item_a != item_b)
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1] / max(len(seq_a), len(seq_b))


def mean_sequence_distance(sequences: list[list[str]]) -> float:
    distances = []
    for index, seq_a in enumerate(sequences):
        for seq_b in sequences[index + 1:]:
            distances.append(sequence_distance(seq_a, seq_b))
    return float(np.mean(distances))


def mean_normalized_edit_distance(sequences: list[list[str]]) -> float:
    distances = []
    for index, seq_a in enumerate(sequences):
        for seq_b in sequences[index + 1:]:
            distances.append(normalized_edit_distance(seq_a, seq_b))
    return float(np.mean(distances))


def mean_tool_set_distance(sequences: list[list[str]]) -> float:
    distances = []
    sets = [set(sequence) for sequence in sequences]
    for index, set_a in enumerate(sets):
        for set_b in sets[index + 1:]:
            distances.append(1 - (len(set_a & set_b) / len(set_a | set_b)))
    return float(np.mean(distances))


def load_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    trace_rows = []
    task_rows = []
    for spec in TASKS:
        files = trace_files(spec.path)
        print(f"Loading {len(files)} traces for {spec.name}", flush=True)
        task_rows.append(
            {
                "task": spec.name,
                "requested_path": str(spec.requested_path),
                "analyzed_path": str(spec.path),
                "trace_files": len(files),
                "path_substituted": spec.path != spec.requested_path,
            }
        )
        for index, path in enumerate(files, start=1):
            if index == 1 or index % 50 == 0 or index == len(files):
                print(f"  {spec.name}: parsing trace {index}/{len(files)}", flush=True)
            trace_rows.append(extract_trace(path, spec.name))
    return pd.DataFrame(trace_rows), pd.DataFrame(task_rows)


def unique_prompt_records(traces: pd.DataFrame) -> pd.DataFrame:
    columns = ["task", "example_id", "prompt_hash", "prompt"]
    return traces[columns].drop_duplicates(["task", "example_id", "prompt_hash"]).reset_index(drop=True)


def compute_metrics(traces: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    metric_rows = []
    for task, prompt_group in prompts.groupby("task", sort=False):
        print(f"Computing diversity metrics for {task}", flush=True)
        texts = prompt_group["prompt"].tolist()
        tokens = tokenize(texts)
        words_per_prompt = [len(tokenize([text])) for text in texts]
        trace_group = traces[traces["task"] == task]
        sequences = trace_group["tool_sequence"].tolist()
        tool_calls = [tool for sequence in sequences for tool in sequence]
        tool_sequence_mean_edit_distance = mean_normalized_edit_distance(sequences)
        metrics = {
            "task": task,
            "num_traces": len(trace_group),
            "num_unique_examples": prompt_group["example_id"].nunique(),
            "num_unique_prompts": prompt_group["prompt_hash"].nunique(),
            "avg_prompt_words": float(np.mean(words_per_prompt)),
            "prompt_word_count_cv": float(np.std(words_per_prompt) / np.mean(words_per_prompt)),
            "distinct_1": len(set(tokens)) / len(tokens),
            "distinct_2": distinct_ngram(tokens, 2),
            "token_entropy_norm": normalized_entropy(tokens),
            "avg_tool_calls": float(trace_group["num_actions"].mean()),
            "tool_call_count_cv": float(trace_group["num_actions"].std(ddof=0) / trace_group["num_actions"].mean()),
            "unique_tool_sequence_rate": trace_group["tool_sequence_key"].nunique() / len(trace_group),
            "tool_name_entropy_norm": normalized_entropy(tool_calls),
            "tool_sequence_mean_distance": tool_sequence_mean_edit_distance,
            "tool_sequence_mean_edit_distance": tool_sequence_mean_edit_distance,
            "tool_set_mean_jaccard_distance": mean_tool_set_distance(sequences),
        }
        metrics.update(text_similarity_metrics(texts, "word", (1, 2), "word_tfidf"))
        metrics.update(text_similarity_metrics(texts, "char_wb", (3, 5), "char_tfidf"))
        metric_rows.append(metrics)
    return pd.DataFrame(metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_DIR / "outputs" / "data")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces, task_paths = load_records()
    prompts = unique_prompt_records(traces)
    metrics = compute_metrics(traces, prompts)

    task_paths.to_csv(args.output_dir / "task_diversity_task_paths.csv", index=False)
    traces.drop(columns=["tool_sequence"]).to_csv(args.output_dir / "task_diversity_trace_records.csv", index=False)
    prompts.to_csv(args.output_dir / "task_diversity_prompt_records.csv", index=False)
    metrics.to_csv(args.output_dir / "task_diversity_metrics.csv", index=False)
    metrics.to_json(args.output_dir / "task_diversity_metrics.json", orient="records", indent=2)
    print(args.output_dir / "task_diversity_metrics.csv")


if __name__ == "__main__":
    main()
