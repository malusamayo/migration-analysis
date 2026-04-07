# BrowseComp-Plus Evaluation

Agents must write their final answer to `answer.txt` in the workspace (concise phrase only, no explanation).

Scoring is done by an LLM judge that compares the agent's answer against the reference answer. Minor spelling variants are accepted for named entities.

## Prerequisites

Download the BM25 index before running:

```bash
cd /mnt/datasets/BrowseComp-Plus
bash scripts_build_index/download_indexes.sh
```

## Run

```bash
uv run python -m src.runner --config tasks/browsecompplus/run.yaml
```
