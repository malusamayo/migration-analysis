# 17. Retriever Tool

`BM25ToolSet` adds in-memory BM25 document retrieval to an agent. It registers as a single named toolset (`"bm25_retriever"`) but expands into two coordinated tools at runtime:

| Tool name | What the agent calls it for |
|---|---|
| `build_index` | Load a JSONL corpus file and build the BM25 index |
| `retrieve` | Query the index with a natural-language string |

The two tools share internal state, so documents indexed by `build_index` are immediately available to `retrieve`. No external libraries are required.

## Corpus format

The corpus must be a **JSONL file** — one JSON object per line — where each object has:

| Field | Type | Required | Indexed? |
|---|---|---|---|
| `"id"` | `str` | yes | no (returned in results) |
| `"text"` | `str` | yes | yes |
| any extra fields | any | no | no (returned in results) |

Example `corpus.jsonl`:
```jsonl
{"id": "doc-1", "text": "The Eiffel Tower is in Paris, France.", "url": "https://..."}
{"id": "doc-2", "text": "Gustave Eiffel designed the tower for the 1889 World's Fair."}
```

The `"text"` field is tokenised and indexed. All other fields are stored verbatim and returned alongside each result, but they do not affect ranking.

## Imports

```python
from openhands.sdk import Agent, Tool
from openhands.tools.retriever.bm25 import BM25ToolSet
```

## Adding the toolset to an agent

```python
agent = Agent(
    llm=llm,
    tools=[Tool(name=BM25ToolSet.name)],
)
```

`BM25ToolSet.name` is `"bm25_retriever"`. No parameters are required at definition time — the corpus path is supplied by the agent at runtime when it calls `build_index`.

## Agent workflow

The system prompt should instruct the agent to:

1. Call `build_index(corpus_path="<path>")` once at the start of the task.
2. Call `retrieve(query="<natural language>", top_k=5)` as many times as needed.
3. Write the final answer (e.g. to `answer.txt`).

Example system prompt excerpt:
```
## Retrieval setup
Call build_index(corpus_path="corpus.jsonl") once before issuing any queries.

## Searching
Use retrieve(query="...", top_k=5) with specific, unusual terms.
For multi-hop questions, chain queries: find a candidate entity first,
then search for it by name to confirm additional facts.
```

## Converting other formats to JSONL

If your corpus is in a different format, provide a conversion script via `get_workspace_scripts()` and instruct the agent to run it first.

**Example: converting `context.txt` (`--- Document: <url> ---` format)**

```python
def get_workspace_scripts() -> dict[str, str]:
    return {
        "to_jsonl.py": """\
import json, re

with open("context.txt", encoding="utf-8") as f:
    raw = f.read()

docs = []
for chunk in re.split(r"--- Document: ", raw):
    chunk = chunk.strip()
    if not chunk:
        continue
    nl = chunk.find("\\n")
    url = chunk[:nl].removesuffix(" ---").strip()
    text = chunk[nl + 1:].replace("--- End of Document ---", "").strip()
    if url and text:
        docs.append({"id": url, "text": text})

with open("corpus.jsonl", "w") as f:
    for doc in docs:
        f.write(json.dumps(doc) + "\\n")
print(f"Wrote {len(docs)} documents.")
""",
    }
```

The agent then runs `python to_jsonl.py` via the terminal tool before calling `build_index`.

## Extending with a custom backend

Subclass `BaseRetrieverToolSet` to plug in a different retrieval algorithm (e.g. dense vector search) while keeping the same `build_index` / `retrieve` interface.

```python
from openhands.tools.retriever.base import (
    BaseRetrieverToolSet,
    BuildIndexAction, BuildIndexObservation,
    RetrieverAction, RetrieverObservation,
)
from openhands.sdk.tool import ToolDefinition, ToolExecutor, register_tool

class MyBuildExecutor(ToolExecutor[BuildIndexAction, BuildIndexObservation]):
    def __init__(self, state): self._state = state

    def __call__(self, action, conversation=None):
        # load action.corpus_path, build your index, store in self._state
        ...

class MyRetrieveExecutor(ToolExecutor[RetrieverAction, RetrieverObservation]):
    def __init__(self, state): self._state = state

    def __call__(self, action, conversation=None):
        # query self._state.index with action.query / action.top_k
        ...

class MyRetrieverToolSet(BaseRetrieverToolSet):
    name = "my_retriever"

    @classmethod
    def create(cls, conv_state=None, **params):
        state = MyState()
        build_tool = ...   # ToolDefinition wrapping MyBuildExecutor
        retrieve_tool = ... # ToolDefinition wrapping MyRetrieveExecutor
        return [build_tool, retrieve_tool]

register_tool(MyRetrieverToolSet.name, MyRetrieverToolSet)
```

## Key classes

| Symbol | Location | Role |
|---|---|---|
| `BM25ToolSet` | `openhands.tools.retriever.bm25` | Registered toolset factory (`"bm25_retriever"`) |
| `BM25Index` | `openhands.tools.retriever.bm25` | Raw BM25 index (usable standalone) |
| `BM25State` | `openhands.tools.retriever.bm25` | Shared mutable state between build/retrieve |
| `BaseRetrieverToolSet` | `openhands.tools.retriever.base` | Abstract base for custom backends |
| `BuildIndexAction` | `openhands.tools.retriever.base` | Action schema for `build_index` |
| `RetrieverAction` | `openhands.tools.retriever.base` | Action schema for `retrieve` |
| `RetrieverResult` | `openhands.tools.retriever.base` | Single result (`id`, `text`, `score`, `extra`) |
