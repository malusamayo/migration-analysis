import json
import os
from typing import Optional

from .common import hash_text


class EvalCache:
    """Persistent JSON cache keyed by (agent_code, example, capture_traces)."""

    def __init__(self, path: str):
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self._path):
            return {}
        with open(self._path) as f:
            return json.load(f)

    def _key(self, agent_code: str, example: dict, capture_traces: bool) -> str:
        code_hash = hash_text(agent_code)
        try:
            example_hash = hash_text(json.dumps(example, sort_keys=True, default=str))
        except Exception:
            example_hash = hash_text(str(example))
        return f"{code_hash}_{example_hash}_{'t' if capture_traces else 'f'}"

    def get(self, agent_code: str, example: dict, capture_traces: bool) -> Optional[dict]:
        return self._data.get(self._key(agent_code, example, capture_traces))

    def put(self, agent_code: str, example: dict, capture_traces: bool, result: dict) -> None:
        self._data[self._key(agent_code, example, capture_traces)] = result

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f)
