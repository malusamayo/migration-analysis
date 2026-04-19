import copy
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from .common import hash_text


class EvalCache:
    """Persistent JSON cache keyed by (agent_code, example)."""

    def __init__(self, path: str):
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not os.path.exists(self._path):
            return {}
        with open(self._path) as f:
            return json.load(f)

    @staticmethod
    def _find_log_file(log_dir: str | None, pattern: str) -> str | None:
        trace_dir = Path(log_dir) if log_dir else None
        if trace_dir is None or not trace_dir.exists():
            return None

        matches = sorted(trace_dir.glob(pattern))
        if not matches:
            return None
        return str(matches[0])

    @staticmethod
    def load_trace_payload(
        log_dir: str | None,
        eval_result: dict | None,
    ) -> dict | None:
        trace_dir = Path(log_dir) if log_dir else None
        if trace_dir is None or not trace_dir.exists():
            return None

        trace_files = sorted(trace_dir.glob("trace_*.json"))
        if not trace_files:
            return None

        with open(trace_files[0]) as f:
            trace_data = json.load(f)
        if eval_result is not None:
            trace_data["eval_result"] = eval_result
        return trace_data

    @staticmethod
    def attach_artifact_metadata(
        result: dict,
        workspace_dir: str,
        log_dir: str,
        config_dir: str,
    ) -> dict:
        result["artifact_workspace_dir"] = workspace_dir
        result["artifact_log_dir"] = log_dir
        result["artifact_config_dir"] = config_dir
        result["trace_json_path"] = EvalCache._find_log_file(log_dir, "trace_*.json")
        result["raw_trace_json_path"] = EvalCache._find_log_file(log_dir, "raw_trace_*.json")
        return result

    def _infer_artifact_dirs(self, result: dict) -> tuple[str | None, str | None, str | None]:
        workspace_dir = result.get("artifact_workspace_dir")
        log_dir = result.get("artifact_log_dir")
        config_dir = result.get("artifact_config_dir")

        output = result.get("output")
        if workspace_dir is None and isinstance(output, dict):
            workspace_dir = output.get("workspace_dir")

        if workspace_dir is not None:
            workspace_path = Path(workspace_dir)
            if log_dir is None:
                log_dir = str(workspace_path.parent / f"{workspace_path.name}_logs")
            if config_dir is None:
                config_dir = str(workspace_path.parent / f"{workspace_path.name}_config")

        return workspace_dir, log_dir, config_dir

    def _copy_artifact_dir(self, source_dir: str | None, target_dir: str) -> None:
        if source_dir is None:
            return

        source_path = Path(source_dir)
        if not source_path.exists():
            return

        target_path = Path(target_dir)
        if source_path.resolve() == target_path.resolve():
            return

        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(source_path, target_path)

    def _prepare_cached_result(
        self,
        cached_result: dict,
        target_workspace_dir: str,
    ) -> tuple[dict, bool]:
        updated_cache = False
        source_workspace_dir, source_log_dir, source_config_dir = self._infer_artifact_dirs(cached_result)

        if source_workspace_dir is not None and cached_result.get("artifact_workspace_dir") is None:
            cached_result["artifact_workspace_dir"] = source_workspace_dir
            updated_cache = True
        if source_log_dir is not None and cached_result.get("artifact_log_dir") is None:
            cached_result["artifact_log_dir"] = source_log_dir
            updated_cache = True
        if source_config_dir is not None and cached_result.get("artifact_config_dir") is None:
            cached_result["artifact_config_dir"] = source_config_dir
            updated_cache = True
        if cached_result.get("trace_json_path") is None:
            cached_result["trace_json_path"] = self._find_log_file(source_log_dir, "trace_*.json")
            updated_cache = updated_cache or cached_result["trace_json_path"] is not None
        if cached_result.get("raw_trace_json_path") is None:
            cached_result["raw_trace_json_path"] = self._find_log_file(source_log_dir, "raw_trace_*.json")
            updated_cache = updated_cache or cached_result["raw_trace_json_path"] is not None

        if cached_result.get("trajectory") is None:
            recovered_trajectory = self.load_trace_payload(source_log_dir, cached_result.get("output"))
            if recovered_trajectory is not None:
                cached_result["trajectory"] = recovered_trajectory
                updated_cache = True

        prepared_result = copy.deepcopy(cached_result)
        target_workspace_path = Path(target_workspace_dir)
        target_log_dir = str(target_workspace_path.parent / f"{target_workspace_path.name}_logs")
        target_config_dir = str(target_workspace_path.parent / f"{target_workspace_path.name}_config")

        self._copy_artifact_dir(source_workspace_dir, target_workspace_dir)
        self._copy_artifact_dir(source_log_dir, target_log_dir)
        self._copy_artifact_dir(source_config_dir, target_config_dir)

        output = prepared_result.get("output")
        if isinstance(output, dict) and "workspace_dir" in output:
            prepared_result["output"] = dict(output)
            prepared_result["output"]["workspace_dir"] = target_workspace_dir

        prepared_result["trace_json_path"] = self._find_log_file(target_log_dir, "trace_*.json")
        prepared_result["raw_trace_json_path"] = self._find_log_file(target_log_dir, "raw_trace_*.json")

        return prepared_result, updated_cache

    def _key(self, agent_code: str, example: dict) -> str:
        code_hash = hash_text(agent_code)
        try:
            example_hash = hash_text(json.dumps(example, sort_keys=True, default=str))
        except Exception:
            example_hash = hash_text(str(example))
        return f"{code_hash}_{example_hash}"

    def get(self, agent_code: str, example: dict,) -> Optional[dict]:
        return self._data.get(self._key(agent_code, example))

    def get_prepared(
        self,
        agent_code: str,
        example: dict,
        target_workspace_dir: str,
    ) -> tuple[Optional[dict], bool]:
        cached_result = self.get(agent_code, example)
        if cached_result is None:
            return None, False
        return self._prepare_cached_result(cached_result, target_workspace_dir)

    def put(self, agent_code: str, example: dict, result: dict) -> None:
        self._data[self._key(agent_code, example)] = result

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f)
