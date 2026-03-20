import json
import logging
import os
import copy
import shutil
import tempfile
import traceback
import argparse
import difflib
import hashlib
import yaml
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Mapping, Sequence, Optional
from pathlib import Path

from gepa import optimize, GEPAAdapter, EvaluationBatch, GEPAResult

from openhands.sdk import LLM, Agent, Tool, Conversation
from openhands.sdk.context.agent_context import AgentContext
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.sdk.context import Skill

from .runner import run_with_agent
from .dataloader import prepare_task
from .utils import LM_DICT, batch_inference
from .review.trajectory_utils import convert_json_to_markdown
from .task_setups import preprocess_example, setup_servers, get_eval_config, get_seed_candidate


# ---------------------------------------------------------------------------
# Proposer system prompt
# ---------------------------------------------------------------------------

PROPOSER_SYSTEM_PROMPT = """You are an agent optimization expert. You improve an AI agent's task performance by analyzing execution trajectories and modifying the agent's configuration code.

## Workspace Layout

- `project/agent.py` — The current agent configuration code. Defines `build_agent(base_dir, lm_model, seed_prompt) -> Agent` using the OpenHands SDK, and optionally `get_workspace_scripts() -> dict[str, str]` for helper scripts deployed to the agent's workspace at runtime.
- `project/seed_prompt.txt` — The original task prompt (passed to `build_agent` as the `seed_prompt` parameter).
- `trajectories/` — Markdown traces of the agent's recent execution on training examples.
- `eval_results.yaml` — Per-example scores and detailed evaluation feedback.
- `proposal_memory/rejected_proposals.yaml` — Recent rejected edits, mainly for this exact parent candidate. Treat these as soft evidence about strategies that already failed to improve scores.

## Your Task

1. Read `eval_results.yaml` to understand overall performance and per-example scores.
2. Read the trajectory files in `trajectories/` to understand the agent's step-by-step behavior.
3. Read `proposal_memory/rejected_proposals.yaml` before editing. Avoid repeating the same non-improving strategy unless the current trajectories provide strong new evidence that the earlier rejection was minibatch-specific.
4. Identify failure modes: wrong actions, missing capabilities, excessive context, stuck loops, missing tools, poor instructions, etc.
5. Modify `project/agent.py` to address the identified failures. You have the full OpenHands SDK at your disposal — consult the SDK Reference skill for available APIs.
6. Optionally add/modify `get_workspace_scripts()` to deploy helper scripts to the agent's workspace.

## Constraints on agent.py

- MUST define `build_agent(base_dir: str, lm_model: str, seed_prompt: str) -> Agent`.
- `base_dir` is a temp directory where the function can write files (prompts, skills, etc.).
- `lm_model` is the model string to use (e.g., "vertex_ai/gemini-3-flash-preview").
- `seed_prompt` is the original task prompt text (passed as a parameter, also available in `project/seed_prompt.txt`).
- Code must be valid, self-contained Python with explicit imports at the top.
- Refer to the **SDK Reference** skill for the full API surface.
- Refer to the **Adaptation Guide** skill for strategies (augment context, trim context, adapt actions, adapt observations, adapt orchestration).

## Output

After modifying agent.py, read it back to verify there are no syntax errors. Make sure the code is complete and self-contained.
"""


# ---------------------------------------------------------------------------
# Candidate execution and validation
# ---------------------------------------------------------------------------

def execute_agent_candidate(
    code: str,
    base_dir: str,
    lm_model: str,
    seed_prompt: str,
) -> Agent:
    """Execute candidate code and return the constructed Agent.

    The candidate code must define build_agent(base_dir, lm_model, seed_prompt) -> Agent.
    """
    namespace = {}
    exec(code, namespace)
    return namespace["build_agent"](base_dir, lm_model, seed_prompt)


def extract_workspace_scripts(code: str) -> dict[str, str]:
    """Execute candidate code and extract optional workspace scripts.

    Returns empty dict if get_workspace_scripts is not defined.
    """
    namespace = {}
    exec(code, namespace)
    fn = namespace.get("get_workspace_scripts")
    if fn is None:
        return {}
    return fn()


def _validate_worker(code: str, lm_model: str, seed_prompt: str) -> tuple[bool, str]:
    """Subprocess worker for validate_agent_candidate.

    Runs exec() in an isolated process so custom Action/Observation class
    registrations (e.g. a redefined FinishAction) never pollute the main
    process's global Pydantic discriminated union.

    Also spawns a short Conversation and calls send_message to trigger
    _ensure_agent_ready() / tool resolution, catching errors like
    AttributeError in ToolDefinition.create() that only surface at init time.
    """
    import tempfile
    from openhands.sdk import Agent as _Agent, Conversation as _Conversation
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            namespace = {}
            exec(code, namespace)
            agent = namespace["build_agent"](tmp_dir, lm_model, seed_prompt)
            if not isinstance(agent, _Agent):
                return False, f"build_agent returned {type(agent).__name__}, expected Agent"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

        # Spawn a real Conversation to trigger tool resolution (_ensure_agent_ready).
        # Errors here (e.g. AttributeError in ToolDefinition.create) would silently
        # produce empty traces during actual evaluation.
        conversation = None
        try:
            conversation = _Conversation(agent=agent, workspace=tmp_dir)
            conversation.send_message("test")
        except Exception as e:
            return False, f"{type(e).__name__} during conversation init: {e}"
        finally:
            if conversation is not None:
                try:
                    conversation.close()
                except Exception:
                    pass

        return True, ""


def validate_agent_candidate(
    code: str,
    lm_model: str,
    seed_prompt: str,
) -> tuple[bool, str]:
    """Validate candidate code by compiling, executing, and building the Agent.

    Returns (success, error_message). On success, error_message is empty.
    The exec() step runs in a subprocess to avoid polluting the main process's
    Pydantic discriminated union with custom Action/Observation registrations.
    """
    # 1. Compile check (safe to do in-process — no side effects)
    try:
        compile(code, "agent.py", "exec")
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # 2. Exec + build_agent check in an isolated subprocess
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as pool:
        future = pool.submit(_validate_worker, code, lm_model, seed_prompt)
        return future.result()


# ---------------------------------------------------------------------------
# Eval feedback formatting (reused from original)
# ---------------------------------------------------------------------------

def _format_eval_feedback(output: dict, score: float) -> str:
    """Build rich evaluation feedback from eval result dict."""
    parts = [f"Score: {score}"]

    feedback = output.get("feedback")
    if feedback:
        parts.append(feedback)

    if "error" in output and output["error"]:
        parts.append(f"Error: {output['error']}")

    test_output = output.get("test_output")
    if test_output:
        for fr in test_output:
            if fr.get("status") in ("failed", "error", "timeout"):
                file_line = f"[{fr['status'].upper()}] {fr.get('file', '?')}"
                stderr = fr.get("stderr", "").strip()
                stdout = fr.get("stdout", "").strip()
                if stderr:
                    file_line += f"\n{stderr[:500]}"
                elif stdout:
                    file_line += f"\n{stdout[:500]}"
                parts.append(file_line)

    return "\n".join(parts)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _build_code_diff(old_text: str, new_text: str, max_lines: int = 120) -> str:
    diff_lines = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile="current.py",
            tofile="proposed.py",
            lineterm="",
        )
    )
    if not diff_lines:
        return "(no code changes)"
    if len(diff_lines) > max_lines:
        diff_lines = diff_lines[:max_lines] + ["... diff truncated ..."]
    return "\n".join(diff_lines)


def _summarize_score_changes(
    batch: Sequence[dict[str, Any]],
    before_scores: Sequence[float],
    after_scores: Sequence[float],
    limit: int = 3,
) -> list[dict[str, Any]]:
    changes = []
    for example, before, after in zip(batch, before_scores, after_scores):
        delta = after - before
        if abs(delta) < 1e-9:
            continue
        changes.append(
            {
                "task_input": example.get("prompt", "")[:300],
                "before": before,
                "after": after,
                "delta": delta,
            }
        )

    changes.sort(key=lambda item: (item["delta"], abs(item["delta"])))
    return changes[:limit]


# ---------------------------------------------------------------------------
# Skill loading helpers
# ---------------------------------------------------------------------------

def _create_skill_file(content: str, name: str, dest_dir: str) -> Skill:
    """Write content as a SKILL.md file and load it as a Skill object."""
    skill_dir = os.path.join(dest_dir, name)
    os.makedirs(skill_dir, exist_ok=True)
    skill_file = os.path.join(skill_dir, "SKILL.md")
    with open(skill_file, "w") as f:
        f.write(f"---\nname: {name}\n---\n\n{content}")
    skill = Skill.load(path=skill_file, strict=False)
    skill.is_agentskills_format = False
    return skill


def _load_proposer_skills(run_dir: str) -> list[Skill]:
    """Load SDK reference and adaptation guide as Skill objects for the proposer."""
    skills_dir = os.path.join(run_dir, "shared", "proposer_skills")
    os.makedirs(skills_dir, exist_ok=True)

    skills = []

    # SDK reference
    sdk_ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "sdk_reference.md")
    if os.path.exists(sdk_ref_path):
        with open(sdk_ref_path) as f:
            sdk_content = f.read()
        skills.append(_create_skill_file(sdk_content, "sdk_reference", skills_dir))

    # Adaptation guide
    adaptation_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "adaptation.md")
    if os.path.exists(adaptation_path):
        with open(adaptation_path) as f:
            adaptation_content = f.read()
        skills.append(_create_skill_file(adaptation_content, "adaptation_guide", skills_dir))

    return skills


# ---------------------------------------------------------------------------
# GEPA Adapter — agentic proposer
# ---------------------------------------------------------------------------

def _evaluate_single_worker(
    i: int,
    example: dict,
    candidate: dict,
    phase_dir: str,
    capture_traces: bool,
    task_id: str,
    model_name: str,
    task_prompt: str,
    eval_lm_name: Optional[str],
) -> dict:
    """Module-level worker for ProcessPoolExecutor: run candidate agent + eval for one example."""
    from .task_setups import get_eval_config
    from .utils import LM_DICT as _LM_DICT

    eval_function = get_eval_config(task_id)["eval_function"]
    eval_lm = _LM_DICT[eval_lm_name] if eval_lm_name else None

    example = copy.deepcopy(example)
    workspace_base = os.path.abspath(os.path.join(phase_dir, f"example{i}"))
    agent_base_dir = os.path.abspath(os.path.join(phase_dir, f"example{i}_config"))
    os.makedirs(workspace_base, exist_ok=True)
    os.makedirs(agent_base_dir, exist_ok=True)

    try:
        # Execute candidate code exactly once to avoid re-registering pydantic models
        # (e.g. CustomFinishAction) which causes "Duplicate class definition" errors.
        namespace = {}
        exec(candidate["agent_code"], namespace)
        agent = namespace["build_agent"](agent_base_dir, model_name, task_prompt)
        fn = namespace.get("get_workspace_scripts")
        workspace_scripts = fn() if fn is not None else {}

        result = run_with_agent(
            agent=agent,
            example=example,
            workspace=workspace_base,
            task_id=task_id,
            workspace_scripts=workspace_scripts,
        )

        # Copy trace files into workspace (eval checks for them there)
        log_dir = Path(workspace_base).parent / f"{Path(workspace_base).name}_logs"
        if log_dir.exists():
            for trace_file in log_dir.glob("trace_*.md"):
                shutil.copy2(trace_file, workspace_base)

        eval_kwargs = {
            "workspace_dir": workspace_base,
            "example": result or example,
            "lm": eval_lm,
        }
        eval_result = eval_function(**eval_kwargs)

        score = float(eval_result.get("score", 0.0))

        trajectory = None
        if capture_traces:
            trace_data = {}
            trace_files = list(log_dir.glob("trace_*.json")) if log_dir.exists() else []
            if trace_files:
                with open(trace_files[0]) as f:
                    trace_data = json.load(f)
            trace_data["eval_result"] = eval_result
            trajectory = trace_data

        return {"output": eval_result, "score": score, "trajectory": trajectory, "error_message": None}

    except Exception as e:
        traceback.print_exc()
        trajectory = {"error": str(e)} if capture_traces else None
        return {
            "output": {"error": str(e), "score": 0.0},
            "score": 0.0,
            "trajectory": trajectory,
            "error_message": f"Error on example {i}: {e}",
        }


class AgentOptimizationAdapter(GEPAAdapter):
    """GEPAAdapter that evolves agent code via an agentic proposer.

    The candidate is Python code defining build_agent(base_dir, lm_model, seed_prompt) -> Agent.
    The proposer is a full SDK agent that analyzes trajectories and modifies the code.
    """

    def __init__(
        self,
        task_id: str,
        model_name: str,
        logger: "GEPAFileLogger",
        reflection_lm_name: str,
        task_prompt: str = "",
        eval_lm_name: Optional[str] = None,
        run_dir: str = "results/gepa",
        max_proposal_retries: int = 2,
        batch_size: Optional[int] = None,
    ):
        self.task_id = task_id
        self.task_prompt = task_prompt
        self.model_name = LM_DICT[model_name].model
        self.logger = logger
        self.reflection_lm_name = reflection_lm_name
        self.run_dir = run_dir
        self.eval_lm_name = eval_lm_name
        self.eval_lm = LM_DICT[eval_lm_name] if eval_lm_name else None
        self._max_proposal_retries = max_proposal_retries
        self._batch_size = batch_size

        self.eval_config = get_eval_config(task_id)
        self.eval_function = self.eval_config["eval_function"]

        # Load proposer skills once
        self._proposer_skills = _load_proposer_skills(run_dir)

        # Iteration tracking
        self._gepa_iter = 0
        self._phase = "seed"
        self._calls_since_reflection = 0
        self._last_batch: list[dict] = []
        self._last_reflection_context: Optional[dict[str, Any]] = None
        self._pending_proposal_context: Optional[dict[str, Any]] = None

        os.makedirs(run_dir, exist_ok=True)

    def _eval_cache_path(self) -> str:
        return os.path.join(self.run_dir, "shared", "eval_cache.json")

    def _eval_cache_key(self, agent_code: str, example: dict, capture_traces: bool) -> str:
        code_hash = _hash_text(agent_code)
        try:
            example_hash = _hash_text(json.dumps(example, sort_keys=True, default=str))
        except Exception:
            example_hash = _hash_text(str(example))
        return f"{code_hash}_{example_hash}_{'t' if capture_traces else 'f'}"

    def _load_eval_cache(self) -> dict:
        path = self._eval_cache_path()
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    def _save_eval_cache(self, cache: dict):
        path = self._eval_cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(cache, f)

    def _iter_dir(self) -> str:
        if self._gepa_iter < 0:
            return os.path.join(self.run_dir, "iterations", "seed")
        return os.path.join(self.run_dir, "iterations", f"iter_{self._gepa_iter:02d}")

    def _phase_dir(self) -> str:
        base = self._iter_dir()
        if self._phase in ("seed", "reflection"):
            return os.path.join(base, "current")
        if self._phase == "candidate":
            return os.path.join(base, "proposed")
        return os.path.join(base, "proposed_val")

    def _proposal_memory_dir(self) -> str:
        return os.path.join(self.run_dir, "shared", "proposal_memory")

    def _proposal_memory_path(self) -> str:
        return os.path.join(self._proposal_memory_dir(), "rejected_proposals.jsonl")

    def _iter_proposal_metadata_path(self) -> str:
        return os.path.join(self._iter_dir(), "proposer", "proposal_metadata.json")

    def _iter_proposal_outcome_path(self) -> str:
        return os.path.join(self._iter_dir(), "proposer", "proposal_outcome.json")

    def _load_rejected_proposals(self) -> list[dict[str, Any]]:
        path = self._proposal_memory_path()
        if not os.path.exists(path):
            return []

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    self.logger.log(f"[iter {self._gepa_iter}] Skipping invalid proposal memory record")
        return records

    def _save_rejected_proposals(self, records: list[dict[str, Any]]):
        path = self._proposal_memory_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for record in records[-200:]:
                f.write(json.dumps(record, default=str))
                f.write("\n")

    def _append_rejected_proposal(self, record: dict[str, Any]):
        records = self._load_rejected_proposals()
        records.append(record)
        self._save_rejected_proposals(records)

    def _select_relevant_rejections(
        self,
        candidate: dict[str, str],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        candidate_hash = _hash_text(candidate["agent_code"])
        records = self._load_rejected_proposals()
        if not records:
            return []

        same_parent = [r for r in records if r.get("parent_candidate_hash") == candidate_hash]
        recent_other = [r for r in records if r.get("parent_candidate_hash") != candidate_hash]

        selected = same_parent[-limit:]
        remaining = max(0, limit - len(selected))
        if remaining > 0:
            selected.extend(recent_other[-min(2, remaining):])
        return selected

    def _write_proposal_metadata(self, record: dict[str, Any]):
        path = self._iter_proposal_metadata_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def _write_proposal_outcome(self, record: dict[str, Any]):
        path = self._iter_proposal_outcome_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def _record_reflection_context(
        self,
        batch: Sequence[dict[str, Any]],
        candidate: dict[str, str],
        scores: list[float],
    ):
        self._pending_proposal_context = None
        self._last_reflection_context = {
            "iteration": self._gepa_iter,
            "parent_candidate_hash": _hash_text(candidate["agent_code"]),
            "before_scores": list(scores),
            "before_score_sum": sum(scores),
            "batch_prompts": [example.get("prompt", "")[:300] for example in batch],
        }

    def _record_candidate_outcome(
        self,
        batch: Sequence[dict[str, Any]],
        candidate: dict[str, str],
        scores: list[float],
        outputs: list[dict[str, Any]],
    ):
        pending = self._pending_proposal_context
        reflection = self._last_reflection_context
        if pending is None or reflection is None:
            return

        proposed_hash = _hash_text(candidate["agent_code"])
        if proposed_hash != pending.get("proposed_candidate_hash"):
            return

        before_scores = list(reflection.get("before_scores", []))
        if len(before_scores) != len(scores):
            self.logger.log(
                f"[iter {self._gepa_iter}] Skipping proposal outcome record due to score length mismatch"
            )
            self._pending_proposal_context = None
            return

        before_sum = float(reflection.get("before_score_sum", 0.0))
        after_sum = sum(scores)
        delta = after_sum - before_sum

        per_example_feedback = [
            {
                "prompt": example.get("prompt", "")[:300],
                "score": score,
                "feedback": output.get("feedback", "") if isinstance(output, dict) else "",
            }
            for example, score, output in zip(batch, scores, outputs)
        ]

        outcome = {
            **pending,
            "status": "accepted_on_subsample" if delta > 0 else "rejected_on_subsample",
            "batch_prompts": list(reflection.get("batch_prompts", [])),
            "before_scores": before_scores,
            "after_scores": list(scores),
            "before_score_sum": before_sum,
            "after_score_sum": after_sum,
            "score_delta": delta,
            "score_delta_examples": _summarize_score_changes(batch, before_scores, scores),
            "per_example_feedback": per_example_feedback,
        }
        self._write_proposal_outcome(outcome)

        if delta <= 0:
            self._append_rejected_proposal(outcome)

        self._pending_proposal_context = None

    def _save_candidate(self, candidate: dict[str, str]):
        iter_dir = self._iter_dir()
        candidates_dir = os.path.join(iter_dir, "candidates")
        os.makedirs(candidates_dir, exist_ok=True)
        filename = "current.py" if self._phase == "reflection" else "proposed.py"
        path = os.path.join(candidates_dir, filename)
        with open(path, "w") as f:
            f.write(candidate["agent_code"])
        self.logger.log(f"[iter {self._gepa_iter}, {self._phase}] Saved candidate to {path}")

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        # Determine phase
        if capture_traces:
            self._gepa_iter += 1
            self._phase = "reflection"
            self._calls_since_reflection = 0
        elif self._gepa_iter < 0:
            self._phase = "seed"
        else:
            self._calls_since_reflection += 1
            self._phase = "candidate" if self._calls_since_reflection == 1 else "valset"

        self._last_batch = batch
        phase_dir = self._phase_dir()
        self._save_candidate(candidate)

        self.logger.log(
            f"[iter {self._gepa_iter}, {self._phase}] Evaluating {len(batch)} examples "
            f"(capture_traces={capture_traces})"
        )

        # Run all examples in separate processes (browser tool requires process isolation)
        max_workers = self._batch_size if self._batch_size is not None else len(batch)
        args_list = [
            {
                "i": i,
                "example": ex,
                "candidate": candidate,
                "phase_dir": phase_dir,
                "capture_traces": capture_traces,
                "task_id": self.task_id,
                "model_name": self.model_name,
                "task_prompt": self.task_prompt,
                "eval_lm_name": self.eval_lm_name,
            }
            for i, ex in enumerate(batch)
        ]

        # Check cache — only dispatch examples not already evaluated with this code
        eval_cache = self._load_eval_cache()
        results_list = [None] * len(args_list)
        uncached_indices = []
        n_hits = 0
        for idx, args in enumerate(args_list):
            key = self._eval_cache_key(candidate["agent_code"], args["example"], capture_traces)
            if key in eval_cache:
                results_list[idx] = eval_cache[key]
                n_hits += 1
            else:
                uncached_indices.append(idx)

        if n_hits:
            self.logger.log(
                f"[iter {self._gepa_iter}, {self._phase}] Cache hits: {n_hits}/{len(batch)}"
            )

        if uncached_indices:
            uncached_args = [args_list[idx] for idx in uncached_indices]
            fresh_results = batch_inference(
                _evaluate_single_worker, uncached_args, max_workers=max_workers, use_process=True
            )
            for idx, result in zip(uncached_indices, fresh_results):
                results_list[idx] = result
                key = self._eval_cache_key(candidate["agent_code"], args_list[idx]["example"], capture_traces)
                eval_cache[key] = result
            self._save_eval_cache(eval_cache)

        outputs = []
        scores = []
        trajectories = [] if capture_traces else None
        for r in results_list:
            if r.get("error_message"):
                self.logger.log(f"[iter {self._gepa_iter}, {self._phase}] {r['error_message']}")
            outputs.append(r["output"])
            scores.append(r["score"])
            if capture_traces:
                trajectories.append(r["trajectory"])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        self.logger.log(
            f"[iter {self._gepa_iter}, {self._phase}] "
            f"scores: {scores}, avg: {avg_score:.3f}"
        )

        if capture_traces:
            self._record_reflection_context(batch, candidate, scores)
        elif self._phase == "candidate":
            self._record_candidate_outcome(batch, candidate, scores, outputs)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    # ------------------------------------------------------------------
    # Reflective dataset: prepare trajectory data for the proposer
    # ------------------------------------------------------------------

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        result = {}

        for component in components_to_update:
            records = []
            for i, (example, output, score) in enumerate(
                zip(self._last_batch, eval_batch.outputs, eval_batch.scores)
            ):
                trajectory = (
                    eval_batch.trajectories[i]
                    if eval_batch.trajectories
                    else {}
                )

                if component == "agent_code":
                    trajectory_md = ""
                    if trajectory and "events" in trajectory:
                        try:
                            trajectory_md = convert_json_to_markdown(trajectory)
                        except Exception:
                            trajectory_md = "(trajectory conversion failed)"

                    feedback = _format_eval_feedback(output, score) if isinstance(output, dict) else f"Score: {score}"

                    records.append({
                        "Task Input": example.get("prompt", ""),
                        "Agent Trajectory": trajectory_md,
                        "Evaluation Feedback": feedback,
                        "eval_output": output if isinstance(output, dict) else {},
                    })

            result[component] = records

        # Save reflective dataset
        iter_dir = self._iter_dir()
        proposer_dir = os.path.join(iter_dir, "proposer")
        os.makedirs(proposer_dir, exist_ok=True)
        reflection_path = os.path.join(proposer_dir, "reflection.json")
        with open(reflection_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        self.logger.log(f"[iter {self._gepa_iter}] Reflection inputs saved to {reflection_path}")

        return result

    # ------------------------------------------------------------------
    # Agentic proposer: run an agent to improve the candidate code
    # ------------------------------------------------------------------

    def _setup_proposer_workspace(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> str:
        """Create the proposer agent's workspace with all context."""
        iter_dir = self._iter_dir()
        workspace = os.path.join(iter_dir, "proposer", "workspace")
        if os.path.exists(workspace):
            shutil.rmtree(workspace)

        # project/ — current candidate code + seed prompt
        project_dir = os.path.join(workspace, "project")
        os.makedirs(project_dir)
        with open(os.path.join(project_dir, "agent.py"), "w") as f:
            f.write(candidate["agent_code"])
        with open(os.path.join(project_dir, "seed_prompt.txt"), "w") as f:
            f.write(self.task_prompt)

        # trajectories/ — one markdown file per training example
        traj_dir = os.path.join(workspace, "trajectories")
        os.makedirs(traj_dir)
        dataset = reflective_dataset.get("agent_code", [])
        for i, record in enumerate(dataset):
            md_parts = [f"# Example {i}"]
            md_parts.append(f"\n## Task Input\n{record.get('Task Input', '')}")
            md_parts.append(f"\n## Evaluation Feedback\n{record.get('Evaluation Feedback', '')}")
            trajectory_md = record.get("Agent Trajectory", "")
            if trajectory_md:
                md_parts.append(f"\n## Agent Trajectory\n{trajectory_md}")
            with open(os.path.join(traj_dir, f"example_{i:02d}.md"), "w") as f:
                f.write("\n".join(md_parts))

        # eval_results.yaml — summary of scores and feedback
        eval_summary = []
        for i, record in enumerate(dataset):
            eval_out = record.get("eval_output", {})
            entry = {
                "example": i,
                "feedback": record.get("Evaluation Feedback", ""),
            }
            for key in ("task_id", "score", "answer", "reference_answers", "eval_types"):
                if key in eval_out:
                    entry[key] = eval_out[key]
            eval_summary.append(entry)
        with open(os.path.join(workspace, "eval_results.yaml"), "w") as f:
            yaml.dump(eval_summary, f, sort_keys=False, default_flow_style=False)

        memory_dir = os.path.join(workspace, "proposal_memory")
        os.makedirs(memory_dir)
        rejected_proposals = self._select_relevant_rejections(candidate)
        memory_payload = rejected_proposals or [
            {
                "status": "no_prior_rejections",
                "note": "No prior rejected proposals recorded for this candidate yet.",
            }
        ]
        with open(os.path.join(memory_dir, "rejected_proposals.yaml"), "w") as f:
            yaml.dump(memory_payload, f, sort_keys=False, default_flow_style=False)

        return workspace

    def _build_proposer_agent(self) -> Agent:
        """Construct the proposer agent with SDK reference and adaptation skills."""
        lm = LM_DICT[self.reflection_lm_name]
        proposer_llm = LLM(model=lm.model)

        proposer_tools = [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ]

        agent_context = AgentContext(skills=self._proposer_skills)

        # Write the proposer system prompt to a temp file
        prompt_dir = os.path.join(self.run_dir, "shared", "proposer_prompt")
        os.makedirs(prompt_dir, exist_ok=True)
        prompt_path = os.path.join(prompt_dir, "system_prompt.md")
        with open(prompt_path, "w") as f:
            f.write(PROPOSER_SYSTEM_PROMPT)

        return Agent(
            llm=proposer_llm,
            tools=proposer_tools,
            system_prompt_filename=os.path.abspath(prompt_path),
            agent_context=agent_context,
        )

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Run the agentic proposer to improve the candidate code."""
        new_candidate = dict(candidate)
        self._pending_proposal_context = None

        if "agent_code" not in components_to_update:
            return new_candidate

        # 1. Set up proposer workspace
        workspace = self._setup_proposer_workspace(candidate, reflective_dataset)
        self.logger.log(f"[iter {self._gepa_iter}] Proposer workspace: {workspace}")

        # 2. Build proposer agent
        proposer_agent = self._build_proposer_agent()

        # 3. Run proposer conversation
        instruction = (
            "Analyze the agent's performance by reading eval_results.yaml and the trajectory files "
            "in trajectories/. Read proposal_memory/rejected_proposals.yaml before editing so you "
            "can avoid repeating non-improving changes. Then improve the agent configuration in "
            "project/agent.py to address the identified failure modes. Use the SDK Reference and "
            "Adaptation Guide skills for available APIs and strategies."
        )

        conversation = Conversation(agent=proposer_agent, workspace=workspace)
        conversation.send_message(instruction)

        last_error = None
        for attempt in range(1 + self._max_proposal_retries):
            if attempt > 0:
                # Resume conversation with validation error
                conversation.send_message(
                    f"Your modified agent.py failed validation with this error:\n"
                    f"```\n{last_error}\n```\n"
                    f"Please fix the code in project/agent.py and ensure "
                    f"`build_agent(base_dir, lm_model, seed_prompt)` returns a valid Agent."
                )

            try:
                conversation.run()
            except Exception as e:
                self.logger.log(f"[iter {self._gepa_iter}] Proposer conversation error: {e}")
                traceback.print_exc()
                break

            # 4. Read back the modified agent.py
            agent_py_path = os.path.join(workspace, "project", "agent.py")
            try:
                with open(agent_py_path) as f:
                    new_code = f.read()
            except FileNotFoundError:
                self.logger.log(f"[iter {self._gepa_iter}] Proposer did not produce agent.py")
                break

            # 5. Validate
            success, error = validate_agent_candidate(new_code, self.model_name, self.task_prompt)
            if success:
                new_candidate["agent_code"] = new_code
                proposal_record = {
                    "iteration": self._gepa_iter,
                    "parent_candidate_hash": _hash_text(candidate["agent_code"]),
                    "proposed_candidate_hash": _hash_text(new_code),
                    "components_to_update": list(components_to_update),
                    "code_diff": _build_code_diff(candidate["agent_code"], new_code),
                }
                self._pending_proposal_context = proposal_record
                self._write_proposal_metadata(proposal_record)
                if attempt > 0:
                    self.logger.log(
                        f"[iter {self._gepa_iter}] Proposal succeeded on attempt {attempt + 1}"
                    )
                last_error = None
                break
            else:
                last_error = error
                self.logger.log(
                    f"[iter {self._gepa_iter}] Proposal attempt {attempt + 1} "
                    f"validation failed: {last_error}"
                )

        if last_error:
            self.logger.log(
                f"[iter {self._gepa_iter}] All {1 + self._max_proposal_retries} proposal "
                f"attempts failed, keeping current candidate"
            )

        # Save proposer conversation trace
        try:
            proposer_log_dir = os.path.join(self._iter_dir(), "proposer", "workspace_logs")
            os.makedirs(proposer_log_dir, exist_ok=True)
            from .runner import save_conversation_trace
            save_conversation_trace(conversation, proposer_log_dir)
        except Exception as e:
            self.logger.log(f"[iter {self._gepa_iter}] Failed to save proposer trace: {e}")

        conversation.close()
        return new_candidate


# ---------------------------------------------------------------------------
# GEPA Logger
# ---------------------------------------------------------------------------

class GEPAFileLogger:
    """Logger that writes GEPA messages to a file and stderr."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        state_dir = os.path.join(run_dir, "shared", "state")
        os.makedirs(state_dir, exist_ok=True)
        self.log_path = os.path.join(state_dir, "gepa.log")

        self._logger = logging.getLogger(f"gepa.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        fh = logging.FileHandler(self.log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self._logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(sh)

    def log(self, message: str):
        self._logger.info(message)


# ---------------------------------------------------------------------------
# Main optimization entry point
# ---------------------------------------------------------------------------

def run_optimization(
    task_id: str,
    model_name: str,
    prompt_name: str = "default",
    max_examples: Optional[int] = None,
    train_ratio: float = 0.7,
    eval_lm_name: Optional[str] = None,
    reflection_lm: str = "gemini-3-flash-preview",
    max_metric_calls: int = 50,
    seed: int = 0,
    run_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    # 0. Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Load data
    data, task_prompt, eval_prompt = prepare_task(
        task_id=task_id,
        model_name=model_name,
        rollout_version="gepa",
        prompt_name=prompt_name,
        max_examples=max_examples,
        data_path=data_path,
    )

    # Apply task-specific server setup and example preprocessing
    setup_servers(task_id)
    data = [preprocess_example(task_id, ex) for ex in data]

    # 2. Split train/val
    split_idx = max(1, int(len(data) * train_ratio))
    trainset = data[:split_idx]
    valset = data[split_idx:] if split_idx < len(data) else data

    # 3. Build seed candidate
    seed_candidate = get_seed_candidate(task_id)
    seed_code = seed_candidate["agent_code"]

    # 4. Set up run directory
    if run_dir is None:
        run_dir = f"results/{task_id}/{model_name}_{prompt_name}/gepa/seed{seed}"
    os.makedirs(run_dir, exist_ok=True)

    gepa_logger = GEPAFileLogger(run_dir)
    gepa_logger.log(f"Task: {task_id}, Model: {model_name}, Prompt: {prompt_name}")
    gepa_logger.log(f"Dataset: {len(data)} total, {len(trainset)} train, {len(valset)} val")
    gepa_logger.log(f"Max metric calls: {max_metric_calls}, Reflection LM: {reflection_lm}")

    # Save seed candidate
    config_dir = os.path.join(run_dir, "shared", "config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "seed_config.py"), "w") as f:
        f.write(seed_code)

    # 5. Create adapter
    adapter = AgentOptimizationAdapter(
        task_id=task_id,
        model_name=model_name,
        logger=gepa_logger,
        reflection_lm_name=reflection_lm,
        task_prompt=task_prompt,
        eval_lm_name=eval_lm_name,
        run_dir=run_dir,
        batch_size=batch_size,
    )

    # 6. Run GEPA optimization
    result: GEPAResult = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        seed=seed,
        run_dir=run_dir,
        display_progress_bar=True,
        logger=gepa_logger,
    )

    # 7. Save results
    best = result.best_candidate
    best_score = result.val_aggregate_scores[result.best_idx]

    config_dir = os.path.join(run_dir, "shared", "config")
    os.makedirs(config_dir, exist_ok=True)

    best_config_path = os.path.join(config_dir, "best_config.py")
    with open(best_config_path, "w") as f:
        f.write(best["agent_code"])

    all_candidates_dir = os.path.join(config_dir, "all_candidates")
    os.makedirs(all_candidates_dir, exist_ok=True)
    for idx, (cand, score) in enumerate(zip(result.candidates, result.val_aggregate_scores)):
        cand_path = os.path.join(all_candidates_dir, f"candidate_{idx}_score{score:.3f}.py")
        with open(cand_path, "w") as f:
            f.write(f"# val_score: {score}\n")
            f.write(cand["agent_code"])

    summary = {
        "best_score": best_score,
        "best_idx": result.best_idx,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "all_scores": result.val_aggregate_scores,
    }
    summary_path = os.path.join(config_dir, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Best score: {best_score}")
    print(f"Best config saved to: {best_config_path}")
    print(f"Summary saved to: {summary_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agentic GEPA optimization")

    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--task_id", type=str, help="Task identifier")
    parser.add_argument("--model_name", type=str, help="Model name for agent execution")
    parser.add_argument("--prompt_name", type=str, default=None, help="Prompt name for seed")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--reflection_lm", type=str, default=None, help="LM for proposer agent")
    parser.add_argument("--eval_lm", type=str, default=None, help="LM for evaluation")
    parser.add_argument("--max_metric_calls", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {args.config}")

    task_id = args.task_id or config.get("task_id")
    model_name = args.model_name or config.get("model_name")
    prompt_name = args.prompt_name or config.get("prompt_name", "default")
    max_examples = args.max_examples if args.max_examples is not None else config.get("max_examples")
    train_ratio = args.train_ratio if args.train_ratio is not None else config.get("train_ratio", 0.7)
    reflection_lm = args.reflection_lm or config.get("reflection_lm", "gemini-3-flash-preview")
    eval_lm_name = args.eval_lm or config.get("eval_lm")
    max_metric_calls = args.max_metric_calls if args.max_metric_calls is not None else config.get("max_metric_calls", 50)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    run_dir = args.run_dir or config.get("run_dir")
    data_path = config.get("data_path")
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size")

    if not task_id:
        parser.error("--task_id is required")
    if not model_name:
        parser.error("--model_name is required")

    run_optimization(
        task_id=task_id,
        model_name=model_name,
        prompt_name=prompt_name,
        max_examples=max_examples,
        train_ratio=train_ratio,
        eval_lm_name=eval_lm_name,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        seed=seed,
        run_dir=run_dir,
        data_path=data_path,
        batch_size=batch_size,
    )
