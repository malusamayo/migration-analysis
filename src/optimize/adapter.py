import copy
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gepa import EvaluationBatch, GEPAAdapter
from gepa.core.state import GEPAState

from .common import (
    add_to_cost_bucket,
    empty_cost_bucket,
    extract_dspy_cost,
    format_eval_feedback,
    hash_text,
)
from .proposer import AgentProposer
from ..review.trajectory_utils import convert_json_to_markdown
from ..runner import run_with_agent
from ..task_setups import get_eval_config
from ..utils import LM_DICT, batch_inference


class CostBudgetStopper:
    """Stop optimization when total accumulated cost exceeds a budget in USD."""

    def __init__(self, max_cost: float, cost_tracker: dict[str, dict]):
        self.max_cost = max_cost
        self._cost_tracker = cost_tracker

    def __call__(self, gepa_state: GEPAState) -> bool:
        total = sum(bucket["accumulated_cost"] for bucket in self._cost_tracker.values())
        return total >= self.max_cost


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
    """Run candidate agent + eval for one example in an isolated worker process."""
    from ..task_setups import get_eval_config
    from ..utils import LM_DICT as _LM_DICT

    eval_function = get_eval_config(task_id)["eval_function"]
    eval_lm = _LM_DICT[eval_lm_name] if eval_lm_name else None

    example = copy.deepcopy(example)
    workspace_base = os.path.abspath(os.path.join(phase_dir, f"example{i}"))
    agent_base_dir = os.path.abspath(os.path.join(phase_dir, f"example{i}_config"))
    os.makedirs(workspace_base, exist_ok=True)
    os.makedirs(agent_base_dir, exist_ok=True)

    try:
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

        log_dir = Path(workspace_base).parent / f"{Path(workspace_base).name}_logs"
        if log_dir.exists():
            for trace_file in log_dir.glob("trace_*.md"):
                shutil.copy2(trace_file, workspace_base)

        eval_kwargs = {
            "workspace_dir": workspace_base,
            "example": result or example,
            "lm": eval_lm,
        }
        eval_history_before = (
            len(eval_lm.history) if eval_lm is not None and hasattr(eval_lm, "history") else 0
        )
        eval_result = eval_function(**eval_kwargs)
        eval_metrics = extract_dspy_cost(eval_lm, eval_history_before)
        score = float(eval_result.get("score", 0.0))

        rollout_metrics = None
        if result and "run_result" in result:
            rollout_metrics = result["run_result"].get("metrics")

        trajectory = None
        if capture_traces:
            trace_data = {}
            trace_files = list(log_dir.glob("trace_*.json")) if log_dir.exists() else []
            if trace_files:
                with open(trace_files[0]) as f:
                    trace_data = json.load(f)
            trace_data["eval_result"] = eval_result
            trajectory = trace_data

        return {
            "output": eval_result,
            "score": score,
            "trajectory": trajectory,
            "rollout_metrics": rollout_metrics,
            "eval_metrics": eval_metrics,
            "error_message": None,
        }
    except Exception as e:
        traceback.print_exc()
        trajectory = {"error": str(e)} if capture_traces else None
        return {
            "output": {"error": str(e), "score": 0.0},
            "score": 0.0,
            "trajectory": trajectory,
            "rollout_metrics": None,
            "eval_metrics": None,
            "error_message": f"Error on example {i}: {e}",
        }


class AgentOptimizationAdapter(GEPAAdapter):
    """GEPA adapter that evolves agent code via an agentic proposer."""

    def __init__(
        self,
        task_id: str,
        model_name: str,
        logger,
        reflection_lm_name: str,
        task_prompt: str = "",
        eval_lm_name: Optional[str] = None,
        run_dir: str = "results/gepa",
        max_proposal_retries: int = 2,
        batch_size: Optional[int] = None,
        use_adaptation_guide: bool = True,
        adaptation_guide_markdown: Optional[str] = None,
    ):
        self.task_id = task_id
        self.task_prompt = task_prompt
        self.model_name = LM_DICT[model_name].model
        self.logger = logger
        self.reflection_lm_name = reflection_lm_name
        self.run_dir = run_dir
        self.eval_lm_name = eval_lm_name
        self.eval_lm = LM_DICT[eval_lm_name] if eval_lm_name else None
        self._batch_size = batch_size

        self.eval_config = get_eval_config(task_id)
        self.eval_function = self.eval_config["eval_function"]

        self._gepa_iter = 0
        self._phase = "seed"
        self._calls_since_reflection = 0
        self._last_batch: list[dict] = []

        self._cost_tracker: dict[str, dict] = {
            "rollout_seed": empty_cost_bucket(),
            "rollout_reflection": empty_cost_bucket(),
            "rollout_candidate": empty_cost_bucket(),
            "rollout_valset": empty_cost_bucket(),
            "eval_lm": empty_cost_bucket(),
            "proposer": empty_cost_bucket(),
        }

        self.proposer = AgentProposer(
            run_dir=run_dir,
            task_prompt=task_prompt,
            model_name=self.model_name,
            reflection_lm_name=reflection_lm_name,
            logger=logger,
            cost_tracker=self._cost_tracker,
            save_cost_summary=self._save_cost_summary,
            max_proposal_retries=max_proposal_retries,
            use_adaptation_guide=use_adaptation_guide,
            adaptation_guide_markdown=adaptation_guide_markdown,
        )

        os.makedirs(run_dir, exist_ok=True)

    def _eval_cache_path(self) -> str:
        return os.path.join(self.run_dir, "shared", "eval_cache.json")

    def _eval_cache_key(self, agent_code: str, example: dict, capture_traces: bool) -> str:
        code_hash = hash_text(agent_code)
        try:
            example_hash = hash_text(json.dumps(example, sort_keys=True, default=str))
        except Exception:
            example_hash = hash_text(str(example))
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

    def _cost_summary_path(self) -> str:
        return os.path.join(self.run_dir, "shared", "cost_summary.json")

    def _save_cost_summary(self) -> None:
        total = empty_cost_bucket()
        for bucket in self._cost_tracker.values():
            add_to_cost_bucket(total, bucket)
        summary = {**self._cost_tracker, "total": total}
        path = self._cost_summary_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.log(
            f"[cost] total=${total['accumulated_cost']:.4f} "
            f"(rollout_reflection=${self._cost_tracker['rollout_reflection']['accumulated_cost']:.4f}, "
            f"rollout_candidate=${self._cost_tracker['rollout_candidate']['accumulated_cost']:.4f}, "
            f"eval_lm=${self._cost_tracker['eval_lm']['accumulated_cost']:.4f}, "
            f"proposer=${self._cost_tracker['proposer']['accumulated_cost']:.4f})"
        )

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
        if capture_traces:
            self._gepa_iter += 1
            self._phase = "reflection"
            self._calls_since_reflection = 0
        elif self._gepa_iter == 0:
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
            self.logger.log(f"[iter {self._gepa_iter}, {self._phase}] Cache hits: {n_hits}/{len(batch)}")

        if uncached_indices:
            uncached_args = [args_list[idx] for idx in uncached_indices]
            fresh_results = batch_inference(
                _evaluate_single_worker,
                uncached_args,
                max_workers=max_workers,
                use_process=True,
            )
            bucket_key = f"rollout_{self._phase}"
            for idx, result in zip(uncached_indices, fresh_results):
                results_list[idx] = result
                key = self._eval_cache_key(candidate["agent_code"], args_list[idx]["example"], capture_traces)
                eval_cache[key] = result
                if result.get("rollout_metrics") and bucket_key in self._cost_tracker:
                    add_to_cost_bucket(self._cost_tracker[bucket_key], result["rollout_metrics"])
                if result.get("eval_metrics"):
                    add_to_cost_bucket(self._cost_tracker["eval_lm"], result["eval_metrics"])
            self._save_eval_cache(eval_cache)
            self._save_cost_summary()

        outputs = []
        scores = []
        trajectories = [] if capture_traces else None
        for result in results_list:
            if result.get("error_message"):
                self.logger.log(f"[iter {self._gepa_iter}, {self._phase}] {result['error_message']}")
            outputs.append(result["output"])
            scores.append(result["score"])
            if capture_traces:
                trajectories.append(result["trajectory"])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        self.logger.log(f"[iter {self._gepa_iter}, {self._phase}] scores: {scores}, avg: {avg_score:.3f}")

        if capture_traces:
            self.proposer.record_reflection_context(self._gepa_iter, batch, candidate, scores)
        elif self._phase == "candidate":
            self.proposer.record_candidate_outcome(
                self._gepa_iter,
                self._iter_dir(),
                batch,
                candidate,
                scores,
                outputs,
            )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

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
                trajectory = eval_batch.trajectories[i] if eval_batch.trajectories else {}

                if component == "agent_code":
                    trajectory_md = ""
                    if trajectory and "events" in trajectory:
                        try:
                            trajectory_md = convert_json_to_markdown(trajectory)
                        except Exception:
                            trajectory_md = "(trajectory conversion failed)"

                    feedback = (
                        format_eval_feedback(output, score)
                        if isinstance(output, dict)
                        else f"Score: {score}"
                    )

                    records.append(
                        {
                            "Task Input": example.get("prompt", ""),
                            "Agent Trajectory": trajectory_md,
                            "Evaluation Feedback": feedback,
                            "eval_output": output if isinstance(output, dict) else {},
                        }
                    )

            result[component] = records

        proposer_dir = os.path.join(self._iter_dir(), "proposer")
        os.makedirs(proposer_dir, exist_ok=True)
        reflection_path = os.path.join(proposer_dir, "reflection.json")
        with open(reflection_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        self.logger.log(f"[iter {self._gepa_iter}] Reflection inputs saved to {reflection_path}")

        return result

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        return self.proposer.propose(
            iteration=self._gepa_iter,
            iter_dir=self._iter_dir(),
            candidate=candidate,
            reflective_dataset=reflective_dataset,
            components_to_update=components_to_update,
        )
