import copy
import json
import os
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gepa import EvaluationBatch, GEPAAdapter
from gepa.core.state import GEPAState

from .cache import EvalCache
from .common import (
    add_to_cost_bucket,
    empty_cost_bucket,
    extract_dspy_cost,
    format_eval_feedback,
    hash_text,
)
from .memory import ProposerMemory
from .proposer import AgentProposer
from ..runner import run_single_instance_agentic
from ..task_setups import get_eval_config
from ..utils import LM_DICT, batch_inference


_PLAIN_SCALARS = (str, int, float, bool, type(None))


def _to_process_safe_data(value: Any) -> Any:
    """Convert worker payloads to plain data before multiprocessing pickles them."""
    if type(value) in _PLAIN_SCALARS:
        return value
    if isinstance(value, Enum):
        return _to_process_safe_data(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        safe_dict = {}
        for key, item in value.items():
            safe_key = _to_process_safe_data(key)
            if type(safe_key) not in _PLAIN_SCALARS:
                safe_key = str(safe_key)
            safe_dict[safe_key] = _to_process_safe_data(item)
        return safe_dict
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_process_safe_data(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_process_safe_data(model_dump(mode="json"))
        except TypeError:
            return _to_process_safe_data(model_dump())
        except Exception:
            pass

    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return str(value)


class CostBudgetStopper:
    """Stop optimization when total accumulated cost exceeds a budget in USD."""

    def __init__(self, max_cost: float, cost_tracker: dict[str, dict]):
        self.max_cost = max_cost
        self._cost_tracker = cost_tracker

    def __call__(self, gepa_state: GEPAState) -> bool:
        total = sum(bucket["accumulated_cost"] for bucket in self._cost_tracker.values())
        return total >= self.max_cost


def _run_agent_worker(
    i: int,
    example: dict,
    candidate: dict,
    phase_dir: str,
    task_id: str,
    model_name: str,
    task_prompt: str,
    eval_lm_name: Optional[str],
    use_docker: bool = False,
    server_image: Optional[str] = None,
    docker_network: Optional[str] = None,
    max_time: Optional[float] = None,
) -> dict:
    """Run the candidate agent for one example and persist workspace artifacts."""
    example = copy.deepcopy(example)
    workspace_base = os.path.abspath(os.path.join(phase_dir, f"example{i}"))
    os.makedirs(workspace_base, exist_ok=True)

    config_dir = Path(workspace_base).parent / f"{Path(workspace_base).name}_config"
    os.makedirs(config_dir, exist_ok=True)
    agent_file = os.path.join(config_dir, f"example{i}_agent.py")
    prompt_file = os.path.join(config_dir, f"example{i}_prompt.md")
    Path(agent_file).write_text(candidate["agent_code"])
    Path(prompt_file).write_text(task_prompt)

    log_dir = Path(workspace_base).parent / f"{Path(workspace_base).name}_logs"
    os.makedirs(log_dir, exist_ok=True)

    try:
        result = run_single_instance_agentic(
            lm=LM_DICT[model_name],
            system_prompt_path=prompt_file,
            example=example,
            workspace=workspace_base,
            task_id=task_id,
            use_docker=use_docker,
            server_image=server_image,
            docker_network=docker_network,
            agent_file=agent_file,
            max_time=max_time,
        )

        rollout_metrics = None
        if result and "run_result" in result:
            rollout_metrics = result["run_result"].get("metrics")
        eval_example = _to_process_safe_data(result or example)

        return {
            "workspace_dir": workspace_base,
            "eval_example": eval_example,
            "config_dir": str(config_dir),
            "log_dir": str(log_dir),
            "rollout_metrics": _to_process_safe_data(rollout_metrics),
            "error_message": None,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "workspace_dir": workspace_base,
            "eval_example": _to_process_safe_data(example),
            "config_dir": str(config_dir),
            "log_dir": "",
            "rollout_metrics": None,
            "error_message": f"Error on example {i}: {e}",
        }


def _run_eval_worker(
    i: int,
    eval_example: dict,
    workspace_dir: str,
    log_dir: str,
    task_id: str,
    eval_lm_name: Optional[str],
) -> dict:
    """Run task evaluation for one example using an existing workspace."""

    eval_function = get_eval_config(task_id)["eval_function"]
    eval_lm = LM_DICT[eval_lm_name] if eval_lm_name else None

    try:
        eval_kwargs = {
            "workspace_dir": workspace_dir,
            "example": eval_example,
            "lm": eval_lm,
        }
        eval_history_before = (
            len(eval_lm.history) if eval_lm is not None and hasattr(eval_lm, "history") else 0
        )
        eval_result = eval_function(**eval_kwargs)
        eval_metrics = extract_dspy_cost(eval_lm, eval_history_before)
        score = float(eval_result.get("score", 0.0))

        trajectory = EvalCache.load_trace_payload(log_dir, eval_result)

        return EvalCache.attach_artifact_metadata({
            "output": eval_result,
            "score": score,
            "trajectory": trajectory,
            "eval_metrics": eval_metrics,
            "error_message": None,
        }, workspace_dir, log_dir, str(Path(workspace_dir).parent / f"{Path(workspace_dir).name}_config"))
    except Exception as e:
        traceback.print_exc()
        trajectory = EvalCache.load_trace_payload(log_dir, None) or {"error": str(e)}
        return EvalCache.attach_artifact_metadata({
            "output": {"error": str(e), "score": 0.0},
            "score": 0.0,
            "trajectory": trajectory,
            "eval_metrics": None,
            "error_message": f"Error on example {i}: {e}",
        }, workspace_dir, log_dir, str(Path(workspace_dir).parent / f"{Path(workspace_dir).name}_config"))


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
        num_exploration: int = 1,
        agent_batch_size: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        use_adaptation_guide: bool = True,
        adaptation_guide_markdown: Optional[str] = None,
        use_docker: bool = False,
        server_image: str = "migration-analysis:latest",
        docker_network: Optional[str] = None,
        max_time: Optional[float] = None,
    ):
        self.task_id = task_id
        self.task_prompt = task_prompt
        self.model_name = model_name
        self.logger = logger
        self.reflection_lm_name = reflection_lm_name
        self.run_dir = run_dir
        self.eval_lm_name = eval_lm_name
        self.eval_lm = LM_DICT[eval_lm_name] if eval_lm_name else None
        self._agent_batch_size = agent_batch_size
        self.use_docker = use_docker
        self.server_image = server_image
        self.docker_network = docker_network
        self.max_time = max_time
        self.num_exploration = num_exploration

        self.eval_config = get_eval_config(task_id)
        self.eval_function = self.eval_config["eval_function"]
        self._eval_batch_size = eval_batch_size

        self._gepa_iter = 0
        self._phase = "seed"
        self._calls_since_reflection = 0
        self._last_batch: list[dict] = []
        self._last_results: list[dict] = []

        self._cost_tracker: dict[str, dict] = {
            "rollout_seed": empty_cost_bucket(),
            "rollout_reflection": empty_cost_bucket(),
            "rollout_candidate": empty_cost_bucket(),
            "rollout_valset": empty_cost_bucket(),
            "eval_lm": empty_cost_bucket(),
            "proposer": empty_cost_bucket(),
        }

        self.memory = ProposerMemory(run_dir)
        self.proposer = AgentProposer(
            run_dir=run_dir,
            task_prompt=task_prompt,
            model_name=self.model_name,
            reflection_lm_name=reflection_lm_name,
            logger=logger,
            cost_tracker=self._cost_tracker,
            save_cost_summary=self._save_cost_summary,
            memory=self.memory,
            max_proposal_retries=max_proposal_retries,
            use_adaptation_guide=use_adaptation_guide,
            adaptation_guide_markdown=adaptation_guide_markdown,
            task_id=task_id,
            use_docker=use_docker,
            server_image=server_image,
            docker_network=docker_network,
        )

        os.makedirs(run_dir, exist_ok=True)
        self._cache = EvalCache(os.path.join(run_dir, "shared", "eval_cache.json"))

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
            f"(rollout_seed=${self._cost_tracker['rollout_seed']['accumulated_cost']:.4f}, "
            f"rollout_reflection=${self._cost_tracker['rollout_reflection']['accumulated_cost']:.4f}, "
            f"rollout_candidate=${self._cost_tracker['rollout_candidate']['accumulated_cost']:.4f}, "
            f"rollout_valset=${self._cost_tracker['rollout_valset']['accumulated_cost']:.4f}, "
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

    def _run_candidate_on_batch(
        self,
        candidate: dict[str, str],
        batch: list[dict],
        phase_dir: str,
        rollout_bucket_key: str,
        log_tag: str,
    ) -> list[dict]:
        """Run a candidate on a batch of examples with EvalCache reuse.

        Returns per-example result dicts (with "score", "output", "trajectory",
        "error_message", artifact paths, ...). Mutates cost_tracker and saves the
        cache on completion. Does NOT touch phase/memory/last_batch state.
        """
        assert self._agent_batch_size is not None, "agent_batch_size must be set to run evaluation"
        agent_max_workers = self._agent_batch_size

        args_list = [
            {
                "i": i,
                "example": ex,
                "candidate": candidate,
                "phase_dir": phase_dir,
                "task_id": self.task_id,
                "model_name": self.model_name,
                "task_prompt": self.task_prompt,
                "eval_lm_name": self.eval_lm_name,
                "use_docker": self.use_docker,
                "server_image": self.server_image,
                "docker_network": self.docker_network,
                "max_time": self.max_time,
            }
            for i, ex in enumerate(batch)
        ]

        results_list: list[Optional[dict]] = [None] * len(args_list)
        uncached_indices: list[int] = []
        n_hits = 0
        cache_updated = False
        for idx, args in enumerate(args_list):
            prepared_result, updated = self._cache.get_prepared(
                candidate["agent_code"],
                args["example"],
                os.path.abspath(os.path.join(phase_dir, f"example{args['i']}")),
            )
            if prepared_result is not None:
                results_list[idx] = prepared_result
                n_hits += 1
                cache_updated = cache_updated or updated
            else:
                uncached_indices.append(idx)

        if n_hits:
            self.logger.log(f"{log_tag} Cache hits: {n_hits}/{len(batch)}")

        if uncached_indices:
            uncached_args = [args_list[idx] for idx in uncached_indices]
            rollout_results = batch_inference(
                _run_agent_worker,
                uncached_args,
                max_workers=agent_max_workers,
                use_process=True,
            )

            eval_indices = []
            eval_args = []
            for idx, rollout_result in zip(uncached_indices, rollout_results):
                if rollout_result.get("rollout_metrics") and rollout_bucket_key in self._cost_tracker:
                    add_to_cost_bucket(self._cost_tracker[rollout_bucket_key], rollout_result["rollout_metrics"])

                if rollout_result.get("error_message"):
                    result = EvalCache.attach_artifact_metadata({
                        "output": {"error": rollout_result["error_message"], "score": 0.0},
                        "score": 0.0,
                        "trajectory": {"error": rollout_result["error_message"]},
                        "rollout_metrics": rollout_result.get("rollout_metrics"),
                        "eval_metrics": None,
                        "error_message": rollout_result["error_message"],
                    }, rollout_result["workspace_dir"], rollout_result["log_dir"], rollout_result["config_dir"])
                    results_list[idx] = result
                    self._cache.put(candidate["agent_code"], args_list[idx]["example"], result)
                else:
                    eval_indices.append(idx)
                    eval_args.append(
                        {
                            "i": args_list[idx]["i"],
                            "eval_example": rollout_result["eval_example"],
                            "workspace_dir": rollout_result["workspace_dir"],
                            "log_dir": rollout_result["log_dir"],
                            "task_id": self.task_id,
                            "eval_lm_name": self.eval_lm_name,
                        }
                    )

            eval_max_workers = (
                self._eval_batch_size
                if self._eval_batch_size is not None
                else self.eval_config["max_workers"]
            )
            fresh_results = batch_inference(
                _run_eval_worker,
                eval_args,
                max_workers=eval_max_workers,
                use_process=self.eval_config["use_process"],
            ) if eval_args else []

            for idx, result in zip(eval_indices, fresh_results):
                results_list[idx] = result
                self._cache.put(candidate["agent_code"], args_list[idx]["example"], result)
                if result.get("eval_metrics"):
                    add_to_cost_bucket(self._cost_tracker["eval_lm"], result["eval_metrics"])
            self._cache.save()
            self._save_cost_summary()
        elif cache_updated:
            self._cache.save()

        for result in results_list:
            if result.get("error_message"):
                self.logger.log(f"{log_tag} {result['error_message']}")

        return results_list

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        is_reflection_pass = capture_traces

        if is_reflection_pass:
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

        log_tag = f"[iter {self._gepa_iter}, {self._phase}]"
        self.logger.log(f"{log_tag} Evaluating {len(batch)} examples")

        results_list = self._run_candidate_on_batch(
            candidate=candidate,
            batch=batch,
            phase_dir=phase_dir,
            rollout_bucket_key=f"rollout_{self._phase}",
            log_tag=log_tag,
        )

        self._last_results = list(results_list)

        outputs = []
        scores = []
        trajectories = [] if is_reflection_pass else None
        for result in results_list:
            outputs.append(result["output"])
            scores.append(result["score"])
            if is_reflection_pass:
                trajectories.append(result["trajectory"])

        avg_score = sum(scores) / len(scores) if scores else 0.0
        self.logger.log(f"{log_tag} scores: {scores}, avg: {avg_score:.3f}")

        if self._phase == "seed":
            self.memory.record_seed(
                candidate["agent_code"],
                list(scores),
                outputs,
                batch,
                trajectory_dir=str(phase_dir),
            )
            self.memory.write_iteration_overview(self._iter_dir(), 0)
        elif is_reflection_pass:
            self.memory.record_reflection(
                self._gepa_iter,
                candidate["agent_code"],
                list(scores),
                [ex.get("prompt", "")[:300] for ex in batch],
                str(self._phase_dir()),
            )
        elif self._phase == "candidate":
            self.memory.record_outcome(
                self._gepa_iter,
                candidate["agent_code"],
                list(scores),
                outputs,
                batch,
                trajectory_dir=str(phase_dir),
            )
            self.memory.write_iteration_overview(self._iter_dir(), self._gepa_iter)
        elif self._phase == "valset":
            avg = sum(scores) / len(scores) if scores else 0.0
            self.memory.record_val_score(
                hash_text(candidate["agent_code"]),
                avg,
                list(scores),
                outputs,
                batch,
                trajectory_dir=str(phase_dir),
            )
            self.memory.write_iteration_overview(self._iter_dir(), self._gepa_iter)

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
                if component == "agent_code":
                    result = self._last_results[i] if i < len(self._last_results) else {}
                    trajectory_json_path = result.get("raw_trace_json_path") or result.get("trace_json_path")
                    feedback = (
                        format_eval_feedback(output, score)
                        if isinstance(output, dict)
                        else f"Score: {score}"
                    )

                    records.append(
                        {
                            "Task Input": example.get("prompt", ""),
                            "Agent Trajectory JSON Path": trajectory_json_path,
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

        self.memory.record_reflective_dataset(result.get("agent_code", []))

        return result

    def _select_best_of_n(
        self,
        proposals: list[dict[str, str]],
        batch: list[dict],
    ) -> dict[str, str]:
        """Score each proposal on `batch`, record outcomes, return the winner.

        Writes per-proposal rollouts to `iter_dir/proposals/prop_kk/` and saves
        the proposed code to `iter_dir/candidates/prop_kk.py`. Charges all
        rollouts to `rollout_candidate`.
        """
        proposals_root = os.path.join(self._iter_dir(), "proposals")
        os.makedirs(proposals_root, exist_ok=True)
        candidates_dir = os.path.join(self._iter_dir(), "candidates")
        os.makedirs(candidates_dir, exist_ok=True)

        log_tag = f"[iter {self._gepa_iter}, proposal_selection]"
        self.logger.log(
            f"{log_tag} Scoring {len(proposals)} proposals on reflection minibatch "
            f"({len(batch)} examples)"
        )

        avg_scores: list[float] = []
        for k, prop in enumerate(proposals):
            prop_dir = os.path.join(proposals_root, f"prop_{k:02d}")
            os.makedirs(prop_dir, exist_ok=True)
            with open(os.path.join(candidates_dir, f"prop_{k:02d}.py"), "w") as f:
                f.write(prop["agent_code"])

            prop_tag = f"[iter {self._gepa_iter}, prop_{k:02d}]"
            results = self._run_candidate_on_batch(
                candidate=prop,
                batch=batch,
                phase_dir=prop_dir,
                rollout_bucket_key="rollout_candidate",
                log_tag=prop_tag,
            )

            outputs = [r["output"] for r in results]
            scores = [r["score"] for r in results]
            avg = sum(scores) / len(scores) if scores else 0.0
            avg_scores.append(avg)
            self.logger.log(f"{prop_tag} scores: {scores}, avg: {avg:.3f}")

            self.memory.record_outcome(
                self._gepa_iter,
                prop["agent_code"],
                list(scores),
                outputs,
                batch,
                trajectory_dir=str(prop_dir),
            )

        best_idx = max(range(len(avg_scores)), key=lambda i: avg_scores[i])
        self.logger.log(
            f"{log_tag} avg scores: {[f'{s:.3f}' for s in avg_scores]}, "
            f"picked prop_{best_idx:02d} (avg={avg_scores[best_idx]:.3f})"
        )
        self.memory.write_iteration_overview(self._iter_dir(), self._gepa_iter)
        return proposals[best_idx]

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        all_proposals = self.proposer.propose(
            iteration=self._gepa_iter,
            iter_dir=self._iter_dir(),
            candidate=candidate,
            components_to_update=components_to_update,
            num_strategies=self.num_exploration,
        )
        if len(all_proposals) == 1:
            return all_proposals[0]
        return self._select_best_of_n(all_proposals, self._last_batch)
