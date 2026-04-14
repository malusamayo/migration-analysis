import argparse
import json
import logging
import os
from typing import Optional

import yaml
from gepa import GEPAResult, StopperProtocol, optimize
from gepa.utils.stop_condition import NoImprovementStopper, ScoreThresholdStopper

from .dataloader import prepare_task
from .optimize import (
    AgentOptimizationAdapter,
    CostBudgetStopper,
    GEPAFileLogger,
    execute_agent_candidate,
    extract_workspace_scripts,
    validate_agent_candidate,
)
from .task_setups import get_seed_candidate, preprocess_example, setup_servers

__all__ = [
    "AgentOptimizationAdapter",
    "CostBudgetStopper",
    "GEPAFileLogger",
    "build_parser",
    "execute_agent_candidate",
    "extract_workspace_scripts",
    "main",
    "run_optimization",
    "validate_agent_candidate",
]


def _print_effective_config(config: dict) -> None:
    print("Effective config:")
    print(yaml.safe_dump(config, sort_keys=False).rstrip())


def run_optimization(
    task_id: str,
    model_name: str,
    prompt_name: str = "default",
    max_examples: Optional[int] = None,
    train_ratio: float = 0.7,
    eval_lm_name: Optional[str] = None,
    reflection_lm: str = "gemini-3.1-pro-preview",
    max_metric_calls: Optional[int] = 50,
    max_cost: Optional[float] = None,
    no_improvement_patience: int = 10,
    seed: int = 0,
    run_dir: Optional[str] = None,
    data_path: Optional[str] = None,
    agent_batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    use_adaptation_guide: bool = True,
    adaptation_guide_markdown: Optional[str] = None,
    docker_network: Optional[str] = None,
    use_docker: bool = False,
    server_image: str = "migration-analysis:latest",
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data, task_prompt, _eval_prompt = prepare_task(
        task_id=task_id,
        model_name=model_name,
        rollout_version="gepa",
        prompt_name=prompt_name,
        max_examples=max_examples,
        data_path=data_path,
    )

    setup_servers(task_id, docker_network=docker_network)
    data = [preprocess_example(task_id, ex) for ex in data]

    split_idx = max(1, int(len(data) * train_ratio))
    trainset = data[:split_idx]
    valset = data[split_idx:] if split_idx < len(data) else data

    seed_candidate = get_seed_candidate(task_id)
    seed_code = seed_candidate["agent_code"]

    if run_dir is None:
        run_dir = f"results/{task_id}/{model_name}_{prompt_name}/gepa/seed{seed}"
    os.makedirs(run_dir, exist_ok=True)

    effective_config = {
        "task_id": task_id,
        "model_name": model_name,
        "prompt_name": prompt_name,
        "max_examples": max_examples,
        "train_ratio": train_ratio,
        "eval_lm": eval_lm_name,
        "reflection_lm": reflection_lm,
        "max_metric_calls": max_metric_calls,
        "max_cost": max_cost,
        "no_improvement_patience": no_improvement_patience,
        "seed": seed,
        "run_dir": run_dir,
        "data_path": data_path,
        "agent_batch_size": agent_batch_size,
        "eval_batch_size": eval_batch_size,
        "use_adaptation_guide": use_adaptation_guide,
        "adaptation_guide_markdown": adaptation_guide_markdown,
        "docker_network": docker_network,
        "use_docker": use_docker,
        "server_image": server_image,
    }
    _print_effective_config(effective_config)

    gepa_logger = GEPAFileLogger(run_dir)
    gepa_logger.log(f"Task: {task_id}, Model: {model_name}, Prompt: {prompt_name}")
    gepa_logger.log(f"Dataset: {len(data)} total, {len(trainset)} train, {len(valset)} val")
    gepa_logger.log(
        "Max metric calls: "
        f"{max_metric_calls}, Max cost: {max_cost}, "
        f"No-improvement patience: {no_improvement_patience}, Reflection LM: {reflection_lm}"
    )
    gepa_logger.log(
        f"Agent batch size: {agent_batch_size}, Eval batch size: {eval_batch_size}"
    )
    gepa_logger.log(f"Use adaptation guide: {use_adaptation_guide}")
    if use_adaptation_guide and adaptation_guide_markdown:
        gepa_logger.log(f"Adaptation guide markdown: {adaptation_guide_markdown}")

    config_dir = os.path.join(run_dir, "shared", "config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "used_config.yaml"), "w") as f:
        yaml.safe_dump(effective_config, f, sort_keys=False)
    with open(os.path.join(config_dir, "seed_config.py"), "w") as f:
        f.write(seed_code)

    adapter = AgentOptimizationAdapter(
        task_id=task_id,
        model_name=model_name,
        logger=gepa_logger,
        reflection_lm_name=reflection_lm,
        task_prompt=task_prompt,
        eval_lm_name=eval_lm_name,
        run_dir=run_dir,
        agent_batch_size=agent_batch_size,
        eval_batch_size=eval_batch_size,
        use_adaptation_guide=use_adaptation_guide,
        adaptation_guide_markdown=adaptation_guide_markdown,
        use_docker=use_docker,
        server_image=server_image,
        docker_network=docker_network,
    )

    stoppers: list[StopperProtocol] = []
    if max_cost is not None:
        stoppers.append(CostBudgetStopper(max_cost, adapter._cost_tracker))
    stoppers.append(NoImprovementStopper(no_improvement_patience))
    stoppers.append(ScoreThresholdStopper(1.0))

    result: GEPAResult = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        stop_callbacks=stoppers,
        seed=seed,
        run_dir=run_dir,
        display_progress_bar=True,
        logger=gepa_logger,
    )

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

    cost_summary = adapter._cost_tracker
    total_cost = sum(bucket["accumulated_cost"] for bucket in cost_summary.values())
    summary = {
        "best_score": best_score,
        "best_idx": result.best_idx,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "all_scores": result.val_aggregate_scores,
        "cost_summary": {**cost_summary, "total_cost": total_cost},
    }
    summary_path = os.path.join(config_dir, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Best score: {best_score}")
    print(f"Best config saved to: {best_config_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Log file: {gepa_logger.log_path}")
    print(f"Total cost: ${total_cost:.4f}")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agentic GEPA optimization")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--task_id", type=str, help="Task identifier")
    parser.add_argument("--model_name", type=str, help="Model name for agent execution")
    parser.add_argument("--prompt_name", type=str, default=None, help="Prompt name for seed")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--reflection_lm", type=str, default=None, help="LM for proposer agent")
    parser.add_argument("--eval_lm", type=str, default=None, help="LM for evaluation")
    parser.add_argument("--max_metric_calls", type=int, default=None)
    parser.add_argument("--max_cost", type=float, default=None, help="Hard cost budget in USD")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument(
        "--agent_batch_size",
        type=int,
        default=None,
        help="Maximum number of agent/example workers to run concurrently.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="GEPA reflection minibatch size used for candidate evaluation.",
    )
    parser.add_argument(
        "--adaptation_guide_markdown",
        type=str,
        default=None,
        help="Optional custom markdown file to use instead of docs/adaptation.md",
    )
    parser.add_argument(
        "--use_docker",
        action="store_true",
        default=None,
        help="Run agent rollouts inside Docker containers",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
    max_metric_calls = (
        args.max_metric_calls if args.max_metric_calls is not None else config.get("max_metric_calls")
    )
    max_cost = args.max_cost if args.max_cost is not None else config.get("max_cost")
    no_improvement_patience = config.get("no_improvement_patience")
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    run_dir = args.run_dir or config.get("run_dir")
    data_path = config.get("data_path")
    agent_batch_size = (
        args.agent_batch_size if args.agent_batch_size is not None else config.get("agent_batch_size")
    )
    eval_batch_size = (
        args.eval_batch_size if args.eval_batch_size is not None else config.get("eval_batch_size")
    )
    docker_network = config.get("docker_network")
    use_docker = args.use_docker if args.use_docker is not None else config.get("use_docker", False)
    server_image = config.get("server_image", "migration-analysis:latest")
    use_adaptation_guide = config.get("use_adaptation_guide", True)
    adaptation_guide_markdown = (
        args.adaptation_guide_markdown
        if args.adaptation_guide_markdown is not None
        else config.get("adaptation_guide_markdown")
    )

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
        max_cost=max_cost,
        no_improvement_patience=no_improvement_patience,
        seed=seed,
        run_dir=run_dir,
        data_path=data_path,
        agent_batch_size=agent_batch_size,
        eval_batch_size=eval_batch_size,
        docker_network=docker_network,
        use_docker=use_docker,
        server_image=server_image,
        use_adaptation_guide=use_adaptation_guide,
        adaptation_guide_markdown=adaptation_guide_markdown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
