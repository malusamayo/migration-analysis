"""
UCB-based skill subset optimization for agent performance improvement.

This module implements an automated skill subset exploration system using the
Upper Confidence Bound (UCB1) bandit algorithm to efficiently discover
high-performing skill combinations.
"""

import argparse
import json
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
import numpy as np

from .dataloader import generate_rollout_version


@dataclass
class SkillArm:
    """Represents a single skill as a bandit arm with pull and reward statistics."""

    skill_name: str
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def average_reward(self) -> float:
        """Calculate average reward per pull."""
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0

    def update(self, reward: float):
        """Update statistics with a new reward."""
        self.pulls += 1
        self.total_reward += reward

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for serialization (native Python types)."""
        return {
            'pulls': int(self.pulls),
            'total_reward': float(self.total_reward),
            'avg_reward': float(self.average_reward)
        }


class UCBSkillSelector:
    """Implements UCB1 algorithm for skill subset selection."""

    def __init__(
        self,
        skill_names: List[str],
        duplicate_counts: Optional[Dict[str, int]] = None,
        exploration_param: float = 1.0,
        duplicate_bonus_weight: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize UCB skill selector.

        Args:
            skill_names: List of all available skill names
            duplicate_counts: Optional dict mapping skill names to duplicate counts (for prior)
            exploration_param: Exploration parameter c in UCB formula (default: 1.0)
            duplicate_bonus_weight: Weight for duplicate_count bonus (default: 0.1)
            seed: Random seed for reproducibility
        """
        self.exploration_param = exploration_param
        self.duplicate_bonus_weight = duplicate_bonus_weight
        self.duplicate_counts = duplicate_counts or {}
        self.arms: Dict[str, SkillArm] = {
            name: SkillArm(skill_name=name) for name in skill_names
        }
        self.total_pulls = 0
        self.rng = np.random.RandomState(seed)

    def compute_ucb_scores(self) -> Dict[str, float]:
        """
        Compute UCB score for each skill.

        UCB Formula: UCB(skill) = avg_reward + c * sqrt(ln(N) / n) + duplicate_bonus

        The duplicate_bonus gives skills with higher duplicate_count a prior advantage,
        since they appeared more frequently during skill generation.

        Returns:
            Dictionary mapping skill names to UCB scores
        """
        if self.total_pulls == 0:
            # First round: use duplicate_count as prior
            ucb_scores = {}
            for name in self.arms.keys():
                duplicate_count = self.duplicate_counts.get(name, 1)
                # Use log to dampen the effect of very high counts
                duplicate_bonus = np.log(duplicate_count) * self.duplicate_bonus_weight
                ucb_scores[name] = duplicate_bonus
            return ucb_scores

        ucb_scores = {}
        for name, arm in self.arms.items():
            # Get duplicate_count bonus
            duplicate_count = self.duplicate_counts.get(name, 1)
            duplicate_bonus = np.log(duplicate_count) * self.duplicate_bonus_weight

            if arm.pulls == 0:
                # Unpulled arms get infinite score to ensure exploration
                # But still add duplicate bonus for tie-breaking
                ucb_scores[name] = float('inf')
            else:
                exploitation = arm.average_reward
                exploration = self.exploration_param * np.sqrt(
                    np.log(self.total_pulls) / arm.pulls
                )
                ucb_scores[name] = exploitation + exploration + duplicate_bonus

        print(self.total_pulls, ucb_scores)
        exit(0)

        return ucb_scores

    def select_subset(self, k: int) -> List[str]:
        """
        Select top-k skills by UCB score with random tie-breaking.

        Args:
            k: Number of skills to select

        Returns:
            List of selected skill names
        """
        if k > len(self.arms):
            raise ValueError(f"Cannot select {k} skills from {len(self.arms)} available")

        ucb_scores = self.compute_ucb_scores()
        skill_names = list(ucb_scores.keys())

        # Sort by (UCB score desc, random value) for tie-breaking
        sorted_skills = sorted(
            skill_names,
            key=lambda name: (ucb_scores[name], self.rng.random()),
            reverse=True
        )

        return sorted_skills[:k]

    def update_arms(self, selected_skills: List[str], reward: float):
        """
        Update all selected skills with the same reward.

        Args:
            selected_skills: List of skill names that were selected
            reward: Reward value (typically performance delta)
        """
        for skill_name in selected_skills:
            if skill_name not in self.arms:
                raise ValueError(f"Unknown skill: {skill_name}")
            self.arms[skill_name].update(reward)

        self.total_pulls += 1

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Export current statistics for all skills.

        Returns:
            Dictionary mapping skill names to their statistics (native Python types)
        """
        ucb_scores = self.compute_ucb_scores()

        stats = {}
        for name, arm in self.arms.items():
            ucb_score = ucb_scores[name]
            stats[name] = {
                'pulls': int(arm.pulls),  # Ensure native int
                'total_reward': float(arm.total_reward),  # Ensure native float
                'avg_reward': float(arm.average_reward),  # Ensure native float
                'ucb_score': float(ucb_score) if ucb_score != float('inf') else None
            }

        return stats


@dataclass
class OptimizationState:
    """Manages state persistence for optimization with YAML serialization."""

    baseline_score: float
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    skill_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_subset: Optional[Dict[str, Any]] = None

    def _to_native_types(self, obj):
        """Convert all values to native Python types for clean YAML serialization."""
        if isinstance(obj, dict):
            return {k: self._to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_native_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif obj is None or obj == float('inf') or obj == float('-inf'):
            return None  # Convert inf to None for YAML compatibility
        else:
            return obj

    def save(self, path: Path):
        """Save state to YAML file with all native Python types."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and ensure all values are native Python types
        data = asdict(self)
        data = self._to_native_types(data)

        with open(path, 'w') as f:
            yaml.dump(data, f, indent=2, allow_unicode=True, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> 'OptimizationState':
        """Load state from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)


class SkillOptimizer:
    """Main orchestrator for UCB-based skill subset optimization."""

    def __init__(
        self,
        model_name: str,
        prompt_name: str,
        skill_dir: Path,
        task_id: str,
        baseline_path: Path,
        output_path: Optional[Path] = None,
        max_rounds: int = 20,
        subset_sizes: List[int] = None,
        exploration_param: float = 1.0,
        duplicate_bonus_weight: float = 0.1,
        eval_lm_name: Optional[str] = None,
        is_agentic: bool = True,
        max_examples: Optional[int] = None,
        seed: Optional[int] = None,
        resume: bool = True
    ):
        """
        Initialize skill optimizer.

        Args:
            model_name: Model name (e.g., "qwen3-coder-30b-a3b")
            prompt_name: Prompt name (e.g., "static")
            skill_dir: Source skill directory path
            task_id: Task identifier (e.g., "webtest")
            baseline_path: Path to baseline eval_results.yaml (v0 performance)
            output_path: Where to save optimization results (auto-generated if None)
            max_rounds: Maximum number of optimization rounds
            subset_sizes: List of subset sizes to explore (default: [3, 5, 10])
            exploration_param: UCB exploration parameter (default: 1.0)
            duplicate_bonus_weight: Weight for duplicate_count bonus (default: 0.1)
            eval_lm_name: LM name for evaluation (required for webtest)
            is_agentic: Whether to use agentic execution
            max_examples: Maximum examples to process per round (None = all)
            seed: Random seed for reproducibility
            resume: Whether to resume from existing checkpoint
        """
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.skill_dir = Path(skill_dir)
        self.task_id = task_id
        self.baseline_path = Path(baseline_path)
        self.max_rounds = max_rounds
        self.subset_sizes = subset_sizes or [3, 5, 10]
        self.exploration_param = exploration_param
        self.duplicate_bonus_weight = duplicate_bonus_weight
        self.eval_lm_name = eval_lm_name
        self.is_agentic = is_agentic
        self.max_examples = max_examples
        self.seed = seed
        self.resume = resume

        # Auto-generate output path if not provided
        if output_path is None:
            results_base = Path(f"results/{task_id}/{model_name}_{prompt_name}")
            self.output_path = results_base / "optimize_results.yaml"
        else:
            self.output_path = Path(output_path)
            

        # Error log path
        self.error_log_path = self.output_path.parent / "optimization_errors.jsonl"

        # Load skills and initialize components
        print(f"🔧 Initializing optimizer...")
        print(f"  Model: {model_name}")
        print(f"  Task: {task_id}")
        print(f"  Skill dir: {skill_dir}")
        print(f"  Baseline: {baseline_path}")

        self.skill_manager = self._load_skill_manager()
        self.baseline_score = self._load_baseline_score()

        # Initialize or load state
        if resume and self.output_path.exists():
            print(f"📂 Resuming from checkpoint: {self.output_path}")
            self.state = OptimizationState.load(self.output_path)
            print(f"  Completed rounds: {len(self.state.rounds)}")
        else:
            print(f"🆕 Starting fresh optimization")
            self.state = OptimizationState(baseline_score=self.baseline_score)

        # Initialize UCB selector with duplicate_count as prior
        skill_names = [skill.skill_name for skill in self.skill_manager.skills]
        duplicate_counts = {
            skill.skill_name: skill.duplicate_count or 1
            for skill in self.skill_manager.skills
        }
        self.selector = UCBSkillSelector(
            skill_names=skill_names,
            duplicate_counts=duplicate_counts,
            exploration_param=exploration_param,
            duplicate_bonus_weight=duplicate_bonus_weight,
            seed=seed
        )

        # Restore UCB state if resuming
        if resume and self.state.skill_statistics:
            self._restore_ucb_state()

        print(f"✅ Initialization complete")
        print(f"  Skills available: {len(skill_names)}")
        print(f"  Baseline score: {self.baseline_score:.3f}")
        print(f"  Subset sizes: {self.subset_sizes}")
        print(f"  Max rounds: {max_rounds}")
        print()

    def _load_skill_manager(self):
        """Load skills from the skill directory."""
        from src.review.skill_manager import SkillManager

        manager = SkillManager()
        manager.load_skills(self.skill_dir / "metadata.yaml")
        return manager

    def _load_baseline_score(self) -> float:
        """Load baseline score from eval_results.yaml."""
        from src.analysis_utils import load_scores_detailed

        print(f"📊 Loading baseline from: {self.baseline_path}")
        result = load_scores_detailed(self.baseline_path)
        return result['avg_score']

    def _restore_ucb_state(self):
        """Restore UCB selector state from saved statistics."""
        for skill_name, stats in self.state.skill_statistics.items():
            if skill_name in self.selector.arms:
                arm = self.selector.arms[skill_name]
                arm.pulls = int(stats['pulls'])
                arm.total_reward = float(stats['total_reward'])

        # Restore total pulls
        if self.selector.arms:
            self.selector.total_pulls = max(
                (arm.pulls for arm in self.selector.arms.values()),
                default=0
            )

        print(f"  Restored UCB state: {self.selector.total_pulls} total pulls")

    def _get_next_round_config(self) -> Tuple[int, int]:
        """
        Get configuration for the next round.

        Returns:
            Tuple of (round_num, subset_size)
        """
        current_round = len(self.state.rounds)
        subset_size = self.subset_sizes[current_round % len(self.subset_sizes)]
        return current_round, subset_size

    def _create_temp_metadata(
        self,
        selected_skills: List[str],
        rollout_version: str
    ) -> Path:
        """
        Create temporary metadata file listing only selected skills.

        Args:
            selected_skills: List of skill names to include
            rollout_version: Rollout version name for this subset

        Returns:
            Path to temporary metadata file
        """
        # Create temp metadata file in the skill directory
        temp_metadata_path = self.skill_dir / f"metadata_{rollout_version}.yaml"

        # Load original metadata to get skill paths
        original_metadata_path = self.skill_dir / "metadata.yaml"
        with open(original_metadata_path, 'r') as f:
            original_metadata = yaml.safe_load(f)

        # Filter to only selected skills
        selected_skills_set = set(selected_skills)
        filtered_skills = [
            skill_meta for skill_meta in original_metadata.get('skills', [])
            if skill_meta.get('name') in selected_skills_set
        ]

        # Create temp metadata with only selected skills
        temp_metadata = {
            'skills': filtered_skills,
            'description': f'Optimization subset for {rollout_version}',
            'num_skills': len(filtered_skills)
        }

        # Write temp metadata file
        with open(temp_metadata_path, 'w') as f:
            yaml.dump(temp_metadata, f, indent=2, allow_unicode=True, sort_keys=False)

        return temp_metadata_path

    def _run_collect_evaluate_cycle(
        self,
        selected_skills: List[str],
        round_num: int,
        subset_size: int
    ) -> Dict[str, Any]:
        """
        Execute one optimization round: collect + evaluate.

        Args:
            selected_skills: Skills to include in this round
            round_num: Current round number
            subset_size: Subset size for this round

        Returns:
            Dictionary with round results
        """
        from src.collect import run_task
        from src.evaluate import run_task_eval
        from src.analysis_utils import load_scores_detailed

        # Generate rollout version name
        rollout_version = f"opt_r{round_num}_k{subset_size}_s{self.seed or 0}"

        print(f"🔄 Round {round_num}: k={subset_size}, version={rollout_version}")
        print(f"  Selected skills: {selected_skills}")

        # Create temp metadata file
        temp_metadata_path = self._create_temp_metadata(selected_skills, rollout_version)

        rollout_version = generate_rollout_version(
            skill_version=str(temp_metadata_path),
            skill_mode="all_loaded",
            subset_mode="all",
        )

        try:
            # Run collection
            print(f"  ⚙️  Running collection...")
            run_task(
                task_id=self.task_id,
                model_name=self.model_name,
                prompt_name=self.prompt_name,
                is_agentic=self.is_agentic,
                skill_path=str(temp_metadata_path),
                skill_mode="all_loaded",
                subset_mode="all",
                max_examples=self.max_examples,
                # resume=False
            )

            # Run evaluation
            print(f"  📊 Running evaluation...")
            run_task_eval(
                task_id=self.task_id,
                model_name=self.model_name,
                eval_lm_name=self.eval_lm_name,
                prompt_name=self.prompt_name,
                rollout_version=rollout_version,
                # resume=False
            )

            # Load and analyze results
            eval_path = Path(
                f"results/{self.task_id}/{self.model_name}_{self.prompt_name}/"
                f"rollouts/{rollout_version}/eval_results.yaml"
            )

            if not eval_path.exists():
                raise FileNotFoundError(f"Evaluation results not found: {eval_path}")

            result = load_scores_detailed(eval_path)
            score = result['avg_score']
            delta = score - self.baseline_score

            print(f"  ✅ Score: {score:.3f} (delta: {delta:+.3f})")

            return {
                'round': round_num,
                'subset_size': subset_size,
                'subset': selected_skills,
                'score': score,
                'delta': delta,
                'rollout_version': rollout_version,
                'metrics': {
                    'avg_cost': result.get('avg_cost'),
                    'total_cost': result.get('total_cost'),
                    'avg_tokens': result.get('avg_total_tokens'),
                    'avg_steps': result.get('avg_agent_steps'),
                    'num_examples': result['num_examples']
                }
            }

        finally:
            # Cleanup temp metadata file (keep rollout results)
            if temp_metadata_path.exists():
                temp_metadata_path.unlink()
                print(f"  🧹 Cleaned up temp metadata file")

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """
        Validate result before using.

        Args:
            result: Result dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        if result.get('score', -1) < 0 or result.get('score', 2) > 1:
            print(f"⚠️  Invalid score: {result.get('score')}")
            return False

        required_fields = ['delta', 'subset', 'score']
        for field in required_fields:
            if field not in result:
                print(f"⚠️  Missing required field: {field}")
                return False

        return True

    def _log_error(self, round_num: int, selected_skills: List[str], error: str):
        """Log error to JSONL file."""
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.error_log_path, 'a') as f:
            json.dump({
                'round': round_num,
                'subset': selected_skills,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }, f)
            f.write('\n')

    def _generate_final_report(self):
        """Generate and print final optimization report."""
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Baseline score: {self.baseline_score:.3f}")
        print(f"Total rounds: {len(self.state.rounds)}")

        if self.state.best_subset:
            best = self.state.best_subset
            print(f"\n🏆 Best subset found:")
            print(f"  Round: {best['round']}")
            print(f"  Score: {best['score']:.3f} (delta: {best['delta']:+.3f})")
            print(f"  Subset size: {best['subset_size']}")
            print(f"  Skills: {best['skills']}")

        # Top skills by UCB
        stats = self.state.skill_statistics
        sorted_skills = sorted(
            stats.items(),
            key=lambda x: x[1].get('avg_reward', 0),
            reverse=True
        )

        print(f"\n📊 Top 10 skills by average reward:")
        for i, (skill, stat) in enumerate(sorted_skills[:10], 1):
            print(f"  {i}. {skill}: "
                  f"avg_reward={stat.get('avg_reward', 0):.3f}, "
                  f"pulls={stat.get('pulls', 0)}")

        print(f"\n💾 Results saved to: {self.output_path}")
        if self.error_log_path.exists():
            print(f"⚠️  Errors logged to: {self.error_log_path}")
        print("=" * 60)

    def optimize(self):
        """Main optimization loop."""
        print(f"🚀 Starting optimization...")
        print(f"  Rounds to complete: {self.max_rounds - len(self.state.rounds)}")
        print()

        while len(self.state.rounds) < self.max_rounds:
            round_num, subset_size = self._get_next_round_config()

            try:
                # Select skills using UCB
                selected_skills = self.selector.select_subset(subset_size)

                # Run collect + evaluate cycle
                result = self._run_collect_evaluate_cycle(
                    selected_skills, round_num, subset_size
                )

                # Validate result
                if not self._validate_result(result):
                    print(f"⚠️  Skipping invalid result for round {round_num}")
                    continue

                # Update bandit statistics
                reward = result['delta']
                self.selector.update_arms(selected_skills, reward)

                # Update state
                self.state.rounds.append(result)
                self.state.skill_statistics = self.selector.get_statistics()

                # Update best subset
                if (self.state.best_subset is None or
                    result['delta'] > self.state.best_subset['delta']):
                    self.state.best_subset = {
                        'skills': result['subset'],
                        'score': result['score'],
                        'delta': result['delta'],
                        'subset_size': result['subset_size'],
                        'round': result['round']
                    }
                    print(f"  🌟 New best subset found!")

                # Save checkpoint
                self.state.save(self.output_path)
                print(f"  💾 Checkpoint saved")
                print()

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                self.state.save(self.output_path)
                print(f"💾 State saved. Resume with --resume flag.")
                raise

            except Exception as e:
                print(f"❌ Error in round {round_num}: {e}")
                traceback.print_exc()

                # Log error
                self._log_error(round_num, selected_skills, str(e))

                # Continue to next round (don't update UCB stats)
                print(f"  ⏭️  Continuing to next round...")
                print()
                continue

        # Generate final report
        self._generate_final_report()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UCB-based skill subset optimization for agent performance"
    )

    # Config file support
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file with default arguments"
    )

    # Required arguments
    parser.add_argument(
        "--model_name", type=str,
        help="Model name (e.g., qwen3-coder-30b-a3b)"
    )
    parser.add_argument(
        "--prompt_name", type=str, 
        help="Prompt name"
    )
    parser.add_argument(
        "--task_id", type=str,
        help="Task identifier (e.g., webtest)"
    )
    parser.add_argument(
        "--skill_dir", type=str,
        help="Path to source skill directory"
    )
    parser.add_argument(
        "--baseline_path", type=str,
        help="Path to baseline eval_results.yaml (v0 performance)"
    )

    # Optional arguments (all default to None; config or code defaults apply)
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output path for results (auto-generated if not specified)"
    )
    parser.add_argument(
        "--max_rounds", type=int, default=None,
        help="Maximum number of optimization rounds (default: 20)"
    )
    parser.add_argument(
        "--subset_sizes", type=int, nargs="+", default=None,
        help="Subset sizes to explore (default: [3, 5, 10])"
    )
    parser.add_argument(
        "--exploration_param", type=float, default=None,
        help="UCB exploration parameter (default: 1.0)"
    )
    parser.add_argument(
        "--duplicate_bonus_weight", type=float, default=None,
        help="Weight for duplicate_count bonus (default: 0.1)"
    )
    parser.add_argument(
        "--eval_lm", type=str, default=None,
        help="LM name for evaluation (required for webtest)"
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Maximum examples per round (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--is_agentic", type=lambda x: str(x).lower() == 'true', default=None,
        help="Whether to use agentic execution (default: true)"
    )
    parser.add_argument(
        "--resume", type=lambda x: str(x).lower() == 'true', default=None,
        help="Resume from checkpoint if exists (default: true)"
    )

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"📋 Loaded config from {args.config}")

    # Merge CLI args with config (CLI takes precedence)
    model_name = args.model_name if args.model_name else config.get("model_name")
    task_id = args.task_id if args.task_id else config.get("task_id")
    skill_dir = args.skill_dir if args.skill_dir else config.get("skill_dir")
    baseline_path = args.baseline_path if args.baseline_path else config.get("baseline_path")

    # Validate required arguments
    if not all([model_name, task_id, skill_dir, baseline_path]):
        parser.error(
            "Missing required arguments. Please provide: "
            "model_name, task_id, skill_dir, baseline_path"
        )

    # Optional arguments (CLI > config > default)
    prompt_name = args.prompt_name or config.get("prompt_name")
    output_path = args.output_path or config.get("output_path")
    max_rounds = args.max_rounds if args.max_rounds != 20 else config.get("max_rounds", 20)
    subset_sizes = args.subset_sizes if args.subset_sizes != [3, 5, 10] else config.get("subset_sizes", [3, 5, 10])
    exploration_param = args.exploration_param if args.exploration_param != 1.0 else config.get("exploration_param", 1.0)
    duplicate_bonus_weight = args.duplicate_bonus_weight if args.duplicate_bonus_weight != 0.1 else config.get("duplicate_bonus_weight", 0.1)
    eval_lm = args.eval_lm or config.get("eval_lm")
    max_examples = args.max_examples or config.get("max_examples")
    seed = args.seed if args.seed is not None else config.get("seed")
    is_agentic = args.is_agentic if hasattr(args, 'is_agentic') else config.get("is_agentic", True)
    resume = args.resume if hasattr(args, 'resume') else config.get("resume", True)

    # Create optimizer
    optimizer = SkillOptimizer(
        model_name=model_name,
        prompt_name=prompt_name,
        skill_dir=Path(skill_dir),
        task_id=task_id,
        baseline_path=Path(baseline_path),
        output_path=Path(output_path) if output_path else None,
        max_rounds=max_rounds,
        subset_sizes=subset_sizes,
        exploration_param=exploration_param,
        duplicate_bonus_weight=duplicate_bonus_weight,
        eval_lm_name=eval_lm,
        is_agentic=is_agentic,
        max_examples=max_examples,
        seed=seed,
        resume=resume
    )

    # Run optimization
    optimizer.optimize()


if __name__ == "__main__":
    main()
