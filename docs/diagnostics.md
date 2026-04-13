Your goal is to optimize agent implementation at the given task

You have an initial agent implementation available at `scripts/agents/multi_agent_rb_no_validation.py`

1. You can run the agent on some examples with `uv run python -m src.collect --config tasks/replicatorbench/run.yaml`.
2. You can evaluate the results with `uv run python -m src.evaluate --config tasks/replicatorbench/run.yaml`. The eval results will be available in a yaml file.
3. You can inspect the task agent trajectory -- they are stored in `results/replicatorbench/qwen3-coder-30b-a3b_default/rollouts/v0_multi_agent_rb/example0_rollout0_logs/trace_*.md`. The exact path depends on the task_id, model_name, prompt_name, agent_file specified in the config file.
4. Then you should diagnoze the results and create new agent implementation under `scripts/agents/`. You should update the config file and test you new implementation through 1-3.

For this specific task, my intuition is to use multi-agents and have each subagent do a subtask independently.

You should repeat the above procedure until we have satisafactory results. For this task, we are aiming for a performance around ~0.8.