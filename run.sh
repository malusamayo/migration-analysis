#!/bin/bash

# Usage: ./run.sh [collect|evaluate]
# Default: runs both collect and evaluate

command=${1:-"both"}

# task_id="webgen"
task_id="webtest"

# model_name="llama3.3-70b"
model_name="qwen3-coder-30b-a3b"
# model_name="gemini-2.5-flash"
# model_name="gemini-3-flash-preview"
# model_name="gemini-3-pro-preview"
# model_name="kimi-k2-thinking"
# model_name="gpt-5.1-codex-mini"

prompt_name="default"
# prompt_name="SKILL"

max_examples=10
n_responses=3
batch_size=32

if [[ "$command" == "collect" || "$command" == "both" ]]; then
    echo "Running collection..."
    uv run -m src.collect \
        --task_id=${task_id} \
        --model=${model_name} \
        --prompt_name=${prompt_name} \
        --max_examples=${max_examples} \
        --n_responses=${n_responses} \
        --batch_size=${batch_size} \
        --is_agentic
fi

if [[ "$command" == "evaluate" || "$command" == "both" ]]; then
    echo "Running evaluation..."
    uv run -m src.evaluate \
        --task_id=${task_id} \
        --model_name=${model_name} \
        --prompt_name=${prompt_name} \
        --max_examples=${max_examples} \
        --n_responses=${n_responses} \
        --batch_size=${batch_size}
fi

if [[ "$command" != "collect" && "$command" != "evaluate" && "$command" != "both" ]]; then
    echo "Error: Invalid command '$command'"
    echo "Usage: ./run.sh [collect|evaluate|both]"
    echo "  collect  - Run data collection only"
    echo "  evaluate - Run evaluation only"
    echo "  both     - Run both collection and evaluation (default)"
    exit 1
fi