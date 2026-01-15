#!/bin/bash

# Usage: ./run.sh [collect|evaluate|both] [--rollout_version v0|v1|v2|...]
# Default: runs both collect and evaluate with v0 (no skills)

command=${1:-"both"}
rollout_version="v1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        collect|evaluate|both)
            command="$1"
            shift
            ;;
        --rollout_version)
            rollout_version="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

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
    echo "Running collection with rollout_version=${rollout_version}..."
    uv run -m src.collect \
        --task_id=${task_id} \
        --model=${model_name} \
        --prompt_name=${prompt_name} \
        --max_examples=${max_examples} \
        --n_responses=${n_responses} \
        --batch_size=${batch_size} \
        --rollout_version=${rollout_version} \
        --is_agentic
fi

if [[ "$command" == "evaluate" || "$command" == "both" ]]; then
    echo "Running evaluation with rollout_version=${rollout_version}..."
    uv run -m src.evaluate \
        --task_id=${task_id} \
        --model_name=${model_name} \
        --prompt_name=${prompt_name} \
        --max_examples=${max_examples} \
        --n_responses=${n_responses} \
        --batch_size=${batch_size} \
        --rollout_version=${rollout_version}
fi

if [[ "$command" != "collect" && "$command" != "evaluate" && "$command" != "both" ]]; then
    echo "Error: Invalid command '$command'"
    echo "Usage: ./run.sh [collect|evaluate|both] [--rollout_version VERSION]"
    echo "  collect          - Run data collection only"
    echo "  evaluate         - Run evaluation only"
    echo "  both             - Run both collection and evaluation (default)"
    echo "  --rollout_version - Specify rollout version (default: v0)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh collect --rollout_version v1"
    echo "  ./run.sh both --rollout_version v0"
    exit 1
fi