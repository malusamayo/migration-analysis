#!/bin/bash
# Wrapper script for review commands with default arguments

# Default values
DEFAULT_TASK_ID="webtest"
# DEFAULT_MODEL_NAME="gemini-2.5-flash"
DEFAULT_MODEL_NAME="qwen3-coder-30b-a3b"
DEFAULT_PROMPT_NAME="default"
DEFAULT_NUM_EXAMPLES="10"
DEFAULT_COMPARISON_MODEL="gemini-2.5-flash"
DEFAULT_ROLLOUT_VERSION="v0"

# Parse arguments to check if defaults should be applied
ARGS=()
HAS_TASK_ID=false
HAS_MODEL_NAME=false
HAS_PROMPT_NAME=false
HAS_NUM_EXAMPLES=false
HAS_COMPARISON_MODEL=false
HAS_ROLLOUT_VERSION=false
MODE=""

# First pass: identify the mode and which arguments are provided
for arg in "$@"; do
    if [[ "$arg" == "analyze" ]] || [[ "$arg" == "generate-patches" ]] || [[ "$arg" == "apply-patches" ]] || [[ "$arg" == "edit-patch" ]] || [[ "$arg" == "generate-skills" ]]  || [[ "$arg" == "analyze-cmp" ]]; then
        MODE="$arg"
    elif [[ "$arg" == "--task_id" ]] || [[ "$arg" == "--task_id="* ]]; then
        HAS_TASK_ID=true
    elif [[ "$arg" == "--model_name" ]] || [[ "$arg" == "--model_name="* ]]; then
        HAS_MODEL_NAME=true
    elif [[ "$arg" == "--prompt_name" ]] || [[ "$arg" == "--prompt_name="* ]]; then
        HAS_PROMPT_NAME=true
    elif [[ "$arg" == "--num_examples" ]] || [[ "$arg" == "--num_examples="* ]]; then
        HAS_NUM_EXAMPLES=true
    elif [[ "$arg" == "--comparison_model" ]] || [[ "$arg" == "--comparison_model="* ]]; then
        HAS_COMPARISON_MODEL=true
    elif [[ "$arg" == "--rollout_version" ]] || [[ "$arg" == "--rollout_version="* ]]; then
        HAS_ROLLOUT_VERSION=true
    fi
done

# Determine which defaults apply for the selected mode
NEEDS_TASK_ID=false
NEEDS_MODEL_NAME=false
NEEDS_PROMPT_NAME=false
NEEDS_NUM_EXAMPLES=false
NEEDS_COMPARISON_MODEL=false
NEEDS_ROLLOUT_VERSION=false

case "$MODE" in
    analyze)
        NEEDS_TASK_ID=true
        NEEDS_MODEL_NAME=true
        NEEDS_PROMPT_NAME=true
        NEEDS_NUM_EXAMPLES=true
        NEEDS_COMPARISON_MODEL=true
        NEEDS_ROLLOUT_VERSION=true
        ;;
    generate-patches)
        NEEDS_TASK_ID=true
        NEEDS_MODEL_NAME=true
        NEEDS_PROMPT_NAME=true
        NEEDS_NUM_EXAMPLES=true
        ;;
    generate-skills)
        NEEDS_TASK_ID=true
        NEEDS_MODEL_NAME=true
        NEEDS_PROMPT_NAME=true
        ;;
    apply-patches)
        NEEDS_TASK_ID=true
        NEEDS_MODEL_NAME=true
        NEEDS_PROMPT_NAME=true
        ;;
    edit-patch)
        NEEDS_MODEL_NAME=true
        NEEDS_PROMPT_NAME=true
        ;;
    cross-model)
        NEEDS_TASK_ID=true
        NEEDS_PROMPT_NAME=true
        NEEDS_NUM_EXAMPLES=true
        NEEDS_COMPARISON_MODEL=true
        NEEDS_ROLLOUT_VERSION=true
        ;;
    analyze-cmp)
        NEEDS_NUM_EXAMPLES=true
        NEEDS_COMPARISON_MODEL=true
        ;;
esac

if [[ "$NEEDS_TASK_ID" == true && "$HAS_TASK_ID" == false ]]; then
    ARGS+=("--task_id" "$DEFAULT_TASK_ID")
fi

if [[ "$NEEDS_MODEL_NAME" == true && "$HAS_MODEL_NAME" == false ]]; then
    ARGS+=("--model_name" "$DEFAULT_MODEL_NAME")
fi

if [[ "$NEEDS_PROMPT_NAME" == true && "$HAS_PROMPT_NAME" == false ]]; then
    ARGS+=("--prompt_name" "$DEFAULT_PROMPT_NAME")
fi

if [[ "$NEEDS_NUM_EXAMPLES" == true && "$HAS_NUM_EXAMPLES" == false ]]; then
    ARGS+=("--num_examples" "$DEFAULT_NUM_EXAMPLES")
fi

if [[ "$NEEDS_COMPARISON_MODEL" == true && "$HAS_COMPARISON_MODEL" == false ]]; then
    ARGS+=("--comparison_model" "$DEFAULT_COMPARISON_MODEL")
fi

if [[ "$NEEDS_ROLLOUT_VERSION" == true && "$HAS_ROLLOUT_VERSION" == false ]]; then
    ARGS+=("--rollout_version" "$DEFAULT_ROLLOUT_VERSION")
fi

# Run with user args + defaults
echo "Running review with arguments: $*" "${ARGS[@]}"
uv run -m src.review.main "$@" "${ARGS[@]}"
