#!/bin/bash
# Wrapper script for review commands with default arguments

# Default values
DEFAULT_TASK_ID="webtest"
DEFAULT_MODEL_NAME="gemini-2.5-flash"
DEFAULT_PROMPT_NAME="SKILL"
DEFAULT_NUM_EXAMPLES="10"
DEFAULT_COMPARISON_MODEL="gemini-2.5-flash"

# Parse arguments to check if defaults should be applied
ARGS=()
HAS_TASK_ID=false
HAS_MODEL_NAME=false
HAS_PROMPT_NAME=false
HAS_NUM_EXAMPLES=false
HAS_COMPARISON_MODEL=false
MODE=""

# First pass: identify the mode and which arguments are provided
for arg in "$@"; do
    if [[ "$arg" == "analyze" ]] || [[ "$arg" == "generate-patches" ]] || [[ "$arg" == "apply-patches" ]] || [[ "$arg" == "edit-patch" ]]; then
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
    fi
done

# Add defaults based on mode
if [[ "$HAS_TASK_ID" == false ]]; then
    ARGS+=("--task_id" "$DEFAULT_TASK_ID")
fi

if [[ "$HAS_MODEL_NAME" == false ]]; then
    ARGS+=("--model_name" "$DEFAULT_MODEL_NAME")
fi

if [[ "$HAS_PROMPT_NAME" == false ]]; then
    ARGS+=("--prompt_name" "$DEFAULT_PROMPT_NAME")
fi

# Mode-specific defaults
if [[ "$MODE" == "analyze" ]]; then
    if [[ "$HAS_NUM_EXAMPLES" == false ]]; then
        ARGS+=("--num_examples" "$DEFAULT_NUM_EXAMPLES")
    fi
    if [[ "$HAS_COMPARISON_MODEL" == false ]]; then
        ARGS+=("--comparison_model" "$DEFAULT_COMPARISON_MODEL")
    fi
elif [[ "$MODE" == "generate-patches" ]]; then
    if [[ "$HAS_NUM_EXAMPLES" == false ]]; then
        ARGS+=("--num_examples" "$DEFAULT_NUM_EXAMPLES")
    fi
fi

# Run with user args + defaults
echo "Running review with arguments: $*" "${ARGS[@]}"
uv run -m src.review.main "$@" "${ARGS[@]}"
