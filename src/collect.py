import pandas as pd
import dspy
import json
import os
import litellm
from datasets import load_dataset, concatenate_datasets
from functools import partial
from copy import deepcopy
from typing import List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils import LM_DICT, batch_inference, use_lm
from .evalutils.webgen import validate_webpage, generate_retry_function_webpage

import os
import numpy as np
import tqdm
import copy
import time
import json
import argparse
import re

from openhands.sdk import LLM, Agent, Conversation, Tool, AgentContext
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.sdk.context import (
    Skill,
)


def run_single_instance(
        lm: dspy.LM,
        system_prompt: str,
        example: dict,
        seed: int,
        max_retries: int = 2,
        validation_fn: Optional[callable] = None,
        retry_gen_fn: Optional[callable] = None,
    ):
    """
    Run a single instance with retry logic and optional output validation.

    Args:
        lm: Language model to use
        system_prompt: System prompt for the model
        example: Example data dictionary
        seed: Random seed/rollout ID
        max_retries: Maximum number of retry attempts (default: 3)
        validation_fn: Optional function that takes the output string and returns (is_valid: bool, error_msg: str)
                      If None, no validation is performed.

    Returns:
        dict: Example with added fields including 'output', 'rollout_id', and optionally 'validation_errors'
    """
    example = copy.deepcopy(example)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['prompt']}
    ]

    validation_errors = []

    for _ in range(max_retries):
        response = lm(messages=messages, rollout_id=seed)
        output = response[0]

        # Validate output if validation function is provided
        if validation_fn is not None:
            is_valid, error_msg = validation_fn(output)
            if not is_valid:
                lm, messages = retry_gen_fn(lm, messages, error_msg)
                validation_errors.append(error_msg)
                continue
        
        # Success - output is valid or no validation required
        example["rollout_id"] = seed
        example["output"] = output
        if validation_errors:
            example["validation_errors"] = validation_errors
        return example

    return example

def run_single_instance_agentic(
        lm: dspy.LM,
        system_prompt_path: str,
        example: dict,
        workspace: str,
    ):
    """
    Run a single instance using OpenHands agents.

    Args:
        lm: Language model to use
        system_prompt_path: Path to the system prompt file
        example: Example data dictionary containing 'prompt' field
        workspace: Workspace directory for the agent

    Returns:
        dict: Example with added 'eval_result' field containing evaluation output and scores
    """
    example = copy.deepcopy(example)

    tools = [
        Tool(
            name=TerminalTool.name,
        ),
        Tool(name=FileEditorTool.name),
    ]

    llm = LLM(model=lm.model)

    # agent_context = AgentContext(
    #     skills=[
    #         Skill.load(path=system_prompt_path),
    #     ],
    # )

    system_prompt_path = os.path.abspath(system_prompt_path)

    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_filename=system_prompt_path,
        # agent_context=agent_context,
    )

    conversation = None
    try:
        conversation = Conversation(agent=agent, workspace=workspace)
        instruction = example['prompt']

        conversation.send_message(instruction)
        conversation.run()

        events = [event.model_dump() for event in conversation.state.events]
        if events[-1]["kind"] == "ObservationEvent":
            eval_output = events[-1]["observation"]["content"][0]["text"]
        elif events[-1]["kind"] == "MessageEvent":
            eval_output = events[-1]["llm_message"]["content"][0]["text"]
        else:
            print("Unexpected final event type:", events[-1]["kind"])
        conversation_data = {
            "conversation_id": str(conversation.id),
            "eval_lm": lm.model,
            "eval_output": eval_output,
        }
        example["eval_result"] = copy.deepcopy(conversation_data)
        conversation_data["events"] = events

        # Write trace to separate file if trace_dir is provided
        trace_filename = f"trace_{conversation.id}.json"
        trace_path = os.path.join(workspace, trace_filename)
        with open(trace_path, "w") as trace_file:
            json.dump(conversation_data, trace_file, indent=2)
        example["eval_result"]["trace_path"] = trace_path

        return example

    except Exception as e:
        print(f"Error during agentic execution: {e}")
        return example

    finally:
        if conversation:
            print("🧹 Cleaning up conversation...")
            conversation.close()

def prepare_task(
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        n_responses: int = 1,
        max_examples: Optional[int] = None,
    ):
    assert task_id in ["webgen", "webtest"], "Unsupported task_id"

    data_path = f"data/{task_id}/{task_id}.csv"
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} examples from {data_path}, keeping max_examples={max_examples}")
    data = data.to_dict(orient="records")[:max_examples]

    task_prompt_path = f"data/{task_id}/prompts/{prompt_name}.md"
    with open(task_prompt_path, "r") as f:
        task_prompt = f.read()
    
    eval_prompt_path = f"data/{task_id}/eval.md"
    with open(eval_prompt_path, "r") as f:
        eval_prompt = f.read()

    for example_id, example in enumerate(data):
        for rollout_id in range(n_responses):
            workspace = f"results/{task_id}/{model_name}_{prompt_name}_agentic_workspace/example{example_id}_rollout{rollout_id}/"
            os.makedirs(workspace, exist_ok=True)
            if task_id == "webtest":
                with open(os.path.join(workspace, "index.html"), "w") as f:
                    f.write(example["html_content"])
    return data, task_prompt, eval_prompt

def run_task(
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        is_agentic: bool = False,
        max_examples: Optional[int] = None,
        n_responses: int = 1,
    ):
    """
    Run a task with specified model and prompt.

    Args:
        task_id: Task identifier
        model_name: Name of the model to use
        prompt_name: Name of the prompt template (default: "default")
        max_examples: Maximum number of examples to process
        n_responses: Number of responses to generate per example
        max_retries: Maximum number of retry attempts for each instance (default: 3)
        validation_fn: Optional validation function for output validation
    """
    data, task_prompt, _ = prepare_task(task_id, 
                                        model_name=model_name, 
                                        prompt_name=prompt_name, 
                                        max_examples=max_examples, 
                                        n_responses=n_responses)
    lm = LM_DICT[model_name]
    if is_agentic:
        results = batch_inference(
            run_single_instance_agentic,
            [{
                "lm": lm,
                "system_prompt_path": f"data/{task_id}/prompts/{prompt_name}.md",
                "example": example,
                "workspace": f"results/{task_id}/{model_name}_{prompt_name}_agentic_workspace/example{example_id}_rollout{rollout_id}/",
            } for example_id, example in enumerate(data) for rollout_id in range(n_responses)],
            use_process=True,
            max_workers=16,
        )
    else:
        results = batch_inference(
            run_single_instance,
            [{
                "lm": lm,
                "system_prompt": task_prompt,
                "example": example,
                "seed": seed,
                "validation_fn": validate_webpage,
                "retry_gen_fn": generate_retry_function_webpage,
            } for example in data for seed in range(n_responses)]
        )
    output_path = f"results/{task_id}/{model_name}_{prompt_name}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task with specified model and prompt.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use.")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to run.")
    parser.add_argument("--prompt_name", type=str, default="default", help="Prompt name to use.")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process.")
    parser.add_argument("--n_responses", type=int, default=1, help="Number of responses to generate per example.")
    parser.add_argument("--is_agentic", action="store_true", help="Whether to use agentic execution.")
    args = parser.parse_args()

    run_task(
        task_id=args.task_id,
        model_name=args.model,
        prompt_name=args.prompt_name,
        is_agentic=args.is_agentic,
        max_examples=args.max_examples,
        n_responses=args.n_responses,
    )