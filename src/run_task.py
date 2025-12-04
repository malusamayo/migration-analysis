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

import os
import numpy as np
import tqdm
import copy
import time
import json
import argparse

def run_single_instance(
        lm: dspy.LM,
        system_prompt: str,
        example: dict,
    ):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['prompt']}
    ]
    response = lm(messages=messages)
    example["output"] = response[0]
    example["outputs"] = response
    return response

def prepare_task(
        task_id: str,
        prompt_name: str = "default",
        max_examples: Optional[int] = None
    ):
    assert task_id in ["webgen"], "Unsupported task_id"

    data_path = f"data/{task_id}/{task_id}.csv"
    data = pd.read_csv(data_path)
    data = data.to_dict(orient="records")[:max_examples]

    task_prompt_path = f"data/{task_id}/prompts/{prompt_name}.md"
    with open(task_prompt_path, "r") as f:
        task_prompt = f.read()
    
    eval_prompt_path = f"data/{task_id}/eval.md"
    with open(eval_prompt_path, "r") as f:
        eval_prompt = f.read()
    
    return data, task_prompt, eval_prompt

def run_task(
        task_id: str,
        model_name: str,
        prompt_name: str = "default",
        max_examples: Optional[int] = None,
    ):
    data, task_prompt, _ = prepare_task(task_id, max_examples=max_examples)
    lm = LM_DICT[model_name]
    _ = batch_inference(
        run_single_instance, 
        [{"lm": lm, "system_prompt": task_prompt, "example": example} for example in data]
    )
    output_path = f"results/{task_id}/{model_name}_{prompt_name}.json"
    with open(output_path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task with specified model and prompt.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use.")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID to run.")
    parser.add_argument("--prompt_name", type=str, default="default", help="Prompt name to use.")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process.")
    args = parser.parse_args()

    run_task(
        task_id=args.task_id,
        model_name=args.model,
        prompt_name=args.prompt_name,
        max_examples=args.max_examples,
    )