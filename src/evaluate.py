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
from .run_task import prepare_task

import os
import numpy as np
import tqdm
import copy
import time
import json
import argparse

USER_PROMPT = """### User Instruction
{instruction}
### Model Response
{response}"""

def run_single_instance_eval(
        lm: dspy.LM,
        system_prompt: str,
        example: dict,
    ):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_PROMPT.format(instruction=example['prompt'], response=example["output"])}
    ]
    response = lm(messages=messages)
    example["eval_output"] = response[0]
    return response

def run_task_eval(
        task_id: str,
        data_path: str,
        max_examples: Optional[int] = None,
    ):
    with open(data_path, "r") as f:
        data = json.load(f)
        data = data[:max_examples]
    
    eval_prompt_path = f"data/{task_id}/eval.md"
    with open(eval_prompt_path, "r") as f:
        eval_prompt = f.read()
    
    lm = LM_DICT["gpt-4.1-mini"]
    _ = batch_inference(
        run_single_instance_eval, 
        [{"lm": lm, "system_prompt": eval_prompt, "example": example} for example in data]
    )
    output_path = data_path.replace(".json", "_eval.json")
    with open(output_path, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    run_task_eval(
        task_id=args.task_id,
        data_path=args.data_path,
        max_examples=args.max_examples,
    )