import pandas as pd
import dspy
import json
import os
import litellm
from datasets import load_dataset, concatenate_datasets
from functools import partial
from copy import deepcopy
from typing import List, Any, Optional
from .utils import LM_DICT, batch_inference, use_lm
from .run_task import prepare_task
from .evalutils.webgen import validate_webpage, run_single_instance_eval_web_browser

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
        system_prompt_path: str,
        example: dict,
        trace_dir: Optional[str] = None,
    ):
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_PROMPT.format(instruction=example['prompt'], response=example["output"])}
    ]
    response = lm(messages=messages)
    example["eval_output"] = response[0]
    return response
    
def get_config(task_id: str):
    web_browser_tasks = ["webgen"]
    if task_id in web_browser_tasks:
        return {
            "eval_function": run_single_instance_eval_web_browser,
            "use_process": True,
            "max_workers": 4,
        }
    else:
        return {
            "eval_function": run_single_instance_eval,
            "use_process": False,
            "max_workers": 32,
        }

def load_and_validate_results(
        output_path: str,
        data_loader: 'EvalDataLoader'
    ) -> tuple[List[dict], int]:
    """
    Load existing results and validate they match the data loader's configuration.

    Args:
        output_path: Path to the results file
        data_loader: EvalDataLoader instance to validate against

    Returns:
        tuple: (results, start_idx) where results is the list of existing results
               and start_idx is the index to resume from (0 if starting fresh)
    """
    if not os.path.exists(output_path):
        return [], 0

    try:
        with open(output_path, "r") as f:
            results = json.load(f)
    except json.JSONDecodeError:
        print("⚠️  Could not parse existing results file, starting fresh")
        return [], 0

    # Calculate expected total results (examples * models)
    expected_total = len(data_loader)

    # Validate that existing results don't exceed expected
    if len(results) > expected_total:
        print(f"⚠️  Existing results ({len(results)}) exceed expected total ({expected_total}), starting fresh")
        return [], 0

    # Determine how many examples are completed (results / number of models)
    num_lms = len(data_loader.lms)
    if num_lms == 0:
        return [], 0

    num_completed_examples = len(results) // num_lms

    # Validate that the number of results is a multiple of the number of models
    if len(results) % num_lms != 0:
        print(f"⚠️  Number of results ({len(results)}) is not a multiple of models ({num_lms}), starting fresh")
        return [], 0

    # Generate expected args for completed examples to validate against
    expected_args = data_loader.get_batch_args(0, num_completed_examples)

    # Validate that results match expected args
    for i, (result, expected_arg) in enumerate(zip(results, expected_args)):
        expected_example = expected_arg["example"]
        if (result.get("prompt") != expected_example.get("prompt") or
            result.get("output") != expected_example.get("output")):
            print(f"⚠️  Data mismatch at result index {i}, starting fresh")
            return [], 0

    start_idx = num_completed_examples
    print(f"🔄 Resuming from {start_idx}/{len(data_loader.data)} completed examples ({len(results)}/{expected_total} total results)")
    return results, start_idx

class EvalDataLoader:
    """
    Global data loader for evaluation that provides arguments for batch inference.
    """
    def __init__(
        self,
        data: List[dict],
        lms: List[dspy.LM],
        system_prompt_path: str,
        trace_dir: str
    ):
        """
        Initialize the data loader.

        Args:
            data: Full list of examples to evaluate
            lms: List of language models to use
            system_prompt_path: Path to the evaluation system prompt
            trace_dir: Directory to save trace files
        """
        self.data = data
        self.lms = lms
        self.system_prompt_path = system_prompt_path
        self.trace_dir = trace_dir
        self.total_size = len(data) * len(lms)

    def get_batch_args(self, batch_start: int, batch_size: int) -> List[dict]:
        """
        Get arguments for a specific batch.

        Args:
            batch_start: Starting index in the original data
            batch_size: Number of examples per batch

        Returns:
            List of argument dictionaries for batch_inference
        """
        batch_end = min(batch_start + batch_size, len(self.data))
        batch_data = self.data[batch_start:batch_end]

        return [
            {
                "lm": lm,
                "system_prompt_path": self.system_prompt_path,
                "example": example,
                "trace_dir": self.trace_dir
            } for example in batch_data for lm in self.lms
        ]

    def __len__(self):
        """Return total number of evaluation tasks (examples * models)."""
        return self.total_size

def run_task_eval(
        task_id: str,
        data_path: str,
        model_names: List[str] = [],
        max_examples: Optional[int] = None,
        batch_size: int = 16,
        resume: bool = True,
    ):
    with open(data_path, "r") as f:
        data = json.load(f)
        data = data[:max_examples]

    cwd = os.getcwd()
    eval_prompt_path = cwd + f"/data/{task_id}/eval.md"
    output_path = data_path.replace(".json", "_eval.json")

    # Create trace directory
    trace_dir = data_path.replace(".json", "_eval_traces")
    os.makedirs(trace_dir, exist_ok=True)

    lms = [LM_DICT[name] for name in model_names]

    config = get_config(task_id)

    # Initialize global data loader first (needed for validation)
    data_loader = EvalDataLoader(
        data=data,
        lms=lms,
        system_prompt_path=eval_prompt_path,
        trace_dir=trace_dir
    )

    # Load existing results if resuming
    if resume:
        results, start_idx = load_and_validate_results(output_path, data_loader)
    else:
        results, start_idx = [], 0

    eval_function = config["eval_function"]

    # Process remaining data in batches
    for i in range(start_idx, len(data_loader), batch_size):
        # Get batch arguments from data loader
        args_list = data_loader.get_batch_args(
            batch_start=i,
            batch_size=batch_size
        )

        batch_results = batch_inference(
            eval_function,
            args_list,
            use_process=config["use_process"],
            max_workers=config["max_workers"],
        )
        results.extend(batch_results)

        # Write partial results after each batch
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved partial results ({len(results)}/{len(data_loader)} completed)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_names", type=str, nargs='+', default=[])
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh instead of resuming from existing results")
    args = parser.parse_args()

    run_task_eval(
        task_id=args.task_id,
        data_path=args.data_path,
        model_names=args.model_names,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        resume=args.resume,
    )