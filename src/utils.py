import os
import subprocess
from typing import List, Any
import dspy
import litellm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import json
import tqdm
from dotenv import load_dotenv
import yaml
from openhands.sdk import LLM

def use_lm(lm, n=1):
    def decorator(program):
        def wrapper(*args, **kwargs):
            max_retries = 3
            initial_delay = 1
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    with dspy.context(lm=lm):
                        return program(*args, **kwargs)
                except litellm.APIError as e:
                    if attempt < max_retries - 1:
                        print(f"API Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                        print(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff
                    else:
                        raise
                except Exception as e:
                    print(f"Error: {e}")
                    return dspy.Example(output="")
        return wrapper
    return decorator

def batch_inference(
    program,
    args_list,
    use_process=False,
    max_workers=32,
    on_batch_complete=None,
) -> List[Any]:
    """
    Execute inference on a list of arguments in parallel.

    Args:
        program: Function to execute
        args_list: List of argument dictionaries
        use_process: Whether to use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
        max_workers: Maximum number of concurrent workers
        on_batch_complete: Optional callback function(completed_results, total_count) called every batch_size completions

    Returns:
        List of results in the same order as args_list
    """
    futures = {}
    results = [None] * len(args_list)
    completed_count = 0

    if use_process:
        executor = ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1)
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)

    with executor:
        for i, args in enumerate(args_list):
            future = executor.submit(
                program,
                **args
            )
            futures[future] = i

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            index = futures[future]
            results[index] = result
            completed_count += 1

            # Invoke callback every completion
            if on_batch_complete:
                # Filter out None values (incomplete results)
                completed_results = [r for r in results if r is not None]
                on_batch_complete(completed_results, len(args_list))

    return results

def load_lmdict(yaml_path: str):
    """Load YAML model list and construct a dict of dspy.LM objects."""

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    lm_dict = {}

    for m in models:
        name = m["name"]
        model_id = m["model"]

        # Base kwargs: all fields except name/model and env-resolved fields
        kwargs = {}
        for k, v in m.items():
            if k in ("name", "model"):
                continue
            elif isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_var = v[2:-1]
                kwargs[k] = os.getenv(env_var)
                if k == "vertex_credentials":
                    kwargs[k] = json.dumps(json.load(open(kwargs[k])))
            else:
                kwargs[k] = v

        # Create LM
        lm_dict[name] = dspy.LM(model_id, **kwargs)

    return lm_dict


def resolve_api_key(lm: dspy.LM) -> str | None:
    api_key = lm.kwargs.get("api_key")
    if api_key == "gcloud":
        return subprocess.check_output(["gcloud", "auth", "print-access-token"]).decode().strip()
    return api_key


def build_sdk_llm(lm: dspy.LM) -> LLM:
    """Construct an OpenHands SDK LLM from a DSPy LM, including custom pricing."""

    return LLM(
        model=lm.model,
        base_url=lm.kwargs.get("api_base"),
        api_key=resolve_api_key(lm),
        input_cost_per_token=lm.kwargs.get("input_cost_per_token"),
        output_cost_per_token=lm.kwargs.get("output_cost_per_token"),
    )


load_dotenv(override=True)
LM_DICT = load_lmdict("configs/models.yaml")
