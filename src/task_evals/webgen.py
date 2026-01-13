import re
from typing import List
import dspy
import copy
from copy import deepcopy
from typing import List, Any, Optional

import os
import numpy as np
import tqdm
import copy
import time
import json

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.browser_use import BrowserToolSet
import tempfile
import subprocess
import re
import socket

from .screenshot import take_screenshot
import asyncio
import base64

def validate_webpage(output: str) -> tuple[bool, str]:    
    # Extract HTML code from markdown code blocks if present
    html_match = re.search(r'```html\s*(.*?)\s*```', output, re.DOTALL)
    if html_match:
        html_content = html_match.group(1)
    else:
        if len(output) < 20:
            return False, "HTML content is too short."
        if '```html' in output:
            return False, "HTML code block not properly closed."
        return False, "No HTML code block found."
    return html_match, html_content

def generate_retry_function_webpage(lm: dspy.LM, messages: List, error_msg: str):
    lm = copy.deepcopy(lm)
    if error_msg == "HTML code block not properly closed.":
        if lm.kwargs.get("max_tokens"):
            lm.kwargs["max_tokens"] *= 2
        if lm.kwargs.get("max_new_tokens"):
            lm.kwargs["max_new_tokens"] *= 2
    elif error_msg == "HTML content is too short.":
        messages.append({"role": "user", "content": "Please provide a more complete HTML page. Do not stop early."})
    return lm, messages



USER_PROMPT = """Please evaluate the generated website based on the user query and provided guidelines.
### User query
{instruction}

### Website path
{website_path}"""

def start_server(html_content: str) -> tuple[subprocess.Popen, int, str]:
    # Create a temporary directory and HTML file
    temp_dir = tempfile.mkdtemp()
    html_file_path = os.path.join(temp_dir, "generated_website.html")

    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    def get_free_port():
        s = socket.socket()
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return port
    free_port = get_free_port()

    # Start a simple HTTP server in the background
    server_process = subprocess.Popen(
        ['/usr/bin/python3', '-m', 'http.server', str(free_port)],
        cwd=temp_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)  # Give the server a moment to start
    return server_process, free_port, temp_dir

def run_ui_agent(
        lm: dspy.LM,
        system_prompt_path: str,
        example: dict,
        website_path: str,
        workspace: str,
        trace_dir: Optional[str] = None,
    ):
    example = copy.deepcopy(example)

    def extract_scores(eval_output: str) -> dict:
        scores = {}

        pattern = r'###\s+(\w+(?:\s+\w+)?)\s*\n(?:.*?\n)*?\*\*Score\*\*:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)'
        matches = re.finditer(pattern, eval_output, re.DOTALL | re.IGNORECASE)

        for match in matches:
            category = match.group(1).strip()
            score = float(match.group(2))
            scores[category] = score

        return scores

    llm = LLM(
        model=lm.model
    )
    agent = Agent(
        llm=llm,
        temperature=1.0,
        system_prompt_filename=system_prompt_path,
        tools=[
            Tool(name=BrowserToolSet.name),
        ],
    )

    try:
        conversation = Conversation(agent=agent, workspace=workspace)
        instruction = USER_PROMPT.format(
            instruction=example['prompt'],
            website_path=website_path
        )

        conversation.send_message(instruction)
        conversation.run()

        events = [event.model_dump() for event in conversation.state.events]
        eval_output = events[-1]["llm_message"]["content"][0]["text"]
        conversation_data = {
            "conversation_id": str(conversation.id),
            "eval_lm": lm.model,
            "eval_output": eval_output,
            "eval_scores": extract_scores(eval_output),
        }
        example["eval_result"] = copy.deepcopy(conversation_data)
        conversation_data["events"] = events

        # Write trace to separate file if trace_dir is provided
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            trace_filename = f"{conversation.id}.json"
            trace_path = os.path.join(trace_dir, trace_filename)
            with open(trace_path, "w") as trace_file:
                json.dump(conversation_data, trace_file, indent=2)
            example["eval_result"]["trace_path"] = trace_path

        return example

    except Exception as e:
        print(f"Error during web browser evaluation: {e}")
        return example

    finally:
        print("🧹 Cleaning up conversation...")
        conversation.close()


def run_ui_agent_browser_use(
        lm: dspy.LM,
        system_prompt_path: str,
        example: dict,
        website_path: str,
        workspace: str,
        trace_dir: Optional[str] = None,
    ):
    """
    Alternative implementation using browser_use library instead of OpenHands SDK.
    """
    from browser_use import Agent as BrowserAgent, Browser, ChatGoogle

    example = copy.deepcopy(example)

    def extract_scores(eval_output: str) -> dict:
        scores = {}

        pattern = r'###\s+(\w+(?:\s+\w+)?)\s*\n(?:.*?\n)*?\*\*Score\*\*:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)'
        matches = re.finditer(pattern, eval_output, re.DOTALL | re.IGNORECASE)

        for match in matches:
            category = match.group(1).strip()
            score = float(match.group(2))
            scores[category] = score

        return scores

    async def run_browser_agent():
        # Read system prompt
        system_content = ""
        if system_prompt_path and os.path.exists(system_prompt_path):
            with open(system_prompt_path, 'r') as f:
                system_content = f.read()

        # Create browser instance
        browser = Browser(
            headless=True,
        )

        llm = ChatGoogle(
            model=lm.model.split('/')[-1],
            vertexai=True,
        )

        # Prepare task instruction
        instruction = USER_PROMPT.format(
            instruction=example['prompt'],
            website_path=website_path
        )

        # Combine system prompt with instruction
        full_task = f"{system_content}\n\n{instruction}" if system_content else instruction

        try:
            # Create agent with the task
            agent = BrowserAgent(
                task=full_task,
                llm=llm,
                browser=browser,
            )

            # Run the agent
            history = await agent.run()

            # Extract evaluation output from history
            eval_output = ""
            if history and hasattr(history, 'all_results') and history.all_results:
                # Get the last result which should be the done action with final output
                final_result = history.all_results[-1]
                if hasattr(final_result, 'extracted_content') and final_result.extracted_content:
                    eval_output = final_result.extracted_content
                elif hasattr(final_result, 'long_term_memory') and final_result.long_term_memory:
                    eval_output = final_result.long_term_memory
                else:
                    eval_output = str(final_result)
            else:
                assert False, f"{history}"

            # Create conversation data with full history details
            conversation_data = {
                "eval_lm": lm.model,
                "eval_output": eval_output,
                "eval_scores": extract_scores(eval_output),
            }

            return conversation_data

        finally:
            print("🧹 Cleaning up browser...")
            try:
                await browser.close()
            except:
                pass

    try:
        # Run the async function
        conversation_data = asyncio.run(run_browser_agent())

        example["eval_result"] = copy.deepcopy(conversation_data)

        # Write trace to separate file if trace_dir is provided
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            trace_filename = f"browser_use_{time.time()}.json"
            trace_path = os.path.join(trace_dir, trace_filename)
            with open(trace_path, "w") as trace_file:
                json.dump(conversation_data, trace_file, indent=2)
            example["eval_result"]["trace_path"] = trace_path

        return example

    except Exception as e:
        print(f"Error during browser_use evaluation: {e}")
        import traceback
        traceback.print_exc()
        return example


def run_screenshot_eval(
        lm: dspy.LM,
        system_prompt_path: str,
        example: dict,
        website_url: str,
        workspace: str,
    ):
    example = copy.deepcopy(example)

    # Take screenshot
    screenshot_path = os.path.join(workspace, f"screenshot_{time.time()}.png")

    asyncio.run(take_screenshot(website_url, screenshot_path))
    print(f"Screenshot saved to {screenshot_path}")

    with open(screenshot_path, "rb") as img_file:
        data = base64.b64encode(img_file.read()).decode('utf-8')

    # Prepare evaluation prompt with screenshot
    instruction = USER_PROMPT.format(
        instruction=example['prompt'],
        website_path=website_url
    )

    # Read system prompt if provided
    system_content = ""
    if system_prompt_path and os.path.exists(system_prompt_path):
        with open(system_prompt_path, 'r') as f:
            system_content = f.read()

    # Evaluate using the LLM with the screenshot
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{data}"
                }
            }
        ]
    })

    # Generate evaluation using the language model
    response = lm(messages=messages)
    eval_output = response[0]

    # Extract scores from the evaluation
    # regex to match Analysis: [2–4 paragraphs addressing all criteria, referencing the instruction] Grade: [1–5]
    def extract_grade(eval_output: str) -> Optional[int]:
        grade_pattern = r'Grade:\s*(\d+)'
        grade_match = re.search(grade_pattern, eval_output, re.IGNORECASE)

        eval_scores = {}
        if grade_match:
            eval_scores['Appearance'] = int(grade_match.group(1))
        return eval_scores

    eval_scores = extract_grade(eval_output)

    # Store evaluation results
    example["eval_result"] = {
        "eval_lm": lm.model,
        "eval_output_vision": eval_output,
        "eval_scores": eval_scores,
        "screenshot_path": screenshot_path
    }

    return example


def run_single_instance_eval(
        lm: dspy.LM,
        system_prompt_path: str,
        example: dict,
        trace_dir: Optional[str] = None,
    ):
    example = copy.deepcopy(example)

    output_text = example["output"]
    is_valid, html_content = validate_webpage(output_text)
    if not is_valid:
        print(f"Warning: HTML validation failed: {html_content}")
        # Try to use the output directly if validation failed
        html_content = output_text
    server_process, free_port, temp_dir = start_server(html_content=html_content)
    website_url = f"http://localhost:{free_port}/generated_website.html"

    result = run_ui_agent_browser_use(
        lm=lm,
        system_prompt_path=system_prompt_path,
        example=example,
        website_path=website_url,
        workspace=temp_dir,
        trace_dir=trace_dir,
    )

    system_prompt_path = system_prompt_path.replace('eval.md', 'eval_screenshot.md')
    result["eval_result_vision"] = run_screenshot_eval(
        lm=lm,
        system_prompt_path=system_prompt_path,
        example=example,
        website_url=website_url,
        workspace=trace_dir,
    )["eval_result"]
        
    # Clean up: stop the server and remove temp files
    print("🧹 Terminating temporary HTTP server...")
    server_process.terminate()
    server_process.wait()

    # Remove temporary files
    print("🧹 Removing temporary files...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    return result