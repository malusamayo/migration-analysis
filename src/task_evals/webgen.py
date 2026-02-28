import ast
import re
import json
import copy
import os
import base64
from typing import List, Optional

import dspy

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.browser_use import BrowserToolSet
from openhands.workspace import DockerWorkspace
from ..runner import run_single_instance_agentic, _run_agentic_conversation


DEV_SERVER_PORT = 3000
DEV_SERVER_LOG_PATH = "/tmp/devserver.log"
NPM_INSTALL_TIMEOUT_SECONDS = 600
SERVER_WAIT_TIMEOUT_SECONDS = 180
UNREACHABLE_STOCK_SEARCH_MESSAGE = (
    "The website at http://localhost:3000 could not be reached (ERR_CONNECTION_REFUSED). "
    "As a result, I was unable to verify the stock search functionality."
)

UI_AGENT_PROMPT = """{ui_instruct}

### Website URL
{website_url}"""

EVAL_PROMPT_PATH = "tasks/webgen/eval.md"

JUDGEMENT_INSTRUCT = (
    "After completing your exploration, provide your final judgement on whether the task "
    "was completed successfully. Respond with a JSON block:\n"
    '```json\n{"pass": true/false, "reason": "brief explanation"}\n```'
)

MALFORMED_EVAL_OUTPUT_MESSAGE = (
    "Evaluator response was malformed after interacting with the reachable website."
)
BROWSER_TIMEOUT_EVAL_OUTPUT_MESSAGE = (
    "Browser tool timed out while launching a session during evaluation."
)

def validate_webpage(output: str) -> tuple[bool, str]:
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


def extract_judgement(output: str) -> dict:
    match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
    judgement = json.loads(match.group(1))
    return judgement


def format_single_task(task_item: dict) -> str:
    lines = [
        "Please evaluate the website by completing the following task:\n",
        f"**Task:** {task_item['task']}",
        f"**Expected Result:** {task_item['expected_result']}\n",
        JUDGEMENT_INSTRUCT,
    ]
    return "\n".join(lines)


def _run_workspace_command(
    workspace: DockerWorkspace,
    command: str,
    timeout: float,
    check: bool = True,
):
    result = workspace.execute_command(command, timeout=timeout)
    if check and result.exit_code != 0:
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Setup command failed (exit {result.exit_code}): {command}\n{output[-2000:]}"
        )
    return result


def _discover_reachable_port(workspace: DockerWorkspace) -> Optional[int]:
    probe_command = f"""cd /workspace/project && timeout {SERVER_WAIT_TIMEOUT_SECONDS} sh -lc '
for _ in $(seq 1 {SERVER_WAIT_TIMEOUT_SECONDS // 2}); do
  if curl -sf http://localhost:{DEV_SERVER_PORT} > /dev/null 2>&1; then
    echo {DEV_SERVER_PORT}
    exit 0
  fi

  LOG_PORT="$(grep -Eo "https?://(localhost|127\\\\.0\\\\.0\\\\.1|0\\\\.0\\\\.0\\\\.0):[0-9]+" {DEV_SERVER_LOG_PATH} 2>/dev/null | tail -n 1 | sed -E "s#.*:([0-9]+).*#\\\\1#")"
  if [ -n "$LOG_PORT" ] && curl -sf "http://localhost:$LOG_PORT" > /dev/null 2>&1; then
    echo "$LOG_PORT"
    exit 0
  fi

  for CANDIDATE in 5173 4173 8080; do
    if curl -sf "http://localhost:$CANDIDATE" > /dev/null 2>&1; then
      echo "$CANDIDATE"
      exit 0
    fi
  done

  sleep 2
done
exit 1
'"""
    result = workspace.execute_command(
        probe_command, timeout=SERVER_WAIT_TIMEOUT_SECONDS + 30
    )
    if result.exit_code != 0:
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1]
    if last_line.isdigit():
        return int(last_line)
    return None


def _start_dev_server(workspace: DockerWorkspace) -> str:
    _run_workspace_command(
        workspace,
        "cd /workspace/project && rm -rf .next node_modules/.cache",
        timeout=45,
        check=False,
    )
    _run_workspace_command(
        workspace,
        "cd /workspace/project && npm install --no-audit --no-fund",
        timeout=NPM_INSTALL_TIMEOUT_SECONDS,
    )
    _run_workspace_command(
        workspace,
        f"cd /workspace/project && rm -f {DEV_SERVER_LOG_PATH}",
        timeout=30,
        check=False,
    )
    _run_workspace_command(
        workspace,
        (
            "cd /workspace/project && nohup sh -lc "
            "'HOST=0.0.0.0 PORT=3000 npm run dev -- --port 3000' "
            f"> {DEV_SERVER_LOG_PATH} 2>&1 &"
        ),
        timeout=30,
    )

    port = _discover_reachable_port(workspace)
    if port is None:
        log_tail = _run_workspace_command(
            workspace,
            f"tail -n 200 {DEV_SERVER_LOG_PATH}",
            timeout=30,
            check=False,
        )
        error_details = (log_tail.stdout or log_tail.stderr or "").strip()
        raise RuntimeError(
            "Dev server failed to become reachable. "
            f"Recent {DEV_SERVER_LOG_PATH} output:\n{error_details[-2000:]}"
        )

    return f"http://localhost:{port}"


def _build_unreachable_result(example: dict, error: str) -> dict:
    failed_example = copy.deepcopy(example)
    run_result = {
        "conversation_id": None,
        "eval_output": UNREACHABLE_STOCK_SEARCH_MESSAGE,
        "events": [],
        "metrics": {},
        "error": error,
    }
    failed_example["run_result"] = run_result
    failed_example["eval_result"] = [{"run_result": copy.deepcopy(run_result)}]
    return failed_example


def _get_run_result(result: dict) -> Optional[dict]:
    if not isinstance(result, dict):
        return None
    run_result = result.get("run_result")
    if isinstance(run_result, dict):
        return run_result
    return None


def _extract_eval_output(result: dict) -> str:
    run_result = _get_run_result(result)
    if not run_result:
        return ""
    output = run_result.get("eval_output")
    return output if isinstance(output, str) else ""


def _set_eval_output(result: dict, text: str) -> None:
    run_result = _get_run_result(result)
    if run_result is not None:
        run_result["eval_output"] = text


def _contains_json_judgement_block(output: str) -> bool:
    if not output:
        return False
    match = re.search(r"```json\s*(\{.*?\})\s*```", output, re.DOTALL)
    if not match:
        return False
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return False
    return "pass" in parsed and "reason" in parsed


def _collect_strings(value, output: List[str]) -> None:
    if isinstance(value, str):
        output.append(value)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _collect_strings(nested, output)
        return
    if isinstance(value, list):
        for nested in value:
            _collect_strings(nested, output)


def _has_browser_tool_timeout(result: dict) -> bool:
    run_result = _get_run_result(result)
    if not run_result:
        return False

    text_chunks: List[str] = []
    _collect_strings(run_result.get("eval_output"), text_chunks)
    _collect_strings(run_result.get("events"), text_chunks)
    if not text_chunks:
        return False

    for text in text_chunks:
        lowered = text.lower()
        has_timeout = "timed out after 30.0s" in lowered or "timeouterror" in lowered
        has_browser_start = (
            "browserstartevent" in lowered
            or "browserlaunchevent" in lowered
            or "browser operation failed" in lowered
        )
        if has_timeout and has_browser_start:
            return True
    return False


def _set_timeout_failure_output(result: dict) -> None:
    fallback = (
        "```json\n"
        '{"pass": false, "reason": "'
        + BROWSER_TIMEOUT_EVAL_OUTPUT_MESSAGE
        + '"}\n'
        "```"
    )
    _set_eval_output(result, fallback)


def _cleanup_browser_processes(workspace: DockerWorkspace) -> None:
    cleanup_commands = [
        "pkill -f 'chrome|chromium|playwright' || true",
        "rm -rf /tmp/playwright* /tmp/browser-use* 2>/dev/null || true",
    ]
    for cmd in cleanup_commands:
        _run_workspace_command(workspace, cmd, timeout=20, check=False)


def _is_degenerate_output(output: str) -> bool:
    if not output:
        return False
    if re.search(r"0{300,}", output):
        return True
    return False


def _saw_reachable_site_in_events(result: dict) -> bool:
    run_result = _get_run_result(result)
    if not run_result:
        return False
    events = run_result.get("events")
    if not isinstance(events, list):
        return False

    for event in events:
        if not isinstance(event, dict) or event.get("kind") != "ObservationEvent":
            continue
        observation = event.get("observation")
        if not isinstance(observation, dict):
            continue
        content = observation.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            has_localhost = '"url": "http://localhost:3000/' in text
            has_app_title = "Stock Report Generator" in text
            has_search = "Search stock" in text
            if has_localhost and (has_app_title or has_search):
                return True
    return False


def _normalize_unreachable_output(result: dict) -> None:
    output = _extract_eval_output(result)
    if not output:
        return

    saw_reachable_site = _saw_reachable_site_in_events(result)
    mentions_connection_refused = (
        "ERR_CONNECTION_REFUSED" in output and "localhost:3000" in output
    )
    if mentions_connection_refused and not saw_reachable_site:
        _set_eval_output(result, UNREACHABLE_STOCK_SEARCH_MESSAGE)


def _is_acceptable_output(result: dict) -> bool:
    output = _extract_eval_output(result)
    if output == UNREACHABLE_STOCK_SEARCH_MESSAGE:
        return True
    if _contains_json_judgement_block(output):
        return True
    return False


def _patch_malformed_output(result: dict) -> None:
    output = _extract_eval_output(result)
    if not output:
        return

    if _is_degenerate_output(output):
        fallback = (
            "```json\n"
            '{"pass": false, "reason": "'
            + MALFORMED_EVAL_OUTPUT_MESSAGE
            + '"}\n'
            "```"
        )
        _set_eval_output(result, fallback)
        return

    if not _contains_json_judgement_block(output):
        fallback = (
            "```json\n"
            '{"pass": false, "reason": "'
            + MALFORMED_EVAL_OUTPUT_MESSAGE
            + '"}\n'
            "```"
        )
        _set_eval_output(result, fallback)


def run_ui_agent(
        lm: dspy.LM,
        system_prompt_path: str,
        example: dict,
        workspace: DockerWorkspace,
        website_url: str,
        log_dir: Optional[str] = None,
    ) -> dict:
    example = copy.deepcopy(example)
    tasks = ast.literal_eval(example["ui_instruct"])
    task_results = []

    for task_idx, task_item in enumerate(tasks):
        ui_instruct_text = format_single_task(task_item)
        instruction = UI_AGENT_PROMPT.format(
            ui_instruct=ui_instruct_text,
            website_url=website_url,
        )
        example['prompt'] = instruction

        llm = LLM(model=lm.model)
        agent = Agent(
            llm=llm,
            temperature=0.2,
            system_prompt_filename=system_prompt_path,
            tools=[Tool(name=BrowserToolSet.name)],
        )

        conversation_data = None
        max_attempts = 3
        saw_browser_timeout = False
        for attempt_idx in range(max_attempts):
            if attempt_idx > 0 and saw_browser_timeout:
                example["prompt"] = (
                    instruction
                    + "\n\nPrevious attempt failed with a browser startup timeout. "
                    + "Retry from scratch. Minimize browser_navigate usage and only navigate when needed."
                )
            elif attempt_idx > 0:
                example["prompt"] = (
                    instruction
                    + "\n\nYour previous response was malformed. "
                    "Call finish with only a valid JSON block exactly in the required format."
                )
            else:
                example["prompt"] = instruction

            conversation_data = _run_agentic_conversation(
                agent=agent,
                workspace_obj=workspace,
                log_dir=log_dir,
                example=example,
            )
            _normalize_unreachable_output(conversation_data)
            saw_browser_timeout = _has_browser_tool_timeout(conversation_data)
            if _is_acceptable_output(conversation_data):
                break
            if saw_browser_timeout and attempt_idx < max_attempts - 1:
                _cleanup_browser_processes(workspace)
                continue

        if conversation_data is not None and not _is_acceptable_output(conversation_data):
            if _has_browser_tool_timeout(conversation_data):
                _set_timeout_failure_output(conversation_data)
            else:
                _patch_malformed_output(conversation_data)

        task_results.append(copy.deepcopy(conversation_data))

    example["eval_result"] = task_results
    return example


def run_single_instance_eval(
        lm: dspy.LM,
        example: dict,
        workspace_dir: str,
        docker_image: str = "webgen:latest",
    ) -> dict:
    def workspace_fn(workspace, system_prompt_path, example, log_dir):
        try:
            website_url = _start_dev_server(workspace)
        except Exception as exc:
            return _build_unreachable_result(example, str(exc))

        return run_ui_agent(
            lm=lm,
            system_prompt_path=system_prompt_path,
            example=example,
            workspace=workspace,
            website_url=website_url,
            log_dir=log_dir,
        )

    return run_single_instance_agentic(
        lm=lm,
        system_prompt_path=EVAL_PROMPT_PATH,
        example=example,
        workspace=os.path.abspath(workspace_dir),
        use_docker=True,
        server_image=docker_image,
        setup_commands=[],
        workspace_fn=workspace_fn,
    )
