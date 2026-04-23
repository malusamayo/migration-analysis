# Reading Trajectory JSON Files

Trajectory JSON files are placed in `memory/current/trajectories/` as `example0.json`, `example1.json`, etc. Each file records one agent run from start to finish.

## Critical: Top-Level Structure

**The file is a JSON array (list), not a dict.** Do NOT do `json.load(f)['events']` — that raises a `TypeError`. The events are the top-level list itself.

```python
import json

events = json.load(open('memory/current/trajectories/example0.json'))
# events is a list[dict], not a dict
```

## Event Types (`kind` field)

Each item in the list has a `kind` field identifying its type:

| `kind` | `source` | Purpose |
|--------|----------|---------|
| `SystemPromptEvent` | `agent` | System prompt and tool specs sent at session start |
| `MessageEvent` | `user` | The task instruction given to the agent |
| `ActionEvent` | `agent` | Agent tool call (terminal, file editor, finish, etc.) |
| `ObservationEvent` | `environment` | Tool output returned to the agent |
| `ConversationStateUpdateEvent` | `environment` | Metadata: `execution_status` or `full_state` |

## Extracting What You Need

### Get the task prompt
```python
for e in events:
    if e.get('kind') == 'MessageEvent' and e.get('source') == 'user':
        task = e['llm_message']['content'][0]['text']
        break
```

### Get the agent's system prompt
```python
for e in events:
    if e.get('kind') == 'SystemPromptEvent':
        sys_prompt = e['system_prompt']['text']
        break
```

### Iterate actions and their outputs
```python
# Build a lookup from tool_call_id → observation text
obs_by_id = {}
for e in events:
    if e.get('kind') == 'ObservationEvent':
        tid = e.get('tool_call_id')
        content_blocks = e.get('observation', {}).get('content', [])
        obs_by_id[tid] = ''.join(b.get('text', '') for b in content_blocks)

# Walk actions in order
for e in events:
    if e.get('kind') != 'ActionEvent':
        continue
    tool = e.get('tool_name', '')
    action = e.get('action', {})
    thought = ''.join(b.get('text', '') for b in (e.get('thought') or []))
    observation = obs_by_id.get(e.get('tool_call_id'), '')
    # Use: tool, action, thought, observation
```

### Filter to specific tool calls
```python
terminal_actions = [e for e in events if e.get('kind') == 'ActionEvent' and e.get('tool_name') == 'terminal']
file_edits = [e for e in events if e.get('kind') == 'ActionEvent' and e.get('tool_name') == 'file_editor']
```

### Check if the agent finished successfully
```python
finish_events = [e for e in events if e.get('kind') == 'ActionEvent' and e.get('action', {}).get('kind') == 'FinishAction']
finished = bool(finish_events)
finish_message = finish_events[0]['action'].get('message', '') if finished else ''
```

## ActionEvent Schema

```
{
  "kind": "ActionEvent",
  "source": "agent",
  "thought": [{"type": "text", "text": "..."}],      # agent's reasoning (may be empty)
  "action": {
    "kind": "TerminalAction" | "FileEditorAction" | "FinishAction" | ...,
    # TerminalAction fields:
    "command": "ls /workspace/project",
    # FileEditorAction fields:
    "command": "view" | "str_replace" | "create" | "insert",
    "path": "/workspace/project/foo.py",
    "old_str": "...",   # for str_replace
    "new_str": "...",   # for str_replace / create
    # FinishAction fields:
    "message": "..."
  },
  "tool_name": "terminal" | "file_editor" | "finish" | ...,
  "tool_call_id": "...",   # matches ObservationEvent.tool_call_id
  "summary": "terminal: {\"command\": \"ls\"}"  # human-readable one-liner
}
```

## ObservationEvent Schema

```
{
  "kind": "ObservationEvent",
  "source": "environment",
  "tool_name": "terminal" | "file_editor" | ...,
  "tool_call_id": "...",   # matches ActionEvent.tool_call_id
  "observation": {
    "content": [{"type": "text", "text": "command output here..."}],
    "is_error": false,
    "exit_code": 0        # present for terminal observations
  }
}
```

## Practical: Summarize a trajectory in 10 lines

```python
import json

events = json.load(open('memory/current/trajectories/example0.json'))

task = next(
    (e['llm_message']['content'][0]['text'] for e in events
     if e.get('kind') == 'MessageEvent' and e.get('source') == 'user'),
    'unknown'
)
steps = [
    f"{e.get('tool_name')}: {e.get('summary', '')}"
    for e in events if e.get('kind') == 'ActionEvent'
]
finished = any(e.get('action', {}).get('kind') == 'FinishAction' for e in events)

print(f"Task: {task[:200]}")
print(f"Steps ({len(steps)}): {steps}")
print(f"Finished: {finished}")
```
