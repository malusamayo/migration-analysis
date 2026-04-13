# 6. Skill

**Import:** `from openhands.sdk.context import Skill, KeywordTrigger, TaskTrigger`

Skills provide reusable knowledge or instructions that can be injected into prompts based on triggers.

## Signature

```python
class Skill(BaseModel):
    name: str
    content: str
    trigger: KeywordTrigger | TaskTrigger | None = None
    source: str | None = None
    description: str | None = None
    mcp_tools: dict | None = None
    is_agentskills_format: bool = False
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: list[str] | None = None
    resources: SkillResources | None = None
```

## Trigger Types

```python
from openhands.sdk.context import KeywordTrigger, TaskTrigger

KeywordTrigger(keywords=["docker", "container", "deploy"])
TaskTrigger(triggers=["code_review", "refactor"])
```

## Skill Behavior

- `trigger=None` means the skill is always included.
- `KeywordTrigger(...)` exposes the skill and injects it when keywords match.
- Skills are deduplicated by name per conversation turn.
