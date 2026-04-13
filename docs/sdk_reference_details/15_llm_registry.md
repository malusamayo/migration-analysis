# 15. LLM Registry

**Import:** `from openhands.sdk import LLMRegistry`

`LLMRegistry` stores and retrieves `LLM` instances by usage ID.

```python
from openhands.sdk import LLM, LLMRegistry

registry = LLMRegistry()
registry.add(LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key="...", usage_id="agent"))
registry.add(LLM(model="anthropic/claude-haiku-4-5-20251001", api_key="...", usage_id="cheap"))

agent_llm = registry.get("agent")
cheap_llm = registry.get("cheap")

print(registry.list_usage_ids())
```
