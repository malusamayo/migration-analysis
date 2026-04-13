# 1. LLM

**Import:** `from openhands.sdk import LLM` or `from openhands.sdk.llm import LLM`

The `LLM` class wraps LiteLLM so the SDK can talk to different providers through one interface.

## Constructor Signature

```python
class LLM(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    api_key: str | SecretStr | None = None
    base_url: str | None = None
    api_version: str | None = None
    usage_id: str = "default"
    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_message_chars: int = 30_000
    num_retries: int = 5
    retry_multiplier: float = 8.0
    retry_min_wait: int = 8
    retry_max_wait: int = 64
    timeout: int | None = 300
    stream: bool = False
    caching_prompt: bool = True
    native_tool_calling: bool = True
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "none"] | None = "high"
    extended_thinking_budget: int | None = 200_000
    seed: int | None = None
    disable_vision: bool | None = None
    model_canonical_name: str | None = None
    extra_headers: dict[str, str] | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    log_completions: bool = False
    custom_tokenizer: str | None = None
    litellm_extra_body: dict[str, Any] = {}
    fallback_strategy: FallbackStrategy | None = None
    drop_params: bool = True
    aws_access_key_id: str | SecretStr | None = None
    aws_secret_access_key: str | SecretStr | None = None
    aws_region_name: str | None = None
```

## Key Properties

- `llm.metrics` exposes usage and cost metrics.
- `llm.model_copy(update={...})` creates a modified copy of the model config.

## Direct Completion

```python
from openhands.sdk import LLM, Message, TextContent

llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key="...")
resp = llm.completion(
    messages=[Message(role="user", content=[TextContent(text="Hello")])]
)
texts = [c.text for c in resp.message.content if isinstance(c, TextContent)]
```

## Example

```python
import os
from openhands.sdk import LLM

llm = LLM(
    model=os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", None),
)
```
