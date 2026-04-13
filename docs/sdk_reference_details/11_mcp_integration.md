# 11. MCP Integration

MCP tools can be added through `Agent.mcp_config`.

```python
mcp_config = {
    "mcpServers": {
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "repomix": {"command": "npx", "args": ["-y", "repomix@1.4.2", "--mcp"]},
    }
}

agent = Agent(
    llm=llm,
    tools=[Tool(name=TerminalTool.name)],
    mcp_config=mcp_config,
    filter_tools_regex="^(?!repomix)(.*)|^repomix.*pack_codebase.*$",
)
```
