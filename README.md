# Local MCP Tools Collection

> A small collection of Model Context Protocol (MCP) tools, build for local LLMs. One venv, many options.

## Features
- [Web Search](WebSearch.py): Use duckduckgo as search engine, provide agent ability to search
- [Python SandBox](python-sandbox.py): Allow Agents to run python, use numpy and sympy, good for math
- [Longterm-Memory](Memory.py): For Agents to memories things for longterm use.

## Requirements
- Python >= 3.13
- Managed with `uv`

## Install
Using uv:
```bash
uv sync
```

## Run the MCP Server
```powershell
python python-sandbox.py
```
The server communicates over stdio (FastMCP). Point your MCP-compatible client at the executable command above.

## Tool Usage Examples

In LM studio mcp.json:
```json
{
  "mcpServers": {
    "server-name": {
      "command": "path\\to\\venv\\python",
      "args": [
        "path\\to\\tool\\file"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
}
```


