# Local MCP Tools Collection

> A small collection of Model Context Protocol (MCP) tools, build for local LLMs. One venv, many options.

## Why is exists?
The MCP server now is mostly scattered. There is no simple tool-pack. We need to set it up per-tool.
This tool pack is targeted for local convenient use. I will expand the collections through time. 
Make LocalLLMs more powerful yet simplier.

## Features
- MCP json Configuration file generation: Run `main.py` and go through the wizard to complete the generation
- One venv for multiple MCP servers

## MCP Servers
- [Web Search](/MCPs/WebSearch.py): Use duckduckgo as search engine, fetch and summarize top results
- [Python SandBox](/MCPs/python-sandbox.py): Allow Agents to run python, use numpy and sympy, good for math
- [Longterm-Memory](/MCPs/Memory.py): For Agents to memories things for longterm use.

## Notes
It is default using **stdio**
You can set it to http in `GlobalConfig`

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
Run `main.py` for json configuration auto generation.
And you will get something like this:
```json
{
  "mcpServers": {
    "memory": {
      "command": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
      "args": [
        "E:\\LMStudio\\mcp\\lmstudio-toolpack\\MCPs\\Memory.py"
      ]
    },
    "python-sandbox": {
      "command": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
      "args": [
        "E:\\LMStudio\\mcp\\lmstudio-toolpack\\MCPs\\python-sandbox.py"
      ]
    },
    "websearch": {
      "command": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
      "args": [
        "E:\\LMStudio\\mcp\\lmstudio-toolpack\\MCPs\\WebSearch.py"
      ]
    }
  }
}
```
Change the name if you needed

## Another Idea
If you choose using http. You can use 1mcp to unify them all.
And run it on a remote server.
Eg. Connect a Resberry PI to TailScale and set it up remotely.
