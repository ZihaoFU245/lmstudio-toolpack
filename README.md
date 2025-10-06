[![MCP Badge](https://lobehub.com/badge/mcp-full/zihaofu245-lmstudio-toolpack)](https://lobehub.com/mcp/zihaofu245-lmstudio-toolpack)

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/zihaofu245-lmstudio-toolpack-badge.png)](https://mseep.ai/app/zihaofu245-lmstudio-toolpack)

[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/acc116e1-414b-418b-8a94-00ab266263d8)


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
1. It is default using **stdio**, You can set it to http in `GlobalConfig`
2. In `python-sandbox.py`, `exec()` function is used to allow agent execute python scripts, keep an eye on Agents.

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
