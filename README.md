[![MCP Badge](https://lobehub.com/badge/mcp-full/zihaofu245-lmstudio-toolpack)](https://lobehub.com/mcp/zihaofu245-lmstudio-toolpack)

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/zihaofu245-lmstudio-toolpack-badge.png)](https://mseep.ai/app/zihaofu245-lmstudio-toolpack)

[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/acc116e1-414b-418b-8a94-00ab266263d8)


# Local MCP Tools Collection

> A small collection of Model Context Protocol (MCP) tools, built for local LLMs. One venv, many options.

## Why it exists
Many MCP servers are distributed as separate projects and need separate setup.
This tool pack keeps a few local MCP servers in one repo and one `uv` environment.

## Features
- MCP JSON configuration generation: run `main.py` and go through the wizard
- One venv for multiple MCP servers

## MCP Servers
- [Web Search](/MCPs/WebSearch.py): Use duckduckgo as search engine, fetch and summarize top results
- [Python SandBox](/MCPs/python-sandbox.py): Allow agents to run Python and use NumPy and SymPy for math tasks
- [Longterm-Memory](/MCPs/Memory.py): Store lightweight long-term notes

## Security Notes
1. Default transport is `stdio`. You can switch to HTTP in `GlobalConfig`.
2. `python-sandbox.py` uses `exec()` and `eval()` for `run_python`. This is not a secure sandbox.
3. Treat `run_python` as local code execution by the agent under your user account.
4. The generated config now sets `PYTHON_SANDBOX_LOG_PATH` for local `python-sandbox` entries so every `run_python` execution is appended to a JSONL audit log.

## Requirements
- Python >= 3.13
- Managed with `uv`

## Install
Using uv:
```bash
uv sync
```

## Generate LM Studio Config
Run:
```bash
uv run python main.py
```

LM Studio currently follows Cursor-style `mcp.json` notation. The generated LM Studio output uses the `mcpServers` object with either:
- `command` + `args` + `env` for local stdio servers
- `url` + `headers` for remote HTTP servers

If you select the local `python-sandbox` server, the wizard will ask for an audit log file path.

## Run the MCP Server
```powershell
python python-sandbox.py
```
The server communicates over stdio (FastMCP). Point your MCP-compatible client at the executable command above.

## Tool Usage Examples
Run `main.py` for JSON configuration generation.
For LM Studio, you will get something like this:
```json
{
  "mcpServers": {
    "memory": {
      "command": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
      "args": [
        "E:\\LMStudio\\mcp\\lmstudio-toolpack\\MCPs\\Memory.py"
      ],
      "env": {}
    },
    "python-sandbox": {
      "command": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
      "args": [
        "E:\\LMStudio\\mcp\\lmstudio-toolpack\\MCPs\\python-sandbox.py"
      ],
      "env": {
        "PYTHON_SANDBOX_LOG_PATH": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\data\\python-sandbox-audit.jsonl"
      }
    },
    "websearch": {
      "command": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
      "args": [
        "E:\\LMStudio\\mcp\\lmstudio-toolpack\\MCPs\\WebSearch.py"
      ],
      "env": {}
    }
  }
}
```
Change the server names if needed.

## Another Idea
If you choose HTTP, you can run a remote MCP deployment instead of local stdio servers.
