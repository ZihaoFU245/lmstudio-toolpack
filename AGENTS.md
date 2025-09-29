Add a config for cursor in `main.py`:
```
{
  "mcp": {
    "servers": {
      "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "env": {
          "MEMORY_PATH": "/tmp/agent_memory"
        }
      },
      "web-search": {
        "command": "npx", 
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env": {
          "BRAVE_API_KEY": "your-api-key"
        }
      },
      "company-api": {
        "url": "https://internal-api.company.com/mcp",
        "headers": {
          "X-API-Key": "${env:INTERNAL_API_KEY}",
          "Content-Type": "application/json"
        }
      },
      "slack": {
        "url": "https://slack-mcp-proxy.example.com/events",
        "headers": {
          "Authorization": "Bearer ${env:SLACK_BOT_TOKEN}"
        }
      }
    }
  }
}
```
Cursor looks like this. And in selection we support http

1. In tool selection (toggle state),  add a toggle to choose http or stdio. ctrl + H, to enable http for all.
2. Add a build config for cursor

In your last edits:

you made:
```json
"python-sandbox": {
        "command": "D:\\MyProject\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
        "args": [
          "D:\\MyProject\\lmstudio-toolpack\\MCPs\\python-sandbox.py"
        ],
        "env": {
          "MCP_TRANSPORT": "http"
        }
```
You edit added a `"MCP_TRANSPORT": "http"`, but this is wrong.

It should be:
```json
      "company-api": {
        "url": "https://internal-api.company.com/mcp",
        "headers": {
          "X-API-Key": "${env:INTERNAL_API_KEY}",
          "Content-Type": "application/json"
        }
```
like this. For http transport.
And for stdio transport:
```json
    "python-sandbox": {
      "command": "D:\\MyProject\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
      "args": [
        "D:\\MyProject\\lmstudio-toolpack\\MCPs\\python-sandbox.py"
      ],
      "env": {}
    },
```
is this.

Leave `"headers"` and `"env"` empty. 

Second thing:
Toggle http should be per-tool toggle, not toggle all. And ctrl + H is not force toggle. It is toggle all of them to use http.

Example: 
"""
3 tools: A, B, C
default to all stdio, they are initially: A (stdio), B (stdio), C (stdio)
Select B and press H: A (stdio), B (http), C (stdio)
Press ctrl + H: A (http), B (http), C (http)
Press ctrl + H again: A (stdio), B (stdio), C (stdio)

After this, the generated json file, if a tool is http, then
```json
"Tool name (A / B / C)": {
  "url": "https://internal-api.company.com/mcp",
  "headers": {}
}
```
if it is stdio, then
```json
"Tool name (A / B / C)": {
  "command": "D:\\MyProject\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
  "args": [
  "D:\\MyProject\\lmstudio-toolpack\\MCPs\\python-sandbox.py"
  ],
  "env": {}
}
```
"""

I have rolled back your last edits. try again.