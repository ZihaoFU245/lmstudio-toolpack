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