
## Troubleshooting: Unified MCP HTTP Gateway

This repository contains a unified MCP gateway (`run_http.py`) that discovers and proxies local MCP servers in `MCPs/` via stdio and exposes them over Streamable HTTP at `/mcp`.

### Symptom: Health shows connected=false and empty tools for all upstreams

Example `/health` response:

```
{
  "status": "ok",
  "initialized": true,
  "upstreams": [
    {"name":"memory","transport":"stdio","tools":[],"connected":false,"error":null},
    {"name":"python-sandbox","transport":"stdio","tools":[],"connected":false,"error":null},
    {"name":"websearch","transport":"stdio","tools":[],"connected":false,"error":null}
  ]
}
```

This indicates the gateway started, but failed to establish the MCP stdio handshake with child processes, so `list_tools()` never populated. When this happens, host clients (LM Studio, VS Code, etc.) cannot see any tools and may report SSE 500s or “Server exited before responding to initialize”.

### Likely Causes

- Python stdio buffering delayed/blocked the MCP handshake from child processes.
- Child process environment/working directory prevented imports (e.g., `GlobalConfig`) or output encoding.
- Host did not run Starlette startup hooks, so upstream connections were never attempted.

### Relevant Logs (as reported during failures)

- LM Studio MCP bridge:
```
SSE error: Non-200 status code (500) code: 500 event.message: 'Non-200 status code (500)'
```

- VS Code MCP extension:
```
Server exited before responding to `initialize` request.
Error sending message to http://0.0.0.0/8000/mcp: TypeError: fetch failed
```

- Uvicorn server:
```
INFO: 127.0.0.1 - "GET /mcp HTTP/1.1" 500 Internal Server Error
RuntimeError: Task group is not initialized. Make sure to use run().
```


