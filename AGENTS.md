# TASK1:
> Support HTTP transport in `main.py`

## Http server format:
```json
{
  "mcpServers": {
    "http example" : {
      "url" : "https://url-to-mcp-server",
      "headers": {}
    }
  }
}
```

## Requirements

1. In the server selection stage: User can choose to configure it as http transport or stdio transport
  Currently, there are 2 key supports, 'A' and 'Ctrl + A'. Toggle to select.
  Use 'H' to select configuration weather stdio or http. 
  Use 'Ctrl + H' to mark all servers use http transport
  > Note: Http and stdio formats are different.
2. `main.py` is for json config file generation. Does not effect by `GlobalConfig.py`
  If in `GlobalConfig.py`, the `transport` is set to stdio and any of the server is target to generate to `http`,
  throw a warning at the end. After line `_write_config(output_path, content)` in `main.py :: main()` function. 
  Use purple color

# Task2:

> Support cursor style json configurtion generation.

## Requirements:
1. Add cursor support

Cursor json configuration looks like this:
```json
{
  "mcp": {
    "servers": {
      "Example-stdio" : {
        "command": "npx or path/to/python/",
        "args": [
          "path/to/MCP/server/.py"
        ]
      },
      "Example-http": {
        "url": "https://example.com/mcp",
        "headers": {
        }
      }
    }
  }
}
```
# Notes:

1. Leave `"headers"` and `"env"` empty, for `http` and `stdio` respectively. 

2. Toggle http should be per-tool toggle, which means stdio and http can be mixed.

3. **Example:**
3 tools: A, B, C
default to all stdio, they are initially: A (stdio), B (stdio), C (stdio)
Select B and press H: A (stdio), B (http), C (stdio)
Press ctrl + H: A (http), B (http), C (http)
Press ctrl + H again: A (stdio), B (stdio), C (stdio)
Select A and press H: A (http), B (stdio), C (stdio)
After this, the generated json file, if a tool is http, then
```json
{
  "mcpServers": {
    "A" : {
      "url" : "https://url-to-mcp-server",
      "headers": {}
    },
    "B" : {
      "command": "path/to/python",
      "args": [
        "path/to/mcp/server"
      ],
      "env": {}
    },
    "C" : {
      "command": "path/to/python",
      "args": [
        "path/to/mcp/server"
      ],
      "env": {}
    }
  }
}
```

Refactor code as you like, only touch `main.py`