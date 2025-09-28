#folder:MCPs There are MCP servers in this folder.
This main.py is for generating json configuation files
Requirements:
1. A CLI Tool
2. Support different types, now only VScode and Lm-Studio.
3. User can select which mcp server to add. In settings, user can choose to default select all or not select anything.
4. Any configuration file for this CLI tool will be put in /data folder, already exists
5. It is an interactive tool, do not allow `main.py --flag` use
Appendix:
For VScode, the json cionfig looks like:
```
{
    "servers": {
        "Web Search": {
            "type": "stdio",
            "command": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
            "args": [
                "E:\\LMStudio\\mcp\\lmstudio-toolpack\\WebSearch.py"
            ]
        }
    }
}
```
And for LM-Studio is:
```
{
  "mcpServers": {
    "web-search": {
      "command": "E:\\LMStudio\\mcp\\lmstudio-toolpack\\.venv\\Scripts\\python.exe",
      "args": [
        "E:\\LMStudio\\mcp\\lmstudio-toolpack\\WebSearch.py"
      ],
    }
  }
```
As I stated, finish the CLI tool