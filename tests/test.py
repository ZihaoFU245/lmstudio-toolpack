import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import MCPs.WebSearch as ws

import asyncio

res = asyncio.run(ws.mcp.get_tools())
print(res)