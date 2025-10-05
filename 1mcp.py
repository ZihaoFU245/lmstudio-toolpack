"""Simple HTTP multiplexer for local MCP servers.

This script discovers all MCP Python scripts inside the ``MCPs`` directory,
launches them via stdio, and exposes every tool they provide through a single
HTTP endpoint.  Each upstream tool becomes available under a namespaced name of
``<server>.<tool>`` (for example ``memory.retrieve``).

Run it with::

    python 1mcp.py --port 3333

The default port falls back to ``GlobalConfig.port`` or ``3333`` if unset.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastmcp import FastMCP
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from GlobalConfig import GlobalConfig

BASE_DIR = Path(__file__).resolve().parent
MCP_DIR = BASE_DIR / "MCPs"
LOGGER = logging.getLogger("1mcp")


@dataclass
class Upstream:
    """Represents a locally launched MCP server."""

    name: str
    script: Path
    session: Optional[ClientSession] = None
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)


UPSTREAMS: List[Upstream] = []
UPSTREAM_MAP: Dict[str, Upstream] = {}
EXIT_STACK: AsyncExitStack | None = None
INITIALIZED: bool = False
INIT_LOCK: asyncio.Lock | None = None
PROXIES_REGISTERED: bool = False
CONFIG: GlobalConfig | None = None

mcp = FastMCP(
    name="lmstudio-1mcp",
    instructions=(
        "A simple HTTP bridge for local MCP servers. "
        "Call tools using '<server>.<tool>' or use invoke(server, tool, args)."
    ),
)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or value


def discover_local_servers(skip: Iterable[str] = ()) -> List[Upstream]:
    """Return a sorted list of MCP Python scripts inside ``MCPs``."""

    directory = MCP_DIR
    if not directory.exists():
        LOGGER.warning("MCP directory '%s' does not exist", directory)
        return []

    skipped = {_slugify(item) for item in skip if item}
    seen: set[str] = set()
    servers: List[Upstream] = []
    for script in sorted(directory.glob("*.py")):
        slug = _slugify(script.stem)
        if slug in skipped:
            LOGGER.info("Skipping MCP script '%s' per configuration", script.name)
            continue
        if slug in seen:
            LOGGER.warning("Duplicate MCP slug '%s' detected, skipping '%s'", slug, script)
            continue
        seen.add(slug)
        servers.append(Upstream(name=slug, script=script))
    return servers


async def _connect_stdio(upstream: Upstream, stack: AsyncExitStack) -> ClientSession:
    params = StdioServerParameters(command=sys.executable, args=[str(upstream.script)])
    read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
    session = ClientSession(read_stream, write_stream)
    session = await stack.enter_async_context(session)
    init = await session.initialize()
    LOGGER.info(
        "[%s] connected to %s %s",
        upstream.name,
        init.serverInfo.name,
        init.serverInfo.version,
    )

    listed = await session.list_tools()
    tools: Dict[str, Dict[str, Any]] = {}
    for tool in listed.tools:
        tools[tool.name] = {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "inputSchema": getattr(tool, "inputSchema", None),
        }
    upstream.session = session
    upstream.tools = tools
    return session


async def _ensure_lock() -> asyncio.Lock:
    global INIT_LOCK
    if INIT_LOCK is None:
        INIT_LOCK = asyncio.Lock()
    return INIT_LOCK


async def ensure_initialized() -> None:
    """Launch every upstream MCP server and register proxy tools once."""

    global EXIT_STACK, INITIALIZED, PROXIES_REGISTERED
    if INITIALIZED:
        return

    lock = await _ensure_lock()
    async with lock:
        if INITIALIZED:
            return
        if CONFIG is None:
            setup()
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            for upstream in UPSTREAMS:
                try:
                    await _connect_stdio(upstream, stack)
                except Exception:  # pragma: no cover - defensive logging
                    LOGGER.exception("Failed to connect to upstream '%s'", upstream.name)
                    raise
            await _register_proxy_tools()
        except Exception:
            await stack.aclose()
            raise
        EXIT_STACK = stack
        INITIALIZED = True
        PROXIES_REGISTERED = True
        LOGGER.info("All upstream MCP servers connected: %s", ", ".join(u.name for u in UPSTREAMS))


def _serialize_input_schema(schema: Any) -> Any:
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump_json"):
        return json.loads(schema.model_dump_json())
    if hasattr(schema, "model_dump"):
        return schema.model_dump()
    if hasattr(schema, "dict"):
        return schema.dict()
    return schema


def _make_proxy(upstream: Upstream, tool_name: str):
    async def proxy(**kwargs: Any) -> Any:  # type: ignore[override]
        session = upstream.session
        if session is None:
            raise RuntimeError(f"Upstream '{upstream.name}' is not connected")
        return await session.call_tool(tool_name, kwargs or {})

    proxy.__name__ = f"{upstream.name}_{tool_name}".replace(".", "_")
    return proxy


def _register_proxy_tools() -> None:
    global PROXIES_REGISTERED
    if PROXIES_REGISTERED:
        return

    for upstream in UPSTREAMS:
        if not upstream.tools:
            continue
        for tool_name, meta in upstream.tools.items():
            proxy = _make_proxy(upstream, tool_name)
            proxy.__doc__ = meta.get("description") or (
                f"Proxy for tool '{tool_name}' on upstream '{upstream.name}'."
            )
            mcp.tool(name=f"{upstream.name}.{tool_name}", description=meta.get("description", ""))(proxy)

    PROXIES_REGISTERED = True


def _format_tool_summary(upstream: Upstream) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for tool_name, meta in sorted(upstream.tools.items()):
        summary.append(
            {
                "name": tool_name,
                "description": meta.get("description", ""),
                "inputSchema": _serialize_input_schema(meta.get("inputSchema")),
            }
        )
    return summary


@mcp.tool(description="Call a tool hosted by a specific upstream server.")
async def invoke(server: str, tool: str, args: Optional[Dict[str, Any]] = None) -> Any:
    await ensure_initialized()
    upstream = UPSTREAM_MAP.get(server)
    if upstream is None:
        raise ValueError(f"Unknown upstream '{server}'")
    session = upstream.session
    if session is None:
        raise RuntimeError(f"Upstream '{server}' is not connected")
    return await session.call_tool(tool, args or {})


@mcp.tool(description="List every MCP server discovered under the MCPs directory.")
async def list_servers() -> Dict[str, Any]:
    await ensure_initialized()
    return {
        "servers": [
            {
                "name": upstream.name,
                "script": str(upstream.script),
                "tools": _format_tool_summary(upstream),
            }
            for upstream in UPSTREAMS
        ]
    }


@mcp.tool(description="Describe how to call each namespaced tool exposed by 1mcp.")
async def list_tools() -> Dict[str, Any]:
    await ensure_initialized()
    results: List[Dict[str, Any]] = []
    for upstream in UPSTREAMS:
        for tool_name in sorted(upstream.tools):
            results.append(
                {
                    "name": f"{upstream.name}.{tool_name}",
                    "server": upstream.name,
                    "tool": tool_name,
                }
            )
    return {"tools": results}


async def shutdown() -> None:
    global EXIT_STACK, INITIALIZED, PROXIES_REGISTERED
    if EXIT_STACK is not None:
        await EXIT_STACK.aclose()
        EXIT_STACK = None
    INITIALIZED = False
    PROXIES_REGISTERED = False
    for upstream in UPSTREAMS:
        upstream.session = None


async def main_async(port: Optional[int]) -> None:
    await ensure_initialized()
    config_port = CONFIG.port if CONFIG else None
    port_to_use = port or config_port or 3333
    LOGGER.info("Starting HTTP MCP server on port %s", port_to_use)
    try:
        await mcp.run_http_async(port_to_use)
    finally:
        await shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the simple 1mcp HTTP bridge.")
    parser.add_argument("--port", type=int, help="Port to expose the HTTP MCP endpoint on.")
    return parser.parse_args()


def setup() -> None:
    global UPSTREAMS, UPSTREAM_MAP, CONFIG
    CONFIG = GlobalConfig()
    skip = getattr(CONFIG, "http_proxy_skip", ())
    UPSTREAMS = discover_local_servers(skip=skip)
    UPSTREAM_MAP = {upstream.name: upstream for upstream in UPSTREAMS}
    if not UPSTREAMS:
        LOGGER.warning("No MCP servers found in %s", MCP_DIR)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    setup()
    args = parse_args()
    try:
        asyncio.run(main_async(args.port))
    except KeyboardInterrupt:
        LOGGER.info("Shutting down 1mcp")


if __name__ == "__main__":
    main()
