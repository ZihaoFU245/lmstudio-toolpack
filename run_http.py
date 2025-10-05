"""
Unified MCP gateway for lmstudio-toolpack
----------------------------------------

What it does
- Runs a single MCP server over Streamable HTTP (/mcp)
- Connects to multiple upstream MCP servers (stdio, http/streamable, http+sse)
- Discovers local MCP scripts in the MCPs/ folder automatically
- Exposes each upstream tool under names like:  <server>.<tool>
- Provides a generic invoke(server, tool, args) fallback
- Adds a /health route and optional Bearer auth hook

Run (dev):
    uvicorn run_http:app --host 0.0.0.0 --port 8000

Clients connect to:
    http://localhost:8000/mcp

Config via env (comma-separated):
    UPSTREAMS='fs:stdio:python,-m,mcp_server_filesystem;github:http:https://mcp.example.com/mcp'
    or, for SSE:
    UPSTREAMS='zapier:sse:https://zapier-mcp.example.com/sse'
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Sequence

from GlobalConfig import GlobalConfig
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# --- FastMCP server (official Python SDK) ---
from mcp.server.fastmcp import FastMCP
from fastmcp.server.auth import StaticTokenVerifier

# --- MCP client bits (official SDK) ---
# The SDK ships client transports for stdio + HTTP/SSE (names can vary slightly across versions).
# We import lazily with fallbacks to keep compatibility across releases.

def _import(path: str):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)

try:
    ClientSession = _import("mcp.client.session.ClientSession")
except Exception:
    ClientSession = _import("mcp.client.session.ClientSession")

try:
    StdioServerParameters = _import("mcp.client.stdio.StdioServerParameters")
    stdio_client = _import("mcp.client.stdio.stdio_client")
except Exception as exc:  # pragma: no cover - SDK compatibility shim
    raise RuntimeError("Stdio client transport is not available in this MCP SDK.") from exc

try:
    streamablehttp_client = _import("mcp.client.streamable_http.streamablehttp_client")
except Exception:
    streamablehttp_client = None

try:
    sse_client = _import("mcp.client.sse.sse_client")
except Exception:
    sse_client = None


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
MCP_DIR = BASE_DIR / "MCPs"
CONFIG = GlobalConfig()


# ------------------------
# Config & Data Structures
# ------------------------

@dataclass
class Upstream:
    name: str  # namespace prefix e.g. "fs", "github"
    transport: str  # "stdio" | "http" | "sse"
    command_or_url: List[str]  # stdio: argv ; http/sse: [url]
    headers: Optional[Dict[str, str]] = None
    session: Optional[Any] = None  # ClientSession
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    get_session_id: Optional[Callable[[], Optional[str]]] = None
    last_error: Optional[str] = None

def _slugify(value: str) -> str:
    value = value.strip()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value)
    cleaned = re.sub(r"-+", "-", cleaned)
    cleaned = cleaned.strip("-")
    return cleaned.lower() or value.lower()


def _normalize_skip(skip: Sequence[str]) -> set[str]:
    return {_slugify(item) for item in skip if isinstance(item, str) and item.strip()}


def _resolve_python_command() -> str:
    return sys.executable


def discover_local_upstreams(python_command: str, skip: Sequence[str]) -> List[Upstream]:
    upstreams: List[Upstream] = []
    skipped = _normalize_skip(skip)
    if not MCP_DIR.exists():
        return upstreams

    for script in sorted(MCP_DIR.glob("*.py")):
        slug = _slugify(script.stem)
        if slug in skipped:
            LOGGER.info("Skipping MCP script '%s' per GlobalConfig.http_proxy_skip", script.stem)
            continue
        upstreams.append(
            Upstream(
                name=slug,
                transport="stdio",
                command_or_url=[python_command, str(script)],
            )
        )
    return upstreams

def parse_upstreams(env_value: str) -> List[Upstream]:
    """
    Parse UPSTREAMS env like:
        fs:stdio:python,-m,mcp_server_filesystem;github:http:https://host/mcp;zapier:sse:https://host/sse
    """
    upstreams: List[Upstream] = []
    if not env_value:
        return upstreams
    for chunk in env_value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            name, transport, rest = chunk.split(":", 2)
        except ValueError:
            LOGGER.warning("Ignoring malformed upstream definition: %s", chunk)
            continue
        if transport == "stdio":
            argv = [s for s in rest.split(",") if s]
            if not argv:
                LOGGER.warning("Ignoring stdio upstream '%s' with empty command", name)
                continue
            upstreams.append(Upstream(name=name, transport="stdio", command_or_url=argv))
        else:
            url = rest.strip()
            if not url:
                LOGGER.warning("Ignoring %s upstream '%s' without URL", transport, name)
                continue
            upstreams.append(Upstream(name=name, transport=transport, command_or_url=[url]))
    return upstreams

def _load_upstreams() -> List[Upstream]:
    python_command = _resolve_python_command()
    configured = parse_upstreams(os.getenv("UPSTREAMS", ""))
    seen = {up.name for up in configured}

    local_upstreams = discover_local_upstreams(python_command, CONFIG.http_proxy_skip)
    for upstream in local_upstreams:
        if upstream.name in seen:
            LOGGER.warning(
                "Skipping local MCP '%s' because an upstream with the same name was provided via UPSTREAMS.",
                upstream.name,
            )
            continue
        seen.add(upstream.name)
        configured.append(upstream)

    if not configured:
        LOGGER.warning(
            "No upstream MCP servers detected. Add MCP scripts under %s or set the UPSTREAMS environment variable.",
            MCP_DIR,
        )
    return configured


UPSTREAMS: List[Upstream] = _load_upstreams()
UPSTREAM_EXIT_STACK: AsyncExitStack | None = None
_INIT_LOCK: asyncio.Lock | None = None
_INITIALIZED: bool = False

# Optional: Bearer token the client must present (if set)
REQUIRE_BEARER = os.getenv("MCP_BEARER", "").strip() or None
if REQUIRE_BEARER:
    LOGGER.info("Bearer authentication enabled via MCP_BEARER environment variable.")
    AUTH_PROVIDER = StaticTokenVerifier(tokens={
        REQUIRE_BEARER: {"client_id": "lmstudio-toolpack", "scopes": []},
    })
else:
    AUTH_PROVIDER = None


# --------------------
# Upstream connections
# --------------------

def _ensure_url(upstream: Upstream) -> str:
    url = upstream.command_or_url[0] if upstream.command_or_url else ""
    if not url:
        raise ValueError(f"Upstream '{upstream.name}' is missing a URL for {upstream.transport} transport")
    return url


async def _connect_stdio(upstream: Upstream, stack: AsyncExitStack) -> tuple[Any, Any]:
    argv = upstream.command_or_url
    if not argv:
        raise ValueError(f"Upstream '{upstream.name}' is missing a command for stdio transport")
    command, *args = argv
    # Ensure unbuffered stdio for Python-based servers so the MCP handshake isn't delayed
    try:
        base = Path(command).name.lower()
    except Exception:
        base = command.lower()
    if (base.startswith("python") or command == sys.executable) and "-u" not in args:
        args = ["-u", *args]
    LOGGER.info("[upstream:%s] spawn stdio: %s %s", upstream.name, command, " ".join(args))
    # Ensure environment and working directory are friendly to our MCP child
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Make sure repository root is importable for GlobalConfig
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{BASE_DIR}{os.pathsep}{py_path}" if py_path else str(BASE_DIR)
    try:
        params = StdioServerParameters(command=command, args=args, env=env, cwd=str(BASE_DIR))
    except TypeError:
        # Older SDK without env/cwd support
        params = StdioServerParameters(command=command, args=args)
    return await stack.enter_async_context(stdio_client(params))


async def _connect_http(upstream: Upstream, stack: AsyncExitStack) -> tuple[Any, Any, Callable[[], Optional[str]]]:
    if streamablehttp_client is None:
        raise RuntimeError("Streamable HTTP client transport not available in this MCP SDK version.")
    url = _ensure_url(upstream)
    return await stack.enter_async_context(
        streamablehttp_client(url, headers=upstream.headers or {})
    )


async def _connect_sse(upstream: Upstream, stack: AsyncExitStack) -> tuple[Any, Any]:
    if sse_client is None:
        raise RuntimeError("SSE client transport not available in this MCP SDK version.")
    url = _ensure_url(upstream)
    return await stack.enter_async_context(
        sse_client(url, headers=upstream.headers or {})
    )


async def connect_upstream(u: Upstream, stack: AsyncExitStack) -> None:
    """Launch/connect and cache tool metadata for an upstream server."""
    LOGGER.info("[upstream:%s] connecting via %s", u.name, u.transport)

    if u.transport == "stdio":
        read_stream, write_stream = await _connect_stdio(u, stack)
    elif u.transport == "http":
        read_stream, write_stream, get_session_id = await _connect_http(u, stack)
        u.get_session_id = get_session_id
    elif u.transport == "sse":
        read_stream, write_stream = await _connect_sse(u, stack)
    else:
        raise ValueError(f"Unknown transport: {u.transport}")

    session = ClientSession(read_stream, write_stream)
    session = await stack.enter_async_context(session)
    init = await session.initialize()
    LOGGER.info("[upstream:%s] connected to %s %s", u.name, init.serverInfo.name, init.serverInfo.version)

    listed = await session.list_tools()
    tools: Dict[str, Dict[str, Any]] = {}
    for tool in listed.tools:
        tools[tool.name] = {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "inputSchema": getattr(tool, "inputSchema", None),
        }
    u.session = session
    u.tools = tools
    u.last_error = None
    LOGGER.info("[upstream:%s] tools: %s", u.name, list(tools.keys()))


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


def _build_args_template(schema: Any) -> dict[str, Any]:
    serialized = _serialize_input_schema(schema)
    if not isinstance(serialized, dict):
        return {}
    props = serialized.get("properties")
    if not isinstance(props, dict):
        return {}
    template: dict[str, Any] = {}
    required = set(serialized.get("required", []) or [])
    for name in props:
        template[name] = "<required>" if name in required else None
    return template


async def setup_all() -> List[Upstream]:
    """Connect to all configured upstreams."""
    global UPSTREAM_EXIT_STACK
    stack = AsyncExitStack()
    await stack.__aenter__()
    try:
        # Best-effort connection: do not fail the whole server if one upstream
        # cannot be reached. This ensures discovery tools like get_registry
        # still work and report disconnected upstreams instead of raising.
        for u in UPSTREAMS:
            try:
                await connect_upstream(u, stack)
            except Exception as exc:
                LOGGER.error("[upstream:%s] failed to connect: %s", u.name, exc)
                u.last_error = str(exc)
                # Leave this upstream marked as disconnected; continue.
    except Exception:
        await stack.aclose()
        raise
    UPSTREAM_EXIT_STACK = stack
    return UPSTREAMS


async def ensure_initialized() -> None:
    """Ensure upstreams are connected and proxy tools registered.

    Some hosts may not trigger Starlette 'startup' events (or may race them).
    This guard initializes lazily on first use to guarantee readiness.
    """
    global _INIT_LOCK, _INITIALIZED
    if _INITIALIZED:
        return
    if _INIT_LOCK is None:
        _INIT_LOCK = asyncio.Lock()
    async with _INIT_LOCK:
        if _INITIALIZED:
            return
        await setup_all()
        await register_proxies()
        _INITIALIZED = True


# ----------------------------
# Build the unified MCP server
# ----------------------------

mcp = FastMCP(
    name="lmstudio-toolpack-unified",
    instructions=(
        "Unified gateway for multiple MCP servers. "
        "Call namespaced tools like 'server.tool'. Use 'invoke' for dynamic calls."
    ),
    auth=AUTH_PROVIDER,
)
def _wrap_fastmcp_list_tools() -> None:
    if getattr(mcp, "_lmstudio_list_tools_wrapped", False):
        return

    original_method = mcp.list_tools

    async def list_tools_with_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        await ensure_initialized()
        return await original_method(*args, **kwargs)

    mcp.list_tools = MethodType(list_tools_with_init, mcp)
    setattr(mcp, "_lmstudio_list_tools_wrapped", True)


_wrap_fastmcp_list_tools()



# --- Generic fallback: invoke(server, tool, args) ---
@mcp.tool()
async def invoke(server: str, tool: str, args: dict[str, Any]) -> Any:
    """
    Call a namespaced upstream tool.

    Args:
        server: Slug for the upstream (e.g. "fs").
        tool: Raw tool name exposed by that upstream (e.g. "read_file").
        args: JSON-serialisable arguments passed straight through.

    Example:
        invoke(server="fs", tool="read_file", args={"path": "/etc/hosts"})
    """
    await ensure_initialized()
    upstream = next((u for u in UPSTREAMS if u.name == server), None)
    if not upstream:
        raise ValueError(f"Unknown upstream '{server}'")
    if not upstream.session:
        raise RuntimeError(f"Upstream '{server}' is not connected")
    return await upstream.session.call_tool(tool, args or {})


@mcp.tool(description="Return all tools discovered from MCP servers in the MCPs directory.")
async def get_registry() -> dict[str, Any]:
    """Expose the available upstream tool registry for discovery and routing."""
    await ensure_initialized()
    registry: list[dict[str, Any]] = []
    for upstream in UPSTREAMS:
        tools: list[dict[str, Any]] = []
        if upstream.session and upstream.tools:
            for tool_name, meta in sorted((upstream.tools or {}).items()):
                tools.append(
                    {
                        "name": tool_name,
                        "description": meta.get("description", ""),
                        "inputSchema": _serialize_input_schema(meta.get("inputSchema")),
                    }
                )
        registry.append(
            {
                "name": upstream.name,
                "transport": upstream.transport,
                "connected": bool(upstream.session),
                "tools": tools,
            }
        )
    return {"servers": registry}


@mcp.tool(description="Explain how to call each upstream tool via invoke().")
async def get_tool_usage() -> dict[str, Any]:
    """
    Provide ready-to-use `invoke` payloads for every discovered tool.

    This bridges host agents that cannot see dynamic tool names by listing the
    namespaced identifiers and example arguments they must send to invoke().
    """

    await ensure_initialized()
    usage: list[dict[str, Any]] = []
    for upstream in UPSTREAMS:
        tools: list[dict[str, Any]] = []
        if upstream.session and upstream.tools:
            for tool_name, meta in sorted((upstream.tools or {}).items()):
                serialized_schema = _serialize_input_schema(meta.get("inputSchema"))
                tools.append(
                    {
                        "name": tool_name,
                        "description": meta.get("description", ""),
                        "inputSchema": serialized_schema,
                        "invoke": {
                            "tool": "invoke",
                            "arguments": {
                                "server": upstream.name,
                                "tool": tool_name,
                                "args": _build_args_template(serialized_schema),
                            },
                        },
                    }
                )
        usage.append(
            {
                "server": upstream.name,
                "transport": upstream.transport,
                "connected": bool(upstream.session),
                "tools": tools,
            }
        )
    return {"usage": usage}


# --- Dynamic proxy tools registration ---
# We construct small wrapper coroutines per upstream tool and register them on the fly.

def _make_proxy(u: Upstream, tool_name: str) -> Callable[..., Any]:
    fq = f"{u.name}.{tool_name}"

    async def proxy(**kwargs):  # type: ignore[no-untyped-def]
        if not u.session:
            raise RuntimeError(f"Upstream {u.name} is not connected")
        return await u.session.call_tool(tool_name, kwargs or {})

    proxy.__name__ = fq.replace(".", "_")
    proxy.__doc__ = f"Proxy for upstream tool '{tool_name}' on server '{u.name}'."
    return proxy


async def register_proxies() -> None:
    for u in UPSTREAMS:
        if not (u.session and u.tools):
            LOGGER.warning(
                "[upstream:%s] skipping proxy registration (connected=%s, tools=%s)",
                u.name,
                bool(u.session),
                bool(u.tools),
            )
            continue
        for tname, meta in u.tools.items():
            proxy_fn = _make_proxy(u, tname)
            mcp.tool(name=f"{u.name}.{tname}", description=meta.get("description") or "")(proxy_fn)


# -------------------------
# ASGI app (HTTP transport)
# -------------------------


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    """Surface connection status for debugging host integrations."""
    await ensure_initialized()
    return JSONResponse(
        {
            "status": "ok",
            "initialized": _INITIALIZED,
            "upstreams": [
                {
                    "name": u.name,
                    "transport": u.transport,
                    "tools": sorted(list((u.tools or {}).keys())),
                    "connected": bool(u.session),
                    "error": u.last_error,
                }
                for u in UPSTREAMS
            ],
        }
    )


app = mcp.streamable_http_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)


@app.on_event("startup")
async def on_startup():
    logging.basicConfig(level=logging.INFO)
    await ensure_initialized()
    LOGGER.info("Unified MCP ready. Connect clients to /mcp.")


@app.on_event("shutdown")
async def on_shutdown():
    global UPSTREAM_EXIT_STACK, _INITIALIZED, _INIT_LOCK
    if UPSTREAM_EXIT_STACK is not None:
        try:
            await asyncio.wait_for(UPSTREAM_EXIT_STACK.aclose(), timeout=5)
        except asyncio.TimeoutError:
            LOGGER.warning("Timed out while closing upstream connections; continuing shutdown anyway.")
        except Exception:
            LOGGER.exception("Error while closing upstream connections.")
        finally:
            UPSTREAM_EXIT_STACK = None
    for u in UPSTREAMS:
        u.session = None
        u.tools = None
        u.get_session_id = None
    _INITIALIZED = False
    _INIT_LOCK = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("run_http:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
