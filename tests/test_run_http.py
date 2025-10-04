import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
import pytest_asyncio



def list_local_mcp_scripts():
    base = Path(__file__).resolve().parent.parent
    mcp_dir = base / "MCPs"
    return sorted([p for p in mcp_dir.glob("*.py")])


@pytest_asyncio.fixture(scope="module")
async def run_http_mod():
    import run_http
    await run_http.ensure_initialized()
    return run_http


@pytest.mark.asyncio
async def test_upstreams_connect_and_list_tools(run_http_mod):
    expected = {p.stem.lower().replace("_", "-") for p in list_local_mcp_scripts()}
    names = {u.name for u in run_http_mod.UPSTREAMS}
    assert expected.issubset(names), f"Missing upstreams: {expected - names}"

    for u in run_http_mod.UPSTREAMS:
        if u.name not in expected:
            continue
        assert u.session is not None, f"Upstream {u.name} not connected (last_error={u.last_error})"
        assert u.tools, f"Upstream {u.name} has no tools"
        assert len(u.tools) > 0, f"Upstream {u.name} tools list is empty"


@pytest_asyncio.fixture(scope="module")
async def registry(run_http_mod):
    return await run_http_mod.get_registry()


@pytest_asyncio.fixture(scope="module")
async def usage(run_http_mod):
    return await run_http_mod.get_tool_usage()


@pytest.mark.asyncio
async def test_registry_and_usage_tools_present(registry, usage):

    servers = {s["name"]: s for s in registry["servers"]}
    assert servers, "Registry returned no servers"

    for p in list_local_mcp_scripts():
        name = p.stem.lower().replace("_", "-")
        assert name in servers, f"Missing server in registry: {name}"
        s = servers[name]
        assert s["connected"], f"Server {name} not connected"
        assert s["tools"], f"Server {name} has empty tool list"

    usage_map = {u["server"]: u for u in usage["usage"]}
    for p in list_local_mcp_scripts():
        name = p.stem.lower().replace("_", "-")
        assert name in usage_map, f"Missing usage for server: {name}"
        assert usage_map[name]["tools"], f"Usage tools empty for server: {name}"


# -------------------------
# New helpers and tests
# -------------------------

def _choose_non_null_type(t):
    if isinstance(t, list):
        for cand in t:
            if cand != "null":
                return cand
        return t[0] if t else None
    return t


def _make_example_value(prop_schema):
    if isinstance(prop_schema, dict) and "enum" in prop_schema:
        vals = prop_schema.get("enum") or []
        if isinstance(vals, list) and vals:
            return vals[0]

    s = prop_schema if isinstance(prop_schema, dict) else {}

    for key in ("oneOf", "anyOf", "allOf"):
        if key in s and isinstance(s[key], list) and s[key]:
            return _make_example_value(s[key][0])

    t = _choose_non_null_type(s.get("type"))
    if t in (None, "string"):
        return "test"
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return False
    if t == "array":
        items = s.get("items")
        if items:
            return [_make_example_value(items)]
        return []
    if t == "object":
        props = s.get("properties") or {}
        req = s.get("required") or []
        return {k: _make_example_value(props.get(k, {})) for k in req}
    return "test"


def _make_args_from_schema(schema):
    """
    Build minimal args dict based on required fields of a JSON Schema object.
    If no schema or no required fields, returns {}.
    """
    if not isinstance(schema, dict):
        return {}
    if schema.get("type") == "null":
        return {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    args = {}
    for name in required:
        args[name] = _make_example_value(props.get(name, {}))
    return args


@pytest.mark.asyncio
async def test_list_tools_for_each_mcp_server_and_non_empty(registry):
    """
    1) Fail if not all servers (from MCPs/*.py) are detected in registry
    2) Fail if any tool list is empty
    """
    expected_servers = {p.stem.lower().replace("_", "-") for p in list_local_mcp_scripts()}
    servers = {s["name"]: s for s in registry["servers"]}
    missing = expected_servers - set(servers.keys())
    assert not missing, f"Missing servers in registry: {missing}"

    for name in expected_servers:
        s = servers[name]
        assert s["connected"], f"Server {name} not connected"
        tool_names = [t["name"] for t in (s["tools"] or [])]
        assert tool_names, f"Server {name} has empty tool list"


@pytest.mark.asyncio
@pytest.mark.parametrize("target_server", ["memory", "websearch"])
async def test_invoke_example_calls_for_targets(run_http_mod, usage, target_server):
    """
    3) Make example tool calls via run_http.invoke for Memory and WebSearch.
       - Discover tools and input schemas from get_tool_usage
       - Build minimal required args from schema
       - Try tools until one invocation succeeds per server
    """
    usage_map = {u["server"]: u for u in usage["usage"]}
    assert target_server in usage_map, f"Expected server missing from usage: {target_server}"
    assert usage_map[target_server]["connected"], f"Server {target_server} not connected in usage"

    tools = usage_map[target_server]["tools"]
    assert tools, f"Server {target_server} has empty tool list in usage"

    invoked_ok = False
    last_error = None

    for tool_meta in tools:
        tool_name = tool_meta["name"]
        schema = tool_meta.get("inputSchema") or {}
        args = _make_args_from_schema(schema)

        try:
            _ = await run_http_mod.invoke(server=target_server, tool=tool_name, args=args)
            invoked_ok = True
            break
        except Exception as exc:
            last_error = exc
            continue

    assert invoked_ok, (
        f"Failed to invoke any tool for server '{target_server}'. Last error: {last_error}"
    )
