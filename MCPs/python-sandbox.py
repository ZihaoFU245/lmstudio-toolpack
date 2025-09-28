from fastmcp import FastMCP
from typing import Optional, Any, Dict, List, Union
import multiprocessing as mp
import io, contextlib, time, ast, os, signal, json

# Added scientific stack imports
import numpy as np
import sympy as sp

mcp = FastMCP("Python SandBox")

SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "range": range, "len": len, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "sorted": sorted, "any": any, "all": all, "print": print,
}

MAX_STDIO = 64_000    # cap stdout/stderr
MAX_RESULT_BYTES = 128_000

def worker_run(src: str, inp: Optional[Dict[str, Any]], q):
    start = time.time()
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    try:
        globals_env = {"__builtins__": SAFE_BUILTINS, "inputs": (inp or {})}
        locals_env: Dict[str, Any] = {}

        try:
            node = ast.parse(src, mode="eval")
            is_expr = True
        except SyntaxError:
            is_expr = False

        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            if is_expr and isinstance(node, ast.Expression):
                result = eval(compile(node, "<string>", "eval"), globals_env, locals_env)
            else:
                compiled = compile(src, "<string>", "exec")
                exec(compiled, globals_env, locals_env)
                result = locals_env.get("result", None)

        payload = result
        try:
            s = json.dumps(payload, default=str)
            if len(s.encode("utf-8")) > MAX_RESULT_BYTES:
                payload = {"_truncated": True, "type": type(result).__name__}
        except Exception as e:
            payload = {"_error": f"SerializeError: {e}"}

        q.put({
            "stdout": stdout_buf.getvalue()[:MAX_STDIO],
            "stderr": stderr_buf.getvalue()[:MAX_STDIO],
            "result": payload,
            "error": None,
            "timed_out": False,
            "duration": time.time() - start,
        })
    except Exception as e:
        q.put({
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result": None,
            "error": f"{type(e).__name__}: {e}",
            "timed_out": False,
            "duration": time.time() - start,
        })


@mcp.tool
def run_python(code: str, inputs: Optional[Dict[str, Any]] = None, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    """Execute a short Python snippet in a restricted subprocess.

    - Safe builtins only, no imports.
    - Captures stdout/stderr.
    - If single expression, returns its value.
    - Otherwise, assign to `result` in code.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=worker_run, args=(code, inputs, q))
    p.start()
    p.join(timeout_seconds)
    if p.is_alive():
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            p.terminate()
        p.join(1)
        return {"stdout": "", "stderr": "", "result": None, "error": "Timeout",
                "timed_out": True, "duration": timeout_seconds}
    try:
        return q.get_nowait()
    except Exception:
        return {"stdout": "", "stderr": "", "result": None, "error": "No result",
                "timed_out": False, "duration": None}


if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_stdio_async())

# -------------------------- Added MCP Tools using NumPy / SymPy --------------------------

@mcp.tool
def numpy_stats(data: List[Union[int, float]]) -> Dict[str, float]:
    """Compute basic statistics on a numeric list using NumPy.

    Parameters
    ----------
    data: list[int|float]
        A non-empty list of numeric values.

    Returns
    -------
    dict with keys: count, mean, std, min, max, sum
    """
    if not data:
        return {"error": "empty input list"}
    arr = np.asarray(data, dtype=float)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


@mcp.tool
def sympy_analyze(expr: str, solve_for: Optional[str] = None) -> Dict[str, Any]:
    """Symbolically analyze an algebraic expression using SymPy.

    - Parses the input string into a SymPy expression.
    - Returns its simplified form, expanded form, factored form.
    - Optionally solves for a single symbol if `solve_for` is provided and the equation contains '='.

    Examples
    --------
    sympy_analyze("(x**2 - 1)/(x-1)")
    sympy_analyze("x^2 - 4 = 0", solve_for="x")
    """
    try:
        # Handle equation case if '=' present
        if '=' in expr:
            left_str, right_str = expr.split('=', 1)
            left = sp.sympify(left_str)
            right = sp.sympify(right_str)
            sym_expr = left - right
        else:
            sym_expr = sp.sympify(expr)
        simplified = sp.simplify(sym_expr)
        expanded = sp.expand(sym_expr)
        factored = sp.factor(sym_expr)

        result: Dict[str, Any] = {
            "input": expr,
            "simplified": str(simplified),
            "expanded": str(expanded),
            "factored": str(factored),
            "free_symbols": sorted(str(s) for s in sym_expr.free_symbols),
        }

        if solve_for:
            symbol = sp.Symbol(solve_for)
            try:
                solutions = sp.solve(sp.Eq(sym_expr, 0), symbol, dict=True)
                # Convert solutions (list[dict]) to serializable form
                ser_solutions = []
                for sol in solutions:
                    ser_solutions.append({str(k): str(v) for k, v in sol.items()})
                result["solutions"] = ser_solutions
            except Exception as e:  # solving may fail
                result["solve_error"] = f"{type(e).__name__}: {e}"

        return result
    except Exception as e:
        return {"error": f"ParseError: {type(e).__name__}: {e}", "input": expr}

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from GlobalConfig import GlobalConfig
    import asyncio
    
    if GlobalConfig.transport == "http":
        asyncio.run(mcp.run_http_async(GlobalConfig.port) if GlobalConfig.port else mcp.run_http_async()) 
    else:
        asyncio.run(mcp.run_stdio_async())
