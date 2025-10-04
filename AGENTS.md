The `run_http.py` file is not usable. 

When use with LM Studio.

```log
2025-10-04 18:08:03 [ERROR]
 [Plugin(mcp/1mcp)] stderr: Error in LM Studio MCP bridge process: _0x45c9e2 [Error]: SSE error: Non-200 status code (500)
    at <computed>.onerror (E:\LMStudio\LM Studio\resources\app\.webpack\lib\mcpbridgeworker.js:29:196739)
    at _0x550d3c._0x13c134 (E:\LMStudio\LM Studio\resources\app\.webpack\lib\mcpbridgeworker.js:29:338977)
    at E:\LMStudio\LM Studio\resources\app\.webpack\lib\mcpbridgeworker.js:29:332689
    at process.processTicksAndRejections (node:internal/process/task_queues:95:5) {
  code: 500,
  event: {
    type: 'error',
    message: 'Non-200 status code (500)',
    code: 500,
    defaultPrevented: false,
    cancelable: false,
    timeStamp: 249.0449
  }
}
_0x45c9e2 [Error]: SSE error: Non-200 status code (500)
    at <computed>.onerror (E:\LMStudio\LM Studio\resources\app\.webpack\lib\mcpbridgeworker.js:29:196739)
    at _0x550d3c._0x13c134 (E:\LMStudio\LM Studio\resources\app\.webpack\lib\mcpbridgeworker.js:29:338977)
    at E:\LMStudio\LM Studio\resources\app\.webpack\lib\mcpbridgeworker.js:29:332689
    at process.processTicksAndRejections (node:internal/process/task_queues:95:5) {
  code: 500,
  event: {
    type: 'error',
    message: 'Non-200 status code (500)',
    code: 500,
    defaultPrevented: false,
    cancelable: false,
    timeStamp: 249.0449
  }
}
```
And when use with VSCode:
```log
2025-10-04 18:07:20.178 [error] Server exited before responding to `initialize` request.
2025-10-04 18:09:26.115 [info] Stopping server 1mcp
2025-10-04 18:09:26.126 [info] Starting server 1mcp
2025-10-04 18:09:26.127 [info] Connection state: Starting
2025-10-04 18:09:26.131 [info] Starting server from LocalProcess extension host
2025-10-04 18:09:26.134 [info] Connection state: Running
2025-10-04 18:09:26.139 [info] Connection state: Error Error sending message to http://0.0.0.0/8000/mcp: TypeError: fetch failed
2025-10-04 18:09:26.139 [error] Server exited before responding to `initialize` request.
```
uvicron log 
```log
INFO:     127.0.0.1:7589 - "GET /mcp HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 403, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        self.scope, self.receive, self.send
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\uvicorn\middleware\proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\middleware\errors.py", line 186, in __call__
    raise exc
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\middleware\errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\middleware\cors.py", line 85, in __call__
    await self.app(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\middleware\exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\routing.py", line 462, in handle
    await self.app(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\middleware\errors.py", line 186, in __call__
    raise exc
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\middleware\errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\middleware\exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\starlette\routing.py", line 290, in handle
    await self.app(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\mcp\server\fastmcp\server.py", line 1014, in __call__
    await self.session_manager.handle_request(scope, receive, send)
  File "D:\MyProject\lmstudio-toolpack\.venv\Lib\site-packages\mcp\server\streamable_http_manager.py", line 138, in handle_request
    raise RuntimeError("Task group is not initialized. Make sure to use run().")
RuntimeError: Task group is not initialized. Make sure to use run().
```

I think the issue is with using uvicron.
FastMCP directly use MCP protocal, and uvicron we had to manually configure to the protocal. 
Current run_http.py does not follow the protocal and that is why it failed. I believe.

If you think it was another problem then fix in another way. 
