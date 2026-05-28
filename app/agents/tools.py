import time
import asyncio
import inspect
from typing import Callable, Any, get_type_hints
from pydantic import BaseModel
import structlog

log = structlog.get_logger()


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any
    error: str = ""
    duration_ms: float = 0.0


_tools: dict[str, Callable] = {}
_tool_descriptions: dict[str, str] = {}
_tool_schemas: dict[str, dict] = {}


def _clean_annotation(annotation) -> str:
    if annotation is inspect.Parameter.empty:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    if hasattr(annotation, "__origin__"):
        return str(annotation)\
            .replace("typing.", "")\
            .replace("builtins.", "")
    return str(annotation).replace("typing.", "").replace("builtins.", "")


def tool(name: str, description: str):
    def decorator(fn: Callable):
        _tools[name] = fn
        _tool_descriptions[name] = description

        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        params = {}
        for param_name, param in inspect.signature(fn).parameters.items():
            type_hint = hints.get(param_name, param.annotation)
            params[param_name] = {
                "type": _clean_annotation(type_hint),
                "required": param.default is inspect.Parameter.empty,
            }
            if param.default is not inspect.Parameter.empty:
                params[param_name]["default"] = param.default

        _tool_schemas[name] = {
            "name": name,
            "description": description,
            "parameters": params,
        }
        return fn
    return decorator


async def run_tool(name: str, timeout_seconds: float = 30.0, **kwargs) -> ToolResult:
    if name not in _tools:
        return ToolResult(tool_name=name, success=False,
                          output=None, error=f"Unknown tool: {name}")
    fn = _tools[name]
    start = time.monotonic()
    try:
        if inspect.iscoroutinefunction(fn):
            output = await asyncio.wait_for(fn(**kwargs), timeout=timeout_seconds)
        else:
            loop = asyncio.get_running_loop()
            output = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: fn(**kwargs)),
                timeout=timeout_seconds,
            )
        duration_ms = (time.monotonic() - start) * 1000
        log.info("tool_executed", tool=name,
                 duration_ms=round(duration_ms, 1), success=True)
        return ToolResult(tool_name=name, success=True,
                          output=output, duration_ms=duration_ms)
    except asyncio.TimeoutError:
        log.error("tool_timeout", tool=name, timeout=timeout_seconds)
        return ToolResult(tool_name=name, success=False,
                          output=None, error=f"Timeout after {timeout_seconds}s")
    except Exception as e:
        log.error("tool_failed", tool=name, error=str(e), exc_info=True)
        return ToolResult(tool_name=name, success=False,
                          output=None, error=str(e))


def list_tools() -> list[dict]:
    return list(_tool_schemas.values())
