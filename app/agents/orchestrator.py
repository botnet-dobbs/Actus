import asyncio
import uuid
import json
import structlog
from app.agents.builder import AgentConfig
from app.agents.tools import run_tool, _tool_schemas, _invoke_stack
from app.context.models import AgentContext, ContextualData
from app.context.store import save_context
from app.rag.retriever import retrieve
from app.config import get_settings
from app.llm.pii import scrub_pii
from app.llm.router import call_llm_with_retry
from app.observability.metrics import active_agent_runs, agent_runs_total

# these tools can run up to AGENT_TOTAL_TIMEOUT; override the 30s per-tool default
_LONG_RUNNING_TOOLS = {"invoke_agent", "invoke_agents_parallel", "chunk_and_index_document"}

log = structlog.get_logger()

_settings = get_settings()

AGENT_TOTAL_TIMEOUT = 600  # 10 minutes max for an entire agent run
LLM_PER_CALL_TIMEOUT = 120  # 120 seconds per LLM call

TOOL_CALL_PROMPT = """Available tools (respond ONLY with JSON):
{tools}
To call a tool:
{{"tool": "name", "args": {{"param": "value"}}}}
After a tool call you will receive the result. Continue reasoning until ready to produce the done signal.
To finish:
{{"done": true, "result": "your final answer", "confidence": 0.95}}
confidence: float 0.0-1.0 indicating certainty in the result (optional)
"""


def _build_rag_query(config: AgentConfig, extra_context: dict | None,
                     bound_log=None) -> str | None:
    if config.rag_query_template:
        try:
            return config.rag_query_template.format(**(extra_context or {}))
        except KeyError:
            (bound_log or log).warning("rag_template_vars_missing",
                                       template=config.rag_query_template,
                                       available=list((extra_context or {}).keys()))
            return config.rag_query_template
    if extra_context and "query" in extra_context:
        return str(extra_context["query"])
    return None


def extract_json(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        content = "\n".join(inner).strip()
    # raw_decode stops at the first balanced } and ignores surrounding text,
    # handling both text-before and text-after the JSON object.
    start = content.find("{")
    if start != -1:
        try:
            _, end = json.JSONDecoder().raw_decode(content, start)
            return content[start:end]
        except (ValueError, json.JSONDecodeError):
            pass
    return content


async def run_agent(config: AgentConfig, extra_context: dict | None = None,
                    event_queue: asyncio.Queue | None = None) -> dict:
    run_id = str(uuid.uuid4())
    bound_log = log.bind(agent_id=config.id, run_id=run_id)
    bound_log.info("agent_run_start")
    active_agent_runs.inc()
    _stack_token = _invoke_stack.set(_invoke_stack.get() + [config.id])
    status = "unknown"
    try:
        result = await _run_agent_inner(config, run_id, bound_log, extra_context, event_queue)
        status = result.get("status", "unknown")
        return result
    except Exception:
        status = "error"
        raise
    finally:
        _invoke_stack.reset(_stack_token)
        active_agent_runs.dec()
        agent_runs_total.labels(agent_id=config.id, status=status).inc()
        if event_queue is not None:
            await event_queue.put(None)  # end-of-stream sentinel, guaranteed on any exit


async def _run_agent_inner(
    config: AgentConfig,
    run_id: str,
    bound_log,
    extra_context: dict | None,
    event_queue: asyncio.Queue | None = None,
) -> dict:
    async def _emit(event: dict) -> None:
        if event_queue is not None:
            await event_queue.put(event)

    tools_desc = json.dumps(list(_tool_schemas.values()), indent=2)
    messages = [{"role": "system",
                 "content": config.system_prompt.rstrip() + "\n\n" + TOOL_CALL_PROMPT.format(tools=tools_desc)}]

    pii_detected = False
    context = AgentContext(agent_id=config.id, run_id=run_id)

    rag_query = _build_rag_query(config, extra_context, bound_log)
    if rag_query:
        try:
            loop = asyncio.get_running_loop()
            retrieved = await loop.run_in_executor(
                None, lambda: retrieve(rag_query, top_k=config.rag_top_k)
            )
            if retrieved:
                raw_text = "\n".join(
                    f"[{i+1}] {r['document']} (score: {r['score']})"
                    for i, r in enumerate(retrieved)
                )
                clean_text, pii_found = scrub_pii(raw_text)
                if pii_found:
                    bound_log.warning("rag_context_pii_scrubbed")
                    pii_detected = True
                messages.append({
                    "role": "user",
                    "content": f"Retrieved context (pre-loaded):\n{clean_text}",
                })
                context.data.append(ContextualData(
                    type="retrieved_documents",
                    object_ids=[r["metadata"]["object_id"] for r in retrieved
                                if "object_id" in r["metadata"]],
                    data=retrieved,
                ))
                bound_log.info("rag_context_loaded", count=len(retrieved), query=rag_query)
        except Exception as e:
            bound_log.warning("rag_context_failed", error=str(e))

    if extra_context:
        context_str = json.dumps(extra_context)
        clean_context, pii_found = scrub_pii(context_str)
        pii_detected = pii_detected or pii_found
        if pii_found:
            bound_log.warning("agent_context_pii_scrubbed")
        messages.append({"role": "user", "content": f"Context: {clean_context}"})

    def _try_save_context() -> None:
        try:
            save_context(context)
        except Exception as e:
            bound_log.error("context_save_failed", run_id=run_id, error=str(e))

    tool_calls_log: list[dict] = []
    iterations_used = 0
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    api_base = config.api_base or (_settings.ollama_base_url if "ollama" in config.model else None)

    for iteration in range(config.max_iterations):
        iterations_used = iteration + 1
        bound_log.info("agent_iteration", iteration=iteration)
        await _emit({"type": "iteration_start", "run_id": run_id, "iteration": iteration})

        try:
            response = await asyncio.wait_for(
                call_llm_with_retry(
                    model=config.model,
                    messages=messages,
                    temperature=config.temperature,
                    max_tokens=config.max_response_tokens,
                    timeout=LLM_PER_CALL_TIMEOUT,
                    api_base=api_base,
                ),
                timeout=LLM_PER_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            err = f"LLM call timed out after {LLM_PER_CALL_TIMEOUT}s"
            bound_log.error("llm_call_timeout", iteration=iteration, timeout=LLM_PER_CALL_TIMEOUT)
            _try_save_context()
            await _emit({"type": "done", "run_id": run_id, "status": "error",
                         "error": err, "iterations": iterations_used, "total_tokens": total_tokens})
            return {"run_id": run_id, "result": None, "confidence": None,
                    "iterations": iterations_used, "status": "error", "error": err,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens, "tool_calls": tool_calls_log,
                    "pii_detected": pii_detected}
        except Exception as e:
            bound_log.error("llm_call_failed", iteration=iteration, error=str(e))
            _try_save_context()
            await _emit({"type": "done", "run_id": run_id, "status": "error",
                         "error": str(e), "iterations": iterations_used, "total_tokens": total_tokens})
            return {"run_id": run_id, "result": None, "confidence": None,
                    "iterations": iterations_used, "status": "error", "error": str(e),
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens, "tool_calls": tool_calls_log,
                    "pii_detected": pii_detected}

        if response.usage:
            total_tokens += response.usage.total_tokens
            prompt_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens += getattr(response.usage, "completion_tokens", 0) or 0
            if total_tokens > config.token_budget:
                bound_log.warning("agent_token_budget_exceeded",
                                  total_tokens=total_tokens, budget=config.token_budget)
                _try_save_context()
                await _emit({"type": "done", "run_id": run_id, "status": "incomplete",
                             "result": "Token budget exceeded", "iterations": iterations_used,
                             "total_tokens": total_tokens})
                return {"run_id": run_id, "result": "Token budget exceeded",
                        "confidence": None, "iterations": iterations_used, "status": "incomplete",
                        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens, "tool_calls": tool_calls_log,
                        "pii_detected": pii_detected}

        raw_content = response.choices[0].message.content
        content = (raw_content or "").strip()
        messages.append({"role": "assistant", "content": content})

        try:
            action = json.loads(extract_json(content))
        except json.JSONDecodeError:
            bound_log.warning("llm_non_json_response", iteration=iteration)
            if iteration < config.max_iterations - 1:
                messages.append({"role": "user",
                                  "content": "Your response must be valid JSON. "
                                             "Respond ONLY with a tool call or the done signal."})
                continue
            _try_save_context()
            await _emit({"type": "done", "run_id": run_id, "status": "incomplete",
                         "result": content, "iterations": iterations_used, "total_tokens": total_tokens})
            return {"run_id": run_id, "result": content, "confidence": None,
                    "iterations": iterations_used, "status": "incomplete",
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens, "tool_calls": tool_calls_log,
                    "pii_detected": pii_detected}

        if not action.get("done") and "tool" not in action:
            bound_log.warning("llm_empty_action", iteration=iteration)
            messages.append({"role": "user",
                              "content": "Your response was not a valid tool call or done signal. "
                                         "Respond with a tool call or the done signal."})
            continue

        if action.get("done"):
            bound_log.info("agent_run_complete", iterations=iterations_used, total_tokens=total_tokens)
            _try_save_context()
            await _emit({"type": "done", "run_id": run_id, "status": "completed",
                         "result": action.get("result"), "confidence": action.get("confidence"),
                         "iterations": iterations_used, "total_tokens": total_tokens})
            return {"run_id": run_id, "result": action.get("result"),
                    "confidence": action.get("confidence"),
                    "iterations": iterations_used, "status": "completed",
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens, "tool_calls": tool_calls_log,
                    "pii_detected": pii_detected}

        if "tool" in action:
            tool_name = action["tool"]
            if tool_name not in config.tools:
                bound_log.warning("agent_used_unauthorised_tool", tool=tool_name)
                messages.append({"role": "user",
                                  "content": f"Tool '{tool_name}' is not available to you. "
                                             f"Available tools: {config.tools}"})
                tool_calls_log.append({"tool": tool_name, "success": False, "detail": "unauthorised"})
                continue

            tool_timeout = (AGENT_TOTAL_TIMEOUT if tool_name in _LONG_RUNNING_TOOLS
                            else 30.0)
            raw_args = action.get("args") or {}
            args = raw_args if isinstance(raw_args, dict) else {}
            if not isinstance(raw_args, dict) and raw_args is not None:
                bound_log.warning("llm_invalid_args_type",
                                  tool=tool_name, args_type=type(raw_args).__name__)
            await _emit({"type": "tool_call", "run_id": run_id, "iteration": iteration,
                         "tool": tool_name, "args": args})
            tool_result = await run_tool(tool_name, timeout_seconds=tool_timeout, **args)
            await _emit({"type": "tool_result", "run_id": run_id, "iteration": iteration,
                         "tool": tool_name, "success": tool_result.success,
                         "preview": json.dumps(tool_result.output, default=str)[:300]})
            tool_calls_log.append({
                "tool": tool_name,
                "success": tool_result.success,
                "detail": tool_result.error if not tool_result.success else "",
            })
            if tool_result.success:
                tool_output = (json.dumps(tool_result.output, default=str)
                               if tool_result.output is not None else "null")
            else:
                tool_output = json.dumps({"error": tool_result.error})
            messages.append({"role": "user", "content": f"Tool result: {tool_output}"})

    bound_log.warning("agent_max_iterations_reached", iterations=iterations_used, total_tokens=total_tokens)
    _try_save_context()
    await _emit({"type": "done", "run_id": run_id, "status": "incomplete",
                 "result": "Max iterations reached", "iterations": iterations_used,
                 "total_tokens": total_tokens})
    return {"run_id": run_id, "result": "Max iterations reached",
            "confidence": None, "iterations": iterations_used, "status": "incomplete",
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": total_tokens, "tool_calls": tool_calls_log,
            "pii_detected": pii_detected}


async def run_agent_with_timeout(config: AgentConfig, extra_context: dict | None = None,
                                  event_queue: asyncio.Queue | None = None) -> dict:
    try:
        return await asyncio.wait_for(
            run_agent(config, extra_context=extra_context, event_queue=event_queue),
            timeout=AGENT_TOTAL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error("agent_total_timeout", agent_id=config.id, timeout=AGENT_TOTAL_TIMEOUT)
        return {"run_id": "unknown", "result": "Agent run exceeded time limit",
                "confidence": None, "iterations": 0, "status": "timeout",
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "tool_calls": [], "pii_detected": False}
