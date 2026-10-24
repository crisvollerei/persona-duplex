"""LLM 适配层。

零依赖：只用 urllib。任何 OpenAI 兼容端点都能用。
也提供 Mock，用于离线跑通全流程与测试。
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from typing import Callable, Iterator

Message = dict[str, str]


def _describe(fn, url, model, **conditions):
    parts = urlsplit(url)
    host = parts.netloc.rsplit("@", 1)[-1]
    fn.metadata = {"kind": "openai_compatible", "conditions_known": True,
                   "endpoint": urlunsplit((parts.scheme, host, parts.path, "", "")),
                   "model": model, **conditions}
    fn.observations = threading.local()
    return fn


def _response(payload, choice=None):
    return {"id": payload.get("id"), "model": payload.get("model"),
            "system_fingerprint": payload.get("system_fingerprint"),
            "usage": payload.get("usage"),
            "finish_reason": (choice or {}).get("finish_reason")}


def _safe_extra(extra):
    allowed = {"temperature", "top_p", "max_tokens", "seed", "stop",
               "presence_penalty", "frequency_penalty", "reasoning_effort",
               "thinking", "response_format"}
    return {k: v for k, v in (extra or {}).items() if k in allowed}


def _extra_conditions(extra):
    safe = _safe_extra(extra)
    unknown = sorted(set(extra or {}) - safe.keys())
    return {"extra": safe, "unrecorded_extra_keys": unknown, "conditions_known": not unknown}


def _validate_extra(extra):
    reserved = {"model", "messages", "stream", "tools", "tool_choice"}
    if reserved.intersection(extra or {}):
        raise ValueError("extra cannot override model, messages or protocol routing")


def _endpoint(base_url: str | None, api_key: str | None,
              timeout: float) -> tuple[str, str, float]:
    base = (base_url or os.environ.get("PERSONA_BASE_URL")
            or "https://api.openai.com/v1").rstrip("/")
    key = api_key or os.environ.get("PERSONA_API_KEY") or ""
    return base + "/chat/completions", key, timeout


def _post(url: str, key: str, body: dict, timeout: float) -> dict:
    """一次非流式调用。所有后端共用，避免四份重复的 urllib 代码。"""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e


def openai_compatible(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.9,
    max_tokens: int = 900,
    timeout: float = 60.0,
    extra: dict | None = None,
) -> Callable[[list[Message]], str]:
    _validate_extra(extra)
    url, key, timeout = _endpoint(base_url, api_key, timeout)
    _default_max = max_tokens

    def call(messages: list[Message], *, max_tokens: int | None = None) -> str:
        """max_tokens 可以在每次调用时覆盖。

        判定器需要按待判条数动态给预算 —— 固定值在挂满预言时会把
        JSON 输出截断，整批解析失败。主对话用默认值即可。
        """
        call.observations.response = None
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or _default_max,
        }
        if extra:
            body.update(extra)
        payload = _post(url, key, body, timeout)
        # 内容缺失返回空串；明确标为截断或拦截的生成不能当作完成。
        try:
            choice = payload["choices"][0]
            content = choice["message"].get("content")
        except (KeyError, IndexError, TypeError):
            return ""
        call.observations.response = _response(payload, choice)
        reason = choice.get("finish_reason")
        if reason not in (None, "stop"):
            raise RuntimeError(f"生成未正常完成：finish_reason={reason}")
        return content if isinstance(content, str) else ""

    return _describe(call, url, model, temperature=temperature, max_tokens=max_tokens,
                     timeout=timeout, **_extra_conditions(extra))


def openai_tool_call(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.9,
    max_tokens: int = 900,
    timeout: float = 60.0,
    extra: dict | None = None,
) -> Callable[..., dict]:
    """强制模型把输出填进一个函数的参数里，返回解析后的 dict。

    请求端点按 JSON Schema 返回结构。兼容端点未必严格执行 schema，
    因此关键阶段仍需由上层验证字段与类型。

    用法：
        fn = openai_tool_call("deepseek-chat")
        data = fn(messages, tool=TOOL_SPEC, max_tokens=400)

    失败（模型不调工具 / JSON 坏）一律抛 RuntimeError ——
    由上层决定要不要退回文本协议。
    """
    _validate_extra(extra)
    url, key, timeout = _endpoint(base_url, api_key, timeout)
    _default_max = max_tokens

    def call(messages: list[Message], *, tool: dict,
             max_tokens: int | None = None) -> dict:
        call.observations.response = None
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or _default_max,
            "tools": [tool],
            "tool_choice": {"type": "function",
                            "function": {"name": tool["function"]["name"]}},
        }
        if extra:
            body.update(extra)
        payload = _post(url, key, body, timeout)
        choice = payload["choices"][0]
        call.observations.response = _response(payload, choice)
        reason = choice.get("finish_reason")
        if reason not in (None, "stop", "tool_calls"):
            raise RuntimeError(f"工具生成未正常完成：finish_reason={reason}")
        msg = choice["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            raise RuntimeError("模型没有调用工具")
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != tool["function"]["name"]:
            raise RuntimeError("模型返回的工具不匹配所请求的阶段")
        raw = calls[0]["function"]["arguments"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"工具参数不是合法 JSON：{raw[:120]}") from e
        if not isinstance(data, dict):
            raise RuntimeError(f"工具参数不是对象：{type(data).__name__}")
        return data

    return _describe(call, url, model, temperature=temperature, max_tokens=max_tokens,
                     timeout=timeout, **_extra_conditions(extra))


def openai_stream(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.9,
    max_tokens: int = 900,
    timeout: float = 90.0,
) -> Callable[[list[Message]], "Iterator[str]"]:
    """流式版本。逐片吐文本增量。

    这里只做 SSE 的最小实现，识别完成、截断与中断，不自动重试。
    更完整的事件处理和 usage 统计可替换适配器，不改变上层交付约定。
    """
    base = (base_url or os.environ.get("PERSONA_BASE_URL")
            or "https://api.openai.com/v1").rstrip("/")
    key = api_key or os.environ.get("PERSONA_API_KEY") or ""
    url = base + "/chat/completions"

    def call(messages: list[Message]):
        call.observations.response = None
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        completed = False
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    raise RuntimeError(f"流式服务错误：{obj['error']}")
                choices = obj.get("choices") if isinstance(obj, dict) else None
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    continue
                choice = choices[0]
                previous = call.observations.response or {}
                observation = _response(obj, choice)
                call.observations.response = {**previous, **{
                    k: v for k, v in observation.items() if v is not None}}
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
                reason = choice.get("finish_reason")
                if reason is not None:
                    if reason != "stop":
                        raise RuntimeError(f"生成未正常完成：finish_reason={reason}")
                    completed = True
        if not completed:
            raise RuntimeError("生成流式中断：连接结束但未收到完成标记")

    return _describe(call, url, model, temperature=temperature, max_tokens=max_tokens,
                     timeout=timeout, stream=True)


def text_prompt(fn: Callable[..., str]) -> Callable[[str], str]:
    """把「messages 列表 → 文本」的调用，包装成「一段文本 → 文本」。

    背景：主对话的 llm 接口是 openai_compatible(...)，它收的是
    [{"role": "user", "content": ...}]。但后台固化只想传一段 prompt 字符串。

    这个函数原本是个空实现（直接 return fn），导致固化线程拿到字符串后
    原样塞进 messages 字段，HTTP 422。因为所有测试都用假 LLM，这个 bug
    一直没暴露 —— 直到真实跑了一次 20 轮对话。

    现在真的做转换：把字符串包成单条 user message。
    """

    def call(prompt: str) -> str:
        return fn([{"role": "user", "content": prompt}])

    call.metadata = getattr(fn, "metadata", {"kind": "custom", "conditions_known": False})
    call.observations = getattr(fn, "observations", threading.local())
    return call


# ---------------------------------------------------------------- mock

class ScriptedLLM:
    """按脚本返回，用于离线跑通和回归测试。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    def __call__(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if not self.replies:
            raise RuntimeError("ScriptedLLM 脚本已用尽")
        return self.replies.pop(0)


def make_reply(head: str, say: str, tail: str = "") -> str:
    """拼一个符合协议的回复，测试和演示都用它。"""
    parts = ["<<<HEAD>>>", head.strip(), "", "<<<SAY>>>", say.strip()]
    if tail.strip():
        parts += ["", "<<<TAIL>>>", tail.strip()]
    return "\n".join(parts)
