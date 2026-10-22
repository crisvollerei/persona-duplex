"""LLM 适配层。

零依赖：只用 urllib。任何 OpenAI 兼容端点都能用。
也提供 Mock，用于离线跑通全流程与测试。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Iterator

Message = dict[str, str]


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
    url, key, timeout = _endpoint(base_url, api_key, timeout)
    _default_max = max_tokens

    def call(messages: list[Message], *, max_tokens: int | None = None) -> str:
        """max_tokens 可以在每次调用时覆盖。

        判定器需要按待判条数动态给预算 —— 固定值在挂满预言时会把
        JSON 输出截断，整批解析失败。主对话用默认值即可。
        """
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or _default_max,
        }
        if extra:
            body.update(extra)
        payload = _post(url, key, body, timeout)
        return payload["choices"][0]["message"]["content"]

    return call


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

    这是「兜底协议」的底层：结构由 JSON Schema 保证，
    字段不可能缺失、类型不可能错、也不会有标记藏在正文里。

    用法：
        fn = openai_tool_call("deepseek-chat")
        data = fn(messages, tool=TOOL_SPEC, max_tokens=400)

    失败（模型不调工具 / JSON 坏）一律抛 RuntimeError ——
    由上层决定要不要退回文本协议。
    """
    url, key, timeout = _endpoint(base_url, api_key, timeout)
    _default_max = max_tokens

    def call(messages: list[Message], *, tool: dict,
             max_tokens: int | None = None) -> dict:
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
        msg = payload["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            raise RuntimeError("模型没有调用工具")
        raw = calls[0]["function"]["arguments"]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"工具参数不是合法 JSON：{raw[:120]}") from e
        if not isinstance(data, dict):
            raise RuntimeError(f"工具参数不是对象：{type(data).__name__}")
        return data

    return call


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

    注意：这里只做 SSE 的最小实现，够跑通。生产环境请换官方 SDK
    处理重连、超时和 usage 统计。
    """
    base = (base_url or os.environ.get("PERSONA_BASE_URL")
            or "https://api.openai.com/v1").rstrip("/")
    key = api_key or os.environ.get("PERSONA_API_KEY") or ""
    url = base + "/chat/completions"

    def call(messages: list[Message]):
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
                    delta = obj["choices"][0].get("delta", {})
                    piece = delta.get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if piece:
                    yield piece

    return call


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
