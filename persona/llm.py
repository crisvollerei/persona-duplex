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
    base = (base_url or os.environ.get("PERSONA_BASE_URL")
            or "https://api.openai.com/v1").rstrip("/")
    key = api_key or os.environ.get("PERSONA_API_KEY") or ""
    url = base + "/chat/completions"

    def call(messages: list[Message]) -> str:
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra:
            body.update(extra)
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
        return payload["choices"][0]["message"]["content"]

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


def text_prompt(fn: Callable[[str], str]) -> Callable[[str], str]:
    """把「一段文本 → 文本」的函数包装成后台用的接口。"""
    return fn


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
