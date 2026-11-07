"""Serializable experiment conditions and per-request traces; never store credentials."""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from uuid import uuid4


def fingerprint(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def model_info(fn) -> dict:
    if fn is None:
        return {"kind": "disabled"}
    meta = getattr(fn, "metadata", None)
    return deepcopy(meta) if isinstance(meta, dict) else {
        "kind": "custom", "conditions_known": False,
        "callable": type(fn).__name__,
    }


def manifest(cfg, *, protocol, llm, stream_llm, tool_llm, options=None) -> dict:
    config = asdict(cfg)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"run_id": uuid4().hex, "schema_version": 1,
            "config": config, "config_hash": fingerprint(config),
            "source_hash": fingerprint(sources), "protocol": protocol,
            "models": {"text": model_info(llm), "stream": model_info(stream_llm),
                       "tool": model_info(tool_llm)}, "options": options or {}}


def traced(fn, route, records, emit=None):
    if fn is None:
        return None

    @wraps(fn)
    def call(*args, **kwargs):
        row = {"route": route, "model": model_info(fn),
               "messages": deepcopy(args[0]), "arguments": deepcopy(kwargs),
               "status": "running"}
        records().append(row)
        if emit is not None:
            emit("model_call_start", deepcopy(row))
        started = time.monotonic()

        def finished(status, error=None):
            row["status"] = status
            row["elapsed"] = round(time.monotonic() - started, 6)
            if error is not None:
                row["error_type"] = type(error).__name__
            observation = getattr(getattr(fn, "observations", None), "response", None)
            if observation is not None:
                row["response"] = deepcopy(observation)
            if emit is not None:
                emit("model_call_end", deepcopy(row))

        try:
            result = fn(*args, **kwargs)
            if route == "stream":
                def iterate():
                    pieces = []
                    try:
                        for piece in result:
                            pieces.append(piece)
                            yield piece
                    except BaseException as exc:
                        row["output"] = "".join(pieces)
                        finished("failed", exc)
                        raise
                    else:
                        row["output"] = "".join(pieces)
                        finished("complete")
                return iterate()
        except BaseException as exc:
            finished("failed", exc)
            raise
        row["output"] = deepcopy(result)
        finished("complete")
        return result

    return call
