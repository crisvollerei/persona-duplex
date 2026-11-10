"""Conservative textual equivalence for memories, not topic similarity.

Only typography is normalized. Different wording is deliberately retained:
neither a shared topic nor a close embedding establishes equal preferences.
"""
from __future__ import annotations

import re
import unicodedata


_PUNCTUATION_STYLES = str.maketrans({
    "，": ",", "；": ";", "：": ":", "！": "!", "？": "?",
    "“": '"', "”": '"', "‘": "'", "’": "'",
})
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]")


def memory_text_key(text: str) -> str:
    """Keep semantic tokens and separators while normalizing limited layout.

    ASCII word/number boundaries stay separate (``Reader Pro`` is not
    ``ReaderPro`` and ``1 2`` is not ``12``). Decimal points, signs, operators,
    hyphens, dates, names, case and internal punctuation are not discarded.
    """
    value = unicodedata.normalize("NFC", text).translate(_PUNCTUATION_STYLES).strip()
    quoted = [(match.start(), match.end()) for match in re.finditer(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`[^`]*`', value)]
    def whitespace(match):
        if any(start <= match.start() < end for start, end in quoted):
            return match.group(0)
        before = value[match.start() - 1:match.start()]
        after = value[match.end():match.end() + 1]
        return " " if _ASCII_WORD.fullmatch(before) and _ASCII_WORD.fullmatch(after) else ""
    value = re.sub(r"\s+", whitespace, value)
    # A Chinese sentence-ending stop is typography. Do not strip a decimal or
    # an ASCII filename's terminal dot, and do not remove internal punctuation.
    return re.sub(r"(?<=[\u4e00-\u9fff])[。.]+$", "", value)


def equivalent_memory_text(a: str, b: str) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    left, right = memory_text_key(a), memory_text_key(b)
    return bool(left and left == right)


def unsupported_model_memory(record: dict) -> bool:
    """Separate unsupported model proposals from application/legacy records."""
    return record.get("source") == "llm" and (
        record.get("evidence_scope") == "unsupported" or
        not (record.get("evidence_ids") or record.get("evidence_quotes")))
