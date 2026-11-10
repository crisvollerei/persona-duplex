"""Local, version-aware lexical recall. Relevance is not semantic truth."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import re

from .memory_text import unsupported_model_memory


def usable_user_memory(record):
    return (not record.get('superseded_by')
            and record.get('source') not in ('heuristic', 'heuristic_fallback')
            and record.get('evidence_scope') != 'need_count'
            and not unsupported_model_memory(record))


def _instant(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Memory time must include a timezone')
    return parsed


def memory_metadata(memory_scope=None, valid_from=None, expires_at=None):
    if memory_scope is not None and (not isinstance(memory_scope, str) or not memory_scope.strip()):
        raise ValueError('memory_scope must be nonempty text or None')
    times = {}
    for name, value in (('valid_from', valid_from), ('expires_at', expires_at)):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError('Memory time must be ISO text')
            times[name] = _instant(value)
    if len(times) == 2 and times['expires_at'] <= times['valid_from']:
        raise ValueError('Memory expiration must follow its start')
    return {k: v for k, v in dict(memory_scope=memory_scope, valid_from=valid_from,
                                 expires_at=expires_at).items() if v is not None}


def _eligible(record, scope, at):
    if not usable_user_memory(record):
        return False
    if record.get('memory_scope') not in (None, scope):
        return False
    try:
        if record.get('valid_from') is not None and at < _instant(record['valid_from']):
            return False
        if record.get('expires_at') is not None and at >= _instant(record['expires_at']):
            return False
    except (ValueError, TypeError):
        return False  # Malformed declared windows are not treated as permanent.
    return True


def tokens(text):
    result = set()
    for word in re.findall(r'[A-Za-z][A-Za-z0-9_+#.-]*|[+-]?\d+(?:\.\d+)?', text):
        result.add('w:' + word.casefold())
    for segment in re.findall(r'[\u4e00-\u9fff]+', text):
        result.update('c:' + c for c in segment)
        result.update('h:' + segment[i:i+2] for i in range(len(segment)-1))
    return result


def recall(query, records, *, limit=5, budget_chars=6000, scope=None, at=None, retriever=None):
    if not isinstance(query, str):
        raise TypeError('Memory query must be text')
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError('Memory limit must be a nonnegative integer or None')
    if type(budget_chars) is not int or budget_chars < 0:
        raise ValueError('Memory budget must be a nonnegative integer')
    if scope is not None and (not isinstance(scope, str) or not scope.strip()):
        raise ValueError('Memory scope must be nonempty text or None')
    instant = _instant(at) if at is not None else datetime.now(timezone.utc)
    candidates = [deepcopy(r) for r in records if _eligible(r, scope, instant)]
    if limit == 0 or budget_chars == 0 or not candidates:
        return []
    if retriever is not None:
        if not callable(retriever):
            raise TypeError('Memory retriever must be callable')
        ids = retriever(query, deepcopy(candidates))
        by_id = {r['id']: i for i,r in enumerate(candidates)}
        if (not isinstance(ids, (list, tuple)) or any(not isinstance(i, str) for i in ids)
                or len(ids) != len(set(ids)) or any(i not in by_id for i in ids)):
            raise ValueError('Memory retriever must return unique eligible IDs')
        order = [by_id[i] for i in ids]
    else:
        documents = [tokens(r.get('judgment', '') + '\n' + '\n'.join(
            q.get('quote', '') for q in r.get('evidence_quotes', []) if isinstance(q, dict)))
                     for r in candidates]
        frequencies = Counter(t for doc in documents for t in doc)
        query_tokens = tokens(query)
        def score(index):
            doc = documents[index]
            value = sum((.18 if t.startswith('c:') else 2 if t.startswith('w:') else 1)
                        * (1 + math.log((len(candidates)+1)/(frequencies[t]+1)))
                        for t in query_tokens & doc)
            return value / math.sqrt(1 + len(doc) / 30)
        order = sorted(range(len(candidates)), key=lambda i: (score(i), i), reverse=True)
    selected, remaining = [], budget_chars
    for i in order:
        size = len(json.dumps(candidates[i], ensure_ascii=False, allow_nan=False))
        if size > remaining:
            continue  # Never truncate a source quote or its negation/condition.
        selected.append(i)
        remaining -= size
        if limit is not None and len(selected) >= limit:
            break
    # Context rendering is chronological (and displays newest first). Relevance
    # chooses membership, not a false claim that the oldest hit is the newest.
    return [candidates[i] for i in sorted(selected)]
