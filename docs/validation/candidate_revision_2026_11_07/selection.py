"""Study-only selector. Model-supplied semantic tags are not verified facts."""
import hashlib
import json


def strict_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=pairs)


def parse_pool(text):
    obj = strict_json(text)
    if not isinstance(obj, dict) or set(obj) != {"candidates"}:
        raise ValueError("expected candidates object")
    pool = obj["candidates"]
    if not isinstance(pool, list) or len(pool) != 6:
        raise ValueError("expected six candidates")
    for i, row in enumerate(pool, 1):
        if not isinstance(row, dict) or set(row) != {"id", "text", "attention", "initiative"}:
            raise ValueError("unexpected candidate fields")
        if row["id"] != f"C{i:02}":
            raise ValueError("candidate ids must be C01 through C06")
        if not isinstance(row["text"], str) or not row["text"].strip() or len(row["text"]) > 600:
            raise ValueError("empty or oversized text")
        for key in ("attention", "initiative"):
            if type(row[key]) is not int or row[key] not in (0, 1, 2):
                raise ValueError("invalid feature")
    if len({row["text"] for row in pool}) != 6:
        raise ValueError("duplicate candidate texts")
    return pool


def distance(candidate, profile):
    return sum(abs(candidate[k] - profile["target"][k]) for k in ("attention", "initiative"))


def select(pool, profile):
    if not pool:
        return None
    return min(pool, key=lambda c: (distance(c, profile), c["id"]))["id"]


def random_select(pool, case_id, profile_id):
    # Deterministic uniform pseudorandom selection; independent of semantic tags.
    key = f"2026110719/{case_id}/{profile_id}".encode()
    import random
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))
    return rng.choice(sorted(pool, key=lambda c: c["id"]))["id"] if pool else None


def parse_choice(text, pool):
    obj = strict_json(text)
    if not isinstance(obj, dict) or set(obj) != {"candidate_id"}:
        raise ValueError("expected candidate_id only")
    chosen = obj["candidate_id"]
    if chosen is not None and (not isinstance(chosen, str) or chosen not in {c["id"] for c in pool}):
        raise ValueError("unknown candidate")
    return chosen
