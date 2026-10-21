"""账本 + 物化视图。

ledger.jsonl 是唯一真相源：只追加，不改写。
state/ 是从账本重放出来的视图，随时可删可重建。

好处不只是「变化有账」字面成立，还在于可以拿同一份账本、
不同版本的代码重放，对比参数策略 —— 调 bias 和位移率时非常有用。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_LOCAL_TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(_LOCAL_TZ).isoformat(timespec="seconds")


def parse_iso(s: str) -> datetime:
    try:
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return datetime.now(_LOCAL_TZ)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_LOCAL_TZ)
    return dt


class Store:
    """线程安全的最小持久层。

    单进程够用。要跨进程再换 sqlite，接口不变。
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.state_dir = self.root / "state"
        self.buffer_dir = self.root / "buffer"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(exist_ok=True)
        self.buffer_dir.mkdir(exist_ok=True)

        self.ledger_path = self.root / "ledger.jsonl"
        self.judgments_path = self.root / "judgments.jsonl"
        self.self_story_path = self.root / "self-story.md"

        self._lock = threading.RLock()
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------ 账本

    def append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        rec = {"t": now_iso(), "kind": kind, **payload}
        with self._lock:
            with self.ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return rec

    def read_ledger(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        out = []
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # 半行 = 上次写到一半被杀了。跳过，不让它毒化重放。
                    continue
        return out

    # ------------------------------------------------------------ ID

    def next_id(self, prefix: str) -> str:
        """每个前缀各自编号。共用计数器会出现 t_0001 / p_0002 这种跳号。"""
        with self._lock:
            if prefix not in self._counters:
                self._counters[prefix] = sum(
                    1 for r in self.read_ledger() if r.get("kind") == "turn"
                ) if prefix == "t" else 0
            self._counters[prefix] += 1
            return f"{prefix}_{self._counters[prefix]:04d}"

    # ------------------------------------------------------------ 视图

    def _view_path(self, name: str) -> Path:
        return self.state_dir / f"{name}.json"

    def load_view(self, name: str, default: dict | None = None) -> dict:
        p = self._view_path(name)
        if not p.exists():
            return dict(default or {})
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return dict(default or {})

    def save_view(self, name: str, data: dict) -> None:
        p = self._view_path(name)
        tmp = p.with_suffix(".tmp")
        with self._lock:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, p)

    # ------------------------------------------------------------ 关系

    def relation(self) -> dict:
        return self.load_view(
            "relation",
            {"closeness": 0.3, "unresolved": [], "sore_points": [], "inside_jokes": []},
        )

    def apply_relation_delta(self, delta: dict, limits) -> dict:
        """只允许白名单内的字段，且限额生效。"""
        rel = self.relation()
        changed: dict[str, Any] = {}

        if "closeness" in delta:
            try:
                step = float(delta["closeness"])
            except (TypeError, ValueError):
                step = 0.0
            step = max(-limits.closeness_step, min(limits.closeness_step, step))
            if step:
                before = rel.get("closeness", 0.0)
                new = max(0.0, min(1.0, before + step))
                if new != before:
                    rel["closeness"] = round(new, 3)
                    # 账本里记的是「增量」，不是结果值。
                    # 否则重放时会把结果当成增量再加一遍。
                    changed["closeness"] = round(rel["closeness"] - before, 3)

        for key in ("unresolved", "sore_points", "inside_jokes"):
            if key in delta:
                val = delta[key]
                items = val if isinstance(val, list) else [str(val)]
                items = [str(x).strip() for x in items if str(x).strip()]
                cur = list(rel.get(key, []))
                for it in items:
                    if it in cur:
                        continue
                    if key == "unresolved" and len(cur) >= limits.unresolved_max:
                        cur.pop(0)          # 最老的先让位，别无限堆
                    cur.append(it)
                if cur != rel.get(key, []):
                    rel[key] = cur
                    changed[key] = cur

        self.save_view("relation", rel)
        return changed

    # ------------------------------------------------------------ 挂起

    def pendings(self) -> list[dict]:
        return self.load_view("pending", {"items": []}).get("items", [])

    def _save_pendings(self, items: list[dict]) -> None:
        self.save_view("pending", {"items": items})

    def add_pending(self, claim: str, confidence: float, predicts: str,
                    turn_id: str, limits, turn_seq: int = 0) -> dict | None:
        items = self.pendings()
        if len(items) >= limits.pending_max:
            return None                  # 满了就不挂，别堆
        p = {
            "id": self.next_id("p"),
            "claim": claim,
            "confidence": round(float(confidence), 3),
            "predicts": predicts,
            "created": now_iso(),
            "turn_id": turn_id,
            "turn_seq": turn_seq,
        }
        items.append(p)
        self._save_pendings(items)
        return p

    def close_pending(self, pid: str, outcome: str, evidence: str = "",
                      delayed: bool = False) -> dict | None:
        items = self.pendings()
        for i, p in enumerate(items):
            if p["id"] == pid:
                items.pop(i)
                self._save_pendings(items)
                return {**p, "outcome": outcome, "evidence": evidence,
                        "delayed": delayed, "settled": now_iso()}
        return None

    def sweep_expired(self, ttl_hours: float) -> list[dict]:
        """超时的不算证实也不算证伪，直接丢。"""
        cutoff = datetime.now(_LOCAL_TZ) - timedelta(hours=ttl_hours)
        items = self.pendings()
        keep, expired = [], []
        for p in items:
            if parse_iso(p.get("created", "")) < cutoff:
                expired.append(p)
            else:
                keep.append(p)
        if expired:
            self._save_pendings(keep)
        return expired

    # ------------------------------------------------------------ 判断

    def judgments(self, limit: int = 5) -> list[dict]:
        return load_judgments(self.judgments_path, limit=limit)

    # ------------------------------------------------------------ 缓冲

    def push_buffer(self, rec: dict) -> None:
        p = self.buffer_dir / f"{datetime.now(_LOCAL_TZ).strftime('%Y%m%d%H%M%S%f')}.json"
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")

    def drain_buffer(self, max_items: int = 20) -> list[dict]:
        out = []
        for p in sorted(self.buffer_dir.glob("*.json"))[:max_items]:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
            p.unlink(missing_ok=True)
        return out


# ---------------------------------------------------------------- 判断记忆

def append_judgment(path: Path, judgment: str, confidence: float,
                    evidence_ids: list[str] | None = None,
                    supersedes: str | None = None) -> dict:
    rec = {
        "t": now_iso(),
        "id": f"j_{int(datetime.now(_LOCAL_TZ).timestamp() * 1000) % 10**9}",
        "judgment": judgment,
        "confidence": round(float(confidence), 3),
        "evidence_ids": evidence_ids or [],
        "supersedes": supersedes,
        "superseded_by": None,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_judgments(path: Path, limit: int = 5) -> list[dict]:
    if not path.exists():
        return []
    recs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # 标记被推翻的
    superseded: dict[str, str] = {}
    for r in recs:
        if r.get("supersedes"):
            superseded[r["supersedes"]] = r["id"]
    for r in recs:
        if r["id"] in superseded:
            r["superseded_by"] = superseded[r["id"]]
    live = [r for r in recs if not r.get("superseded_by")]
    return live[-limit:]


# ---------------------------------------------------------------- 重放

def rebuild_views(store: "Store") -> dict:
    """从账本重放，重建 state/。

    state 坏了就用它。也是「同一份账本换策略重放」的入口。
    """
    rel = {"closeness": 0.3, "unresolved": [], "sore_points": [], "inside_jokes": []}
    pend: dict[str, dict] = {}

    for rec in store.read_ledger():
        k = rec.get("kind")
        if k == "turn":
            for key, val in (rec.get("relation_delta") or {}).items():
                if key == "closeness":
                    rel["closeness"] = round(
                        max(0.0, min(1.0, rel["closeness"] + float(val))), 3
                    )
                elif key in ("unresolved", "sore_points", "inside_jokes"):
                    for it in (val if isinstance(val, list) else [val]):
                        if it not in rel[key]:
                            rel[key].append(it)
        elif k == "pending_add":
            pend[rec["pending_id"]] = {
                "id": rec["pending_id"], "claim": rec["claim"],
                "confidence": rec["confidence"], "predicts": rec["predicts"],
                "created": rec["t"], "turn_id": rec.get("turn_id", ""),
            }
        elif k == "pending_settle":
            pend.pop(rec["pending_id"], None)
        elif k == "pending_expire":
            pend.pop(rec["pending_id"], None)

    store.save_view("relation", rel)
    store.save_view("pending", {"items": list(pend.values())})
    return {"relation": rel, "pending": len(pend)}
