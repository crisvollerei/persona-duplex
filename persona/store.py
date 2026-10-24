"""追加式账本与状态视图。

已入账的观测和变更可以恢复；未记录的进程内上下文不能凭空重建。
rebuild_views 只恢复既有结果，不会按新参数重跑模型或重新判定，不是反事实实验。
"""
from __future__ import annotations

import json
import math
import os
import threading
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        # 固化跨模型调用互斥；与前台读写锁分离，不能卡住对话。
        self.consolidation_lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._transaction: dict | None = None
        self._recovery_needed = True
        self.recover_commits()

    # ------------------------------------------------------------ 账本

    def append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        rec = {"t": now_iso(), "kind": kind, **payload}
        with self._lock:
            if self._transaction is not None:
                self._transaction["events"].append(rec)
            else:
                self._append_record(rec)
        return rec

    def _append_record(self, rec: dict) -> None:
        encoded = (json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with self.ledger_path.open("a+b") as f:
            f.seek(0, os.SEEK_END)
            if f.tell():
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    # Preserve damaged bytes, but never concatenate the next record to them.
                    f.write(b"\n")
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())

    def _raw_ledger(self) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        out = []
        with self.ledger_path.open("rb") as f:
            for line in f:
                try:
                    rec = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    def read_ledger(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for rec in self._raw_ledger():
                if rec.get("kind") == "commit":
                    out.extend(r for r in rec.get("events", []) if isinstance(r, dict))
                else:
                    out.append(rec)
            if self._transaction is not None:
                out.extend(self._transaction["events"])
            return deepcopy(out)

    @contextmanager
    def transaction(self):
        """Short local commit only; never hold this context over a model request."""
        with self._lock:
            if self._transaction is not None:
                yield
                return
            self.recover_commits()
            tx = {"t": now_iso(), "kind": "commit", "commit_id": uuid4().hex,
                  "events": [], "views": {}, "buffers": []}
            self._transaction = tx
            try:
                yield
                if not (tx["events"] or tx["views"] or tx["buffers"]):
                    return
                if tx["views"] and tx["events"]:
                    tx["events"][-1]["state_after"] = deepcopy(tx["views"])
                self._recovery_needed = True
                self._append_record(tx)
            finally:
                self._transaction = None
            self.recover_commits()

    def recover_commits(self) -> None:
        """Publish committed views/outbox after a crash; completed memory is not requeued."""
        with self._lock:
            if not self._recovery_needed or self._transaction is not None:
                return
            rows = self._raw_ledger()
            checkpoint = self._read_view("commit_checkpoint", {}).get("commit_id")
            commits = [r for r in rows if r.get("kind") == "commit"]
            consumed = {eid for r in rows for event in (
                r.get("events", []) if r.get("kind") == "commit" else [r])
                if event.get("kind") == "memory_batch_commit"
                for eid in event.get("event_ids", [])}
            start = next((i + 1 for i, r in enumerate(commits)
                          if r.get("commit_id") == checkpoint), 0)
            for tx in commits[start:]:
                for name, data in tx.get("views", {}).items():
                    self._write_view(name, data)
                for item in tx.get("buffers", []):
                    if item["event"].get("_event_id") not in consumed:
                        self._publish_buffer(item)
                if any(r.get("kind") in ("story_seed", "story_entry") for r in tx.get("events", [])):
                    self.rebuild_story()
                self._write_view("commit_checkpoint", {"commit_id": tx["commit_id"]})
            self._recovery_needed = False

    def rebuild_story(self) -> bool:
        """恢复有账本依据的叙事；没有新记录时保留旧版独立文件。"""
        parts = []
        for rec in self.read_ledger():
            if rec.get("kind") == "story_seed":
                parts = [rec.get("text", "")]
            elif rec.get("kind") == "story_entry":
                parts.append(f"\n## {rec['t']}\n\n{rec.get('entry', '')}\n")
        if not parts:
            return False
        tmp = self.self_story_path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                f.write("".join(parts))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.self_story_path)
        finally:
            tmp.unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------ ID

    # 前缀 → (账本里的记录类型, 记录里的 id 字段)。续号靠它。
    _ID_KINDS = {"t": ("turn", "turn_id"), "p": ("pending_add", "pending_id")}

    def next_id(self, prefix: str) -> str:
        """每个前缀各自编号。共用计数器会出现 t_0001 / p_0002 这种跳号。

        重启后必须从账本续号。只对 t 续号而 p 从 0 起，会让新预言复用
        已经结算过的 id —— 重放时两条不同的预言撞成一个。
        """
        with self._lock:
            if prefix not in self._counters:
                kind, key = self._ID_KINDS.get(prefix, (None, None))
                used = 0
                if kind:
                    for r in self.read_ledger():
                        if r.get("kind") != kind and not (
                                prefix == "t" and r.get("kind") == "turn_start"):
                            continue
                        try:
                            used = max(used, int(str(r.get(key, "")).rsplit("_", 1)[-1]))
                        except (TypeError, ValueError):
                            continue
                self._counters[prefix] = used
            self._counters[prefix] += 1
            return f"{prefix}_{self._counters[prefix]:04d}"

    # ------------------------------------------------------------ 视图

    def _view_path(self, name: str) -> Path:
        return self.state_dir / f"{name}.json"

    def load_view(self, name: str, default: dict | None = None) -> dict | None:
        """读一个 state 视图。

        default=None 时，文件不存在/损坏就返回 None —— 调用方靠它
        区分「这个视图确实不存在」和「存在但是空的」。
        比如 pendings() 要用这个信号触发从账本重放。
        """
        with self._lock:
            self.recover_commits()
            if self._transaction is not None and name in self._transaction["views"]:
                return deepcopy(self._transaction["views"][name])
            return self._read_view(name, default)

    def _read_view(self, name: str, default=None):
        p = self._view_path(name)
        if not p.exists():
            return None if default is None else dict(default)
        try:
            got = json.loads(p.read_text(encoding="utf-8"))
            return got if isinstance(got, dict) else (None if default is None
                                                      else dict(default))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None if default is None else dict(default)

    def save_view(self, name: str, data: dict) -> None:
        with self._lock:
            if self._transaction is not None:
                self._transaction["views"][name] = deepcopy(data)
            else:
                self.recover_commits()
                self._write_view(name, data)

    def _write_view(self, name: str, data: dict) -> None:
        p = self._view_path(name)
        tmp = p.with_suffix(f".{uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        finally:
            tmp.unlink(missing_ok=True)

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
                added: list[str] = []
                for it in items:
                    if it in cur:
                        continue
                    # max(1, ...) 防止配置写成 0 时 pop 空列表崩溃
                    if key == "unresolved" and len(cur) >= max(1, limits.unresolved_max):
                        cur.pop(0)          # 最老的先让位，别无限堆
                    cur.append(it)
                    added.append(it)
                if cur != rel.get(key, []):
                    rel[key] = cur
                    # 账本里记**新增了哪几条**，不是整个列表。
                    #
                    # 记整个列表的话，重放时会把已经被 pop 掉的项也加回来 ——
                    # 实测：6 轮后运行时 3 条（事3/4/5），
                    # 重放后变 0 条（旧代码根本不处理列表字段）或多出 6 条。
                    # 记增量 + 重放走同一套限幅，才能保证 runtime == rebuild。
                    changed[key] = added

        self.save_view("relation", rel)
        return changed

    # ------------------------------------------------------------ 挂起

    def pendings(self) -> list[dict]:
        """读挂起的预言。文件坏了就从账本重放。

        不重建的话，state/pending.json 一损坏就静默返回 []，
        而账本里 pending_add 还在 —— 预言凭空消失，没有任何提示。
        """
        data = self.load_view("pending", None)
        if data is None:
            # 文件不存在 —— 试着从账本重放
            rebuilt = replay_pendings(self.read_ledger())
            if rebuilt:
                self._save_pendings(rebuilt)
            return rebuilt
        return data.get("items", [])

    def _save_pendings(self, items: list[dict]) -> None:
        self.save_view("pending", {"items": items})

    def add_pending(self, claim: str, confidence: float, predicts: str,
                    turn_id: str, limits, turn_seq: int = 0) -> dict | None:
        items = self.pendings()
        # pending_max 是「每轮最多挂几条」，不是总数。按总数算的话，
        # 只要还有一条 WATCH 挂在 72 小时窗口里，预测链路就整个停摆。
        this_turn = sum(1 for it in items if it.get("turn_id") == turn_id)
        if this_turn >= limits.pending_max:
            return None
        total_max = getattr(limits, "pending_total_max", 8)
        if len(items) >= total_max:
            return None                  # 总闸，防止无限堆积
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

    def judgments(self, limit: int | None = 5) -> list[dict]:
        """判断记忆。从**账本**重放，不是从 jsonl 读。

        为什么要迁进账本：判断层原本是账本之外的旁路文件，
        导致「同一份账本换策略重放」这个承诺对判断层不成立 ——
        state/ 删了能重建关系，但判断永远重建不出来。

        包含用户判断和诊断记录；前台请使用 user_judgments()。
        limit=None 返回全部活跃记录。兼容旧 judgments.jsonl 的一次性导入。
        """
        ledger = self.read_ledger()
        if not any(r.get("kind", "").startswith("judgment") for r in ledger):
            self._import_legacy_judgments()
            ledger = self.read_ledger()
        records = replay_judgments(ledger)
        return records if limit is None else records[-limit:] if limit > 0 else []

    def user_judgments(self, limit: int | None = 5) -> list[dict]:
        """先排除模型读法计数，再截取供用户记忆使用的暂定判断。"""
        records = [j for j in self.judgments(limit=None) if not is_diagnostic_judgment(j)]
        return records if limit is None else records[-limit:] if limit > 0 else []

    def add_judgment(self, judgment: str, confidence: float,
                     evidence_ids: list[str] | None = None,
                     supersedes: str | None = None,
                     affects: dict | None = None, source: str = "explicit",
                     batch_id: str = "", evidence_scope: str = "specified") -> dict:
        """写一条判断。落账本。

        affects 是「这条判断指向哪个人格参数、什么方向」：
            {"intervention": -1}  表示「他不想被建议」-> 压低 intervention
        中期层靠它把多条判断聚合成一次参数推动。
        没有 affects 的判断只进 prompt，不影响人格。
        """
        jid = self.next_judgment_id()
        # confidence 必须夹到 0..1。不夹的话，一条 conf=5.0 的判断
        # 单独就能越过 min_consensus=1.4 的共识闸门 ——
        # 实测「孤证」一条就能推动参数。
        try:
            conf = float(confidence)
            conf = max(0.0, min(1.0, conf)) if math.isfinite(conf) else 0.5
        except (TypeError, ValueError):
            conf = 0.5

        # supersedes 必须指向一条**当前活跃**的判断。
        # 不校验的话：指向不存在的 id 会静默悬空（谁都没被推翻，也不报错）；
        # 低置信判断能推翻高置信的（0.2 推翻 0.9）。
        if supersedes:
            live = {j["id"]: j for j in replay_judgments(self.read_ledger())}
            target = live.get(str(supersedes))
            if target is None:
                supersedes = None          # 悬空 -> 当作不推翻
            elif is_diagnostic_judgment(target) != is_diagnostic_judgment({
                    "source": source, "evidence_scope": evidence_scope}):
                supersedes = None
            elif float(target.get("confidence", 0.0)) > conf + 0.2:
                # 置信度差太多，新证据不足以推翻旧的
                supersedes = None
        self.append("judgment_add", {
            "judgment_id": jid,
            "judgment": judgment,
            "confidence": round(conf, 3),
            "evidence_ids": evidence_ids or [],
            "supersedes": supersedes,
            "affects": dict(affects or {}),
            "source": source, "batch_id": batch_id, "evidence_scope": evidence_scope,
        })
        return {"id": jid, "judgment": judgment,
                "confidence": round(conf, 3),
                "evidence_ids": evidence_ids or [],
                "supersedes": supersedes, "affects": dict(affects or {}),
                "source": source, "batch_id": batch_id, "evidence_scope": evidence_scope,
                "superseded_by": None}

    def next_judgment_id(self) -> str:
        with self._lock:
            used = 0
            for r in self.read_ledger():
                if r.get("kind") != "judgment_add":
                    continue
                try:
                    used = max(used, int(str(r.get("judgment_id", "")).rsplit("_", 1)[-1]))
                except (TypeError, ValueError):
                    continue
            self._counters["j"] = max(self._counters.get("j", 0), used) + 1
            return f"j_{self._counters['j']:04d}"

    def _import_legacy_judgments(self) -> int:
        """把老的 judgments.jsonl 导进账本，只导一次。

        不删原文件 —— 它是迁移前的唯一副本，留着当保险。
        导入后账本里就有 judgment_add 记录，下次不会再走这条路。
        """
        if not self.judgments_path.exists():
            return 0
        n = 0
        try:
            for line in self.judgments_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                jid = str(rec.get("id") or f"j_import_{n:04d}")
                self.append("judgment_add", {
                    "judgment_id": jid,
                    "judgment": rec.get("judgment", ""),
                    "confidence": rec.get("confidence", 0.5),
                    "evidence_ids": rec.get("evidence_ids", []),
                    "supersedes": rec.get("supersedes"),
                    "affects": rec.get("affects", {}),
                    "source": rec.get("source", "legacy_unknown"),
                    "batch_id": rec.get("batch_id", ""),
                    "evidence_scope": rec.get("evidence_scope", "unknown"),
                    "imported": True,
                })
                n += 1
        except OSError:
            return 0
        return n

    # ------------------------------------------------------------ 人格

    def traits(self) -> dict:
        """人格参数的当前状态。从账本重放。

        和 judgments 同样的道理：人格也是记忆的一种形态，
        必须能从账本算出来，否则「同一份账本换策略重放」就不成立。

        账本里没有 traits_init 时返回空 —— 由 Session 用 persona.toml 初始化。
        """
        return replay_traits(self.read_ledger())

    def init_traits(self, state: dict, *, persona: str = "") -> None:
        with self.transaction():
            self._init_traits(state, persona=persona)

    def _init_traits(self, state: dict, *, persona: str = "") -> None:
        """首次载入时把 persona.toml 的参数快照进账本。

        只写一次。之后所有变化都靠 trait_update / anchor_update 记录，
        重放时从这份快照往后推 —— 不依赖 toml 的当前内容。

        **persona 参数是必需的**：同一个 data 目录换人格时必须重新初始化，
        否则新人设会拿到旧人设的参数。

        实测踩过：守岸人（warmth anchor 0.70）跑过之后，
        用同一目录载入 ache（warmth anchor 0.30），
        实际拿到 0.45 —— 被 ache 的 drift 范围夹住的守岸人值。
        人格整个串了，而且看不出来。
        """
        existing = [r for r in self.read_ledger() if r.get("kind") == "traits_init"]
        if existing:
            prev = self.persona_name()
            # 老账本没有 persona 字段（这个字段是后加的）。
            # 那种情况**不能**当作「换了人格」—— 否则升级一次就把所有
            # 历史人格漂移清零。实测：warmth 0.80 升级后变回 0.70。
            # 判据：persona 为空 = 老账本，认作「同一个人格」，保留状态。
            if prev == "" or not persona or prev == persona:
                if prev == "" and persona:
                    # 顺便补上人格名，下次就能正确判断了
                    self.append("traits_adopt", {"persona": persona})
                return
            # 换了人格 —— 之前的 traits 状态对新人格没有意义
            self.append("traits_reset", {"from": prev, "to": persona})
        self.append("traits_init", {"traits": state, "persona": persona})

    def persona_name(self) -> str:
        """账本里记录的人格名。空串表示没有初始化过。

        老账本的 traits_init 没有 persona 字段；如果后来用
        traits_adopt 补记过，以补记的为准。
        """
        name = ""
        for r in self.read_ledger():
            k = r.get("kind")
            if k == "traits_init":
                name = str(r.get("persona", "") or "") or name
            elif k == "traits_adopt":
                name = str(r.get("persona", "") or "") or name
            elif k == "traits_reset":
                name = str(r.get("to", "") or "")
        return name

    def update_trait(self, name: str, delta: float, *,
                     reason: str = "", source: str = "consolidate") -> dict | None:
        """记一次参数推动。返回变更明细，没有实际变化则返回 None。

        只记录**变化量**，不记录结果值 —— 和 relation.closeness 一样的道理。
        重放时把 delta 累加上去，才能保证同一份账本算出同一个结果。
        """
        if not delta:
            return None
        self.append("trait_update", {
            "name": name, "delta": round(float(delta), 6),
            "reason": reason[:200], "source": source,
        })
        return {"name": name, "delta": round(float(delta), 6), "reason": reason}

    def update_anchor(self, name: str, delta: float, *,
                      reason: str = "") -> dict | None:
        """记一次基线推动。长期层专用。

        **必须有 reason** —— 改基线是一个人格级的事件，
        没有叙事依据的改动应该被拒绝。
        """
        if not delta:
            return None
        if not (reason or "").strip():
            return None
        self.append("anchor_update", {
            "name": name, "delta": round(float(delta), 6),
            "reason": reason[:400],
        })
        return {"name": name, "delta": round(float(delta), 6), "reason": reason}

    # ------------------------------------------------------------ 游标

    def cursor(self, name: str, key: str = "ids") -> object:
        """读一个游标（已消费的判断 id / 叙事读到的位置）。

        游标必须能从账本重放 —— 它和 relation / traits 一样是跨轮累积量。
        只存在 state/ 里的话，删掉 state 重建后同一批判断会被二次推动、
        同一段叙事会被二次消费。实测：
            judgment_cursor 丢失后 intervention 从 0.7036 又推到 0.7072
            story_cursor 丢失后 anchor 从 0.706 又推到 0.712
        """
        value = [] if key == "ids" else 0
        for r in self.read_ledger():
            if r.get("kind") != f"{name}_cursor":
                continue
            value = r.get(key, value)
        return value

    def set_cursor(self, name: str, value, key: str = "ids") -> None:
        """写游标。落账本，可重放。"""
        self.append(f"{name}_cursor", {key: value})

    # ------------------------------------------------------------ 缓冲

    def push_buffer(self, rec: dict) -> None:
        """写一条待固化的积压。

        文件名必须**唯一**。原来只用微秒时间戳，同一微秒内连续写入会
        互相覆盖 —— 实测 10 条只存下 5 条。真实对话碰不到（一轮要几秒），
        但离线批量补跑、测试、脚本回放都会触发，而且是静默丢数据。

        使用稳定事件 ID 和唯一文件名，提交恢复时可避免重复入队。
        """
        with self._lock:
            stamp = datetime.now(_LOCAL_TZ).strftime("%Y%m%d%H%M%S%f")
            eid = rec.get("_event_id") or uuid4().hex
            item = {"filename": f"{stamp}_{eid}.json", "event": {**rec, "_event_id": eid}}
            if self._transaction is not None:
                self._transaction["buffers"].append(item)
            else:
                self._publish_buffer(item)

    def _publish_buffer(self, item: dict) -> None:
        p = self.buffer_dir / item["filename"]
        if p.exists() or (self.processing_dir / p.name).exists():
            return
        tmp = p.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(item["event"], f, ensure_ascii=False, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        finally:
            tmp.unlink(missing_ok=True)

    @property
    def processing_dir(self) -> Path:
        d = self.buffer_dir / "processing"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def recover_buffer(self) -> None:
        """在固化锁内恢复上次崩溃留下的暂存事件，再进行本批分档。"""
        self._drained = [p.name for p in self.processing_dir.glob("*.json")]
        self.restore_drained()
        consumed = {eid for r in self.read_ledger() if r.get("kind") == "memory_batch_commit"
                    for eid in r.get("event_ids", [])}
        for p in self.buffer_dir.glob("*.json"):
            try:
                event = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(event, dict):
                    raise ValueError("Buffer event must be an object")
                if event.get("_event_id", p.stem) in consumed:
                    p.unlink()
            except ValueError as exc:
                self._quarantine_buffer(p, exc)
            except OSError:
                continue

    def _quarantine_buffer(self, path: Path, error: Exception) -> None:
        self.append("buffer_error", {"file": path.name, "error_type": type(error).__name__})
        target = path.with_suffix(".invalid")
        if target.exists():
            target = path.with_suffix(f".{uuid4().hex}.invalid")
        path.rename(target)

    def drain_buffer(self, max_items: int = 20) -> list[dict]:
        """把积压的事件取出来。

        **不直接删除** —— 先移到 buffer/processing/，等调用方确认处理完
        再调 commit_drained()。处理失败就调 restore_drained() 放回去。
        """
        out, moved = [], []
        for p in sorted(self.buffer_dir.glob("*.json"))[:max_items]:
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(rec, dict):
                    raise ValueError("Buffer event must be an object")
            except ValueError as exc:
                self._quarantine_buffer(p, exc)
                continue
            except OSError:
                continue
            rec.setdefault("_event_id", p.stem)
            try:
                p.rename(self.processing_dir / p.name)
                moved.append(p.name)
            except OSError:
                continue
            out.append(rec)
        self._drained = moved
        return out

    def commit_drained(self) -> None:
        """处理成功，把 staging 里的删掉。"""
        for name in getattr(self, "_drained", []):
            (self.processing_dir / name).unlink(missing_ok=True)
        self._drained = []

    def restore_drained(self) -> None:
        """处理失败，把事件放回 buffer。

        不这么做的话，drain 之后任何异常都会让那批对话永久丢失 ——
        实测 OSError 之后 buffer 3 -> 0，事件没了，判断也没产出。
        """
        for name in getattr(self, "_drained", []):
            src = self.processing_dir / name
            if src.exists():
                try:
                    dst = self.buffer_dir / name
                    if dst.exists():
                        dst = self.buffer_dir / f"{src.stem}_recovered_{uuid4().hex}.json"
                    src.rename(dst)
                except OSError:
                    pass
        self._drained = []


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


def replay_anchor_pushes(ledger: list[dict]) -> list[str]:
    """频率状态也能恢复；旧版每次 anchor_update 的时间作保守回退。"""
    times = [r.get("at") or r.get("t") for r in ledger if r.get("kind") == "anchor_push"]
    times = [t for t in times if isinstance(t, str) and t]
    if not times:
        times = sorted({r["t"] for r in ledger
                        if r.get("kind") == "anchor_update" and isinstance(r.get("t"), str)})
    return times


def replay_pendings(ledger: list[dict]) -> list[dict]:
    """从账本重放出「还在挂着的预言」。

    pending_add 之后没有对应 pending_settle / pending_expire 的，
    就是还挂着的。用来在 state/pending.json 损坏时重建。
    """
    live: dict[str, dict] = {}
    for r in ledger:
        k = r.get("kind")
        if k == "pending_add":
            pid = r.get("pending_id")
            if pid:
                live[str(pid)] = {
                    "id": str(pid),
                    "claim": r.get("claim", ""),
                    "confidence": r.get("confidence", 0.0),
                    "predicts": r.get("predicts", ""),
                    "created": r.get("t", ""),
                    "turn_id": r.get("turn_id", ""),
                    "turn_seq": r.get("turn_seq", 0),
                }
        elif k in ("pending_settle", "pending_expire"):
            live.pop(str(r.get("pending_id") or ""), None)
    return list(live.values())


def replay_traits(ledger: list[dict]) -> dict:
    """从账本重放出人格参数状态。

    结构是 {name: {"anchor", "value", "plasticity", "drift", "lo", "hi"}}，
    和 Trait.to_state() 的形状一致。

    重放顺序：
        1. traits_init  给出初始快照
        2. trait_update 累加 value 的 delta（并被 drift 夹住）
        3. anchor_update 改 anchor，**value 跟着平移**（保持相对位置）
    """
    state: dict[str, dict] = {}
    for rec in ledger:
        k = rec.get("kind")
        if k == "traits_reset":
            # 换人格了 —— 之前的参数状态作废，等下一个 traits_init
            state.clear()
        elif k == "traits_init":
            for name, d in (rec.get("traits") or {}).items():
                if isinstance(d, dict):
                    state[name] = dict(d)
        elif k == "trait_update":
            name = rec.get("name")
            if name not in state:
                continue
            cur = state[name]
            cur["value"] = _clamp_trait(
                float(cur.get("value", cur.get("anchor", 0.0)))
                + float(rec.get("delta", 0.0)), cur)
        elif k == "anchor_update":
            name = rec.get("name")
            if name not in state:
                continue
            cur = state[name]
            delta = float(rec.get("delta", 0.0))
            cur["anchor"] = float(cur.get("anchor", 0.0)) + delta
            # value 跟着平移，保持相对位置 —— 否则 anchor 一动，
            # value 可能立刻越界被夹住，看起来像人格突变。
            from .config import Trait
            moved = Trait.from_state({**cur, "value": float(cur.get("value", 0.0)) + delta})
            cur["anchor"], cur["value"] = moved.anchor, moved.value
    return state


def _clamp_trait(value: float, spec: dict) -> float:
    """把值夹进合法范围。

    **必须复用 Trait.clamp**，不能另写一份 —— 两份实现会分叉。
    实测踩过：配置矛盾（lo=0.8 hi=0.2）时
        replay_traits 算 0.8，Trait.from_state 算 0.5。
    而「重放 == 运行时」是这个项目的核心承诺。
    """
    from .config import Trait
    t = Trait.from_state({**spec, "value": value})
    return t.value


def is_diagnostic_judgment(record: dict) -> bool:
    """旧启发式记录也只证明模型反复这样读，不能作为用户倾向。"""
    return (record.get("source") in ("heuristic", "heuristic_fallback")
            or record.get("evidence_scope") == "need_count")


def replay_judgments(ledger: list[dict]) -> list[dict]:
    """从账本重放出判断记忆。

    规则和原来一样：被 supersedes 引用的判断算「已推翻」，
    不进活跃列表。区别只在于数据来源是账本。

    返回**全部活跃判断**（不截断）—— 截断是调用方的事，
    因为中期层需要看到完整列表才能判断「多条一致」。
    """
    recs: list[dict] = []
    for r in ledger:
        if r.get("kind") != "judgment_add":
            continue
        recs.append({
            "id": r.get("judgment_id", ""),
            "judgment": r.get("judgment", ""),
            "confidence": float(r.get("confidence", 0.0)),
            "evidence_ids": list(r.get("evidence_ids") or []),
            "supersedes": r.get("supersedes"),
            "superseded_by": None,
            "affects": dict(r.get("affects") or {}),
            "source": r.get("source", "legacy_unknown"),
            "batch_id": r.get("batch_id", ""),
            "evidence_scope": r.get("evidence_scope", "unknown"),
            "t": r.get("t", ""),
        })
    superseded: dict[str, str] = {}
    for r in recs:
        if r.get("supersedes"):
            superseded[r["supersedes"]] = r["id"]
    for r in recs:
        if r["id"] in superseded:
            r["superseded_by"] = superseded[r["id"]]
    return [r for r in recs if not r.get("superseded_by")]


def load_judgments(path: Path, limit: int = 5) -> list[dict]:
    """兼容旧接口：从 jsonl 读。**新代码不要用**，走 Store.judgments()。"""
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

def rebuild_views(store: "Store", limits=None) -> dict:
    """从账本重放，重建 state/。

    只恢复已记录的结果，不会按新参数重新运行模型或决策，不是反事实实验。

    新记录以实际提交的 state_after 快照为准；旧记录优先使用随事件记录的 limits，
    缺失时才使用显式 limits 或默认值。这不是换参数重做决策的入口。

    要重放的**全部**在线状态都得在这里：
      relation  —— 关系
      pending   —— 挂起的预言
      repair    —— 修复账本（否则受伤之后就永远不再修复了）
      style     —— 语域基线（否则「受伤」的尺子会退回默认值）
    """
    from .config import Limits as _Limits
    unresolved_max = int(getattr(limits or _Limits(), "unresolved_max", 3))
    rel = {"closeness": 0.3, "unresolved": [], "sore_points": [], "inside_jokes": []}
    pend: dict[str, dict] = {}
    repairs: list[dict] = []
    style: dict = {}

    # 老账本（加 turn_seq 之前）的 pending_add 里没有这个字段。
    # 但 turn_id 的序号就是当时的轮次 —— 可以无损推回来。
    seq_by_turn: dict[str, int] = {}
    ledger = store.read_ledger()
    for rec in ledger:
        if rec.get("kind") == "turn":
            try:
                seq_by_turn[rec.get("turn_id", "")] = int(
                    str(rec.get("turn_id", "")).rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                continue

    def _repair(note: str, **kw) -> dict:
        for r in repairs:
            if r.get("note") == note:
                r.update(kw)
                return r
        r = {"note": note, "turn_seq": 0, "level": "plain", "attempts": 0,
             "state": "open", "created": ""}
        r.update(kw)
        repairs.append(r)
        return r

    for rec in ledger:
        k = rec.get("kind")
        if limits is None and isinstance(rec.get("limits"), dict):
            unresolved_max = int(rec["limits"].get("unresolved_max", unresolved_max))
        if k == "turn":
            for key, val in (rec.get("relation_delta") or {}).items():
                if key == "closeness":
                    rel["closeness"] = round(
                        max(0.0, min(1.0, rel["closeness"] + float(val))), 3
                    )
                elif key in ("unresolved", "sore_points", "inside_jokes"):
                    # 账本里记的是「新增项」。重放时要用和运行时**同一套限幅**，
                    # 否则 pop(0) 淘汰掉的项会在重建后重新出现。
                    items = val if isinstance(val, list) else [val]
                    for it in items:
                        if it in rel[key]:
                            continue
                        if key == "unresolved" and len(rel[key]) >= max(1, unresolved_max):
                            rel[key].pop(0)
                        rel[key].append(it)
        elif k == "style_update":
            if isinstance(rec.get("formality"), (int, float)):
                style["formality"] = rec["formality"]
            if isinstance(rec.get("length"), (int, float)):
                style["length"] = rec["length"]
            n = rec.get("n")
            if isinstance(n, int):
                hist = (style.get("length_hist", []) + [n])[-20:]
                style["length_hist"] = hist
        elif k == "pending_add":
            pend[rec["pending_id"]] = {
                "id": rec["pending_id"], "claim": rec["claim"],
                "confidence": rec["confidence"], "predicts": rec["predicts"],
                "created": rec["t"], "turn_id": rec.get("turn_id", ""),
                "turn_seq": rec.get("turn_seq")
                            or seq_by_turn.get(rec.get("turn_id", ""), 0),
            }
        elif k in ("pending_settle", "pending_expire"):
            pend.pop(rec["pending_id"], None)
        elif k == "repair_open":
            _repair(rec["note"], level=rec.get("level", "plain"),
                    state="open", created=rec.get("t", ""),
                    turn_seq=int(rec.get("turn_seq", 0) or 0))
            if rec["note"] not in rel["unresolved"]:
                if len(rel["unresolved"]) >= max(1, unresolved_max):
                    rel["unresolved"].pop(0)
                rel["unresolved"].append(rec["note"])
        elif k == "repair_attempt":
            r = _repair(rec["note"])
            r["state"] = "repairing"
            r["attempts"] = int(r.get("attempts", 0)) + 1
            r["level"] = rec.get("level", r.get("level", "plain"))
        elif k == "repair_healed":
            _repair(rec["note"], state="healed")
            rel["unresolved"] = [u for u in rel["unresolved"] if u != rec["note"]]
        elif k in ("repair_rejected", "repair_residual"):
            _repair(rec["note"], state="residual")
            rel["unresolved"] = [u for u in rel["unresolved"] if u != rec["note"]]
            if rec["note"] not in rel["sore_points"]:
                rel["sore_points"].append(rec["note"])
        elif k == "repair_paused":
            _repair(rec["note"], state="paused")
            rel["unresolved"] = [u for u in rel["unresolved"] if u != rec["note"]]

        snapshot = rec.get("state_after") or {}
        if "relation" in snapshot:
            rel = deepcopy(snapshot["relation"])
        if "pending" in snapshot:
            pend = {p["id"]: deepcopy(p) for p in snapshot["pending"]["items"]}
        if "repair" in snapshot:
            repairs = deepcopy(snapshot["repair"]["items"])
        if "style" in snapshot:
            style = deepcopy(snapshot["style"])
        if "runtime" in snapshot:
            store.save_view("runtime", snapshot["runtime"])

    store.save_view("relation", rel)
    store.save_view("pending", {"items": list(pend.values())})
    # 判断记忆也在账本里，一并重放。
    store.save_view("judgments", {"items": replay_judgments(ledger)})
    # 游标同样 —— 它们决定「哪些判断已经用过」，不重放会二次消费。
    for name, key in (("judgment", "ids"), ("story", "offset")):
        value = store.cursor(name, key)
        store.save_view(f"{name}_cursor", {key: value})
    # 人格参数同样。
    store.save_view("traits", {"traits": replay_traits(ledger)})
    store.save_view("anchor_pushes", {"times": replay_anchor_pushes(ledger)})
    # repair 从第一版就在写 repair_open，所以空列表就是真空 —— 一律覆盖。
    store.save_view("repair", {"items": repairs})
    store.rebuild_story()
    # style_update 是后加的：老账本里没有，此时保留现有基线而不是抹成空。
    if style:
        store.save_view("style", style)
    return {"relation": rel, "pending": len(pend),
            "judgments": len(replay_judgments(ledger)),
            "traits": len(replay_traits(ledger)),
            "repair": len([r for r in repairs if r.get("state") in ("open", "repairing")]),
            "repairs_total": len(repairs)}
