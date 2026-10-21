"""后台线程：低频、异步、不占对话链路。

负责慢层：judgments 和 self-story。
会话期间用的是旧版 self-story —— 这是对的，人也不会边说话边改写自我概念。

限幅：
  · 每次最多影响 M 条
  · 每个判断只能被推翻一次
  · 未处理完的积压进 buffer，不丢也不立刻生效
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .store import Store, append_judgment, now_iso

SELF_STORY_HEADER = """# 我如何变成现在这样

> 这个文件只记录变化，以及变化的原因。
> 覆盖会显得善变，并排会像个数据库，只有「原来 X，现在 Y，因为 Z」才像成长。

"""


@dataclass
class Consolidator:
    """把积压事件压成判断，再重写自我叙事。"""

    store: Store
    llm: Callable[[str], str] | None = None
    max_judgments_per_run: int = 3
    max_evidence: int = 8

    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    # ------------------------------------------------------------ 主逻辑

    def run_once(self) -> dict:
        events = self.store.drain_buffer(max_items=self.max_evidence)
        if not events:
            return {"judgments": 0, "self_story": False, "reason": "no events"}

        added = self._consolidate(events)
        story = self._rewrite_story(events, added) if added else False
        return {"judgments": len(added), "self_story": story}

    def _consolidate(self, events: list[dict]) -> list[dict]:
        """从事件里提炼判断。没有 llm 就走启发式，保证离线可跑。"""
        if self.llm is None:
            return self._heuristic_consolidate(events)

        prompt = self._consolidation_prompt(events)
        try:
            raw = self.llm(prompt)
            items = json.loads(_extract_json(raw))
        except Exception:
            # 后台失败不能影响任何东西，静默降级
            return self._heuristic_consolidate(events)

        existing = {j["judgment"] for j in self.store.judgments(limit=50)}
        out = []
        for it in items[: self.max_judgments_per_run]:
            txt = str(it.get("judgment", "")).strip()
            if not txt or txt in existing:
                continue
            rec = append_judgment(
                self.store.judgments_path,
                judgment=txt,
                confidence=float(it.get("confidence", 0.5)),
                supersedes=it.get("supersedes"),
            )
            out.append(rec)
        return out

    def _heuristic_consolidate(self, events: list[dict]) -> list[dict]:
        """无模型兜底：只提炼最稳的那种判断 —— 重复出现的行为倾向。"""
        counts: dict[str, int] = {}
        for e in events:
            need = (e.get("need") or "").strip()
            if need:
                counts[need] = counts.get(need, 0) + 1
        out = []
        existing = {j["judgment"] for j in self.store.judgments(limit=50)}
        for need, n in sorted(counts.items(), key=lambda x: -x[1]):
            if n < 2:
                continue                      # 出现一次不算倾向
            txt = f"这种时候更可能是「{need}」"
            if txt in existing:
                continue
            rec = append_judgment(
                self.store.judgments_path, txt,
                confidence=min(0.8, 0.4 + 0.1 * n),
            )
            out.append(rec)
            if len(out) >= self.max_judgments_per_run:
                break
        return out

    def _consolidation_prompt(self, events: list[dict]) -> str:
        lines = [
            "下面是最近几轮对话里，AI 当时读到的需求和它真正想说的话。",
            "请提炼 1-3 条关于这个人稳定倾向的判断。",
            "要求：",
            "  · 只写有重复证据的，一次不算",
            "  · 写倾向，不写事实（「倾向于先自我否定」而不是「他昨天辞职了」）",
            "  · 如果与已有判断冲突，在 supersedes 里填那条判断的 id",
            "",
            "已有判断：",
        ]
        for j in self.store.judgments(limit=10):
            lines.append(f"  [{j['id']}] {j['judgment']}（{j['confidence']}）")
        lines += ["", "最近事件："]
        for e in events:
            lines.append(f"  - 用户: {e.get('user', '')[:80]}")
            lines.append(f"    当时读成: {e.get('need', '')}")
            lines.append(f"    真正想说的: {e.get('direct', '')[:80]}")
        lines += [
            "",
            '只输出 JSON 数组，形如：',
            '[{"judgment": "...", "confidence": 0.6, "supersedes": null}]',
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------ 自传

    def _rewrite_story(self, events: list[dict], added: list[dict]) -> bool:
        """把新判断写进自传。带原因，不是覆盖。"""
        path = self.store.self_story_path
        if not path.exists():
            path.write_text(SELF_STORY_HEADER, encoding="utf-8")

        if self.llm is not None and added:
            entry = self._story_entry_llm(events, added)
        else:
            entry = self._story_entry_template(added)
        if not entry:
            return False

        with path.open("a", encoding="utf-8") as f:
            f.write(f"\n## {now_iso()}\n\n{entry}\n")
        return True

    def _story_entry_template(self, added: list[dict]) -> str:
        lines = []
        for j in added:
            lines.append(f"- 我开始觉得：{j['judgment']}（把握 {j['confidence']:.2f}）")
        if not lines:
            return ""
        lines.append("")
        lines.append("原因：这个印象在最近几轮里重复出现，不是我一次就下的结论。")
        return "\n".join(lines)

    def _story_entry_llm(self, events: list[dict], added: list[dict]) -> str:
        prev = self.store.self_story_path.read_text(encoding="utf-8")[-1500:]
        prompt = (
            "你是同一个人的内心。下面是你最近新形成的几个判断，以及你过去的自我叙述。\n"
            "请用第一人称写一小段（不超过 120 字），说明「我原来是怎样的，"
            "现在变成怎样，因为什么」。\n"
            "不要写成总结报告，要像一个人在回想。\n\n"
            f"[过去的自我叙述]\n{prev}\n\n"
            f"[新形成的判断]\n" + "\n".join(f"- {j['judgment']}" for j in added) + "\n\n"
            f"[触发的事件]\n" + "\n".join(
                f"- {e.get('user', '')[:60]}" for e in events[:5]
            )
        )
        try:
            text = self.llm(prompt).strip()
        except Exception:
            return self._story_entry_template(added)
        return text[:600]

    # ------------------------------------------------------------ 线程

    def start(self, interval_sec: float = 300.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(interval_sec):
                try:
                    self.run_once()
                except Exception:
                    pass          # 后台永不向外抛

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="persona-consolidator")
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None


def _extract_json(raw: str) -> str:
    """模型经常包一层 ```json，剥掉。"""
    s = raw.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("["):
                return p
    i, j = s.find("["), s.rfind("]")
    if i != -1 and j > i:
        return s[i:j + 1]
    return s
