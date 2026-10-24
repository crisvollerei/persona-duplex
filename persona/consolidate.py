"""后台提炼用户的暂定倾向和角色的陪伴记录。

常规固化不把用户判断、预测命中率或角色自述当作人格成长依据。
不自动推 traits.value / anchor；push_traits、push_anchors 仅保留为显式实验接口，
其旧式映射策略未经效果验证。人格允许未来成长，但信号与策略仍待研究。
失败时保留积压，已发生的记忆与显式实验变更落账以便恢复。
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .store import Store, is_diagnostic_judgment, now_iso, parse_iso as _parse_dt
from .audit import fingerprint, model_info, traced

# 现在是什么时候。命名成 _now_dt 是为了和 store.now_iso 区分：
# 后者返回字符串（写账本用），这个返回 datetime（算时间差用）。
from .store import now_iso as _now_iso_str


def _now_dt():
    return _parse_dt(_now_iso_str())

SELF_STORY_HEADER = """# 人格的陪伴记录

> 「我」指记录中的人格，不是用户或模型的现实身份。记录有依据的认识，不强求每次都发生变化。
> 这些文字不是已验证的成长证据，默认不据此自动修改人格参数或基线。

"""


# 活跃度分档。判断容量和「认一条倾向需要几次重复」都随它变。
#
# 为什么密集时要更高的触发阈值：
#   短时间大量对话里，同一个人会反复说同一件事（情绪宣泄）。
#   那不是「稳定倾向」，是「这一阵子的状态」。
#   要求更多次重复，才能把状态和倾向分开。
#
# 为什么稀疏时用下限：
#   隔几天聊几句的人，一次重复就已经很说明问题了 ——
#   他本就没多少话，能反复提到的必然重要。
ACTIVITY_DENSE_HOURS = 1.0        # <= 1h 两次固化之间 -> 密集
ACTIVITY_SPARSE_HOURS = 24.0      # > 24h -> 稀疏

ACTIVITY_PROFILES = {
    "dense":  {"capacity": 12, "threshold": 5},
    "medium": {"capacity": 8,  "threshold": 3},
    "sparse": {"capacity": 5,  "threshold": 2},
}


def activity_of(events: list[dict], *, gap_hours: float | None = None) -> str:
    """把一个时间间隔映射成活跃度档位（dense / medium / sparse）。

    优先用显式给的 gap_hours，没有就用事件内部的时间跨度。
    都没有时**按 sparse（最宽松）处理**，不是 medium。理由：
      · 这是保守选择 —— 阈值低，更容易形成判断，不会把该记的漏掉
      · 没有时间信息通常意味着「离线补跑」或「首次导入」，
        这种场景本来就该把已有证据都用上
      · 默认 medium 会让「两条事件的离线测试」永远出不了判断，
        这不是设计意图，是 bug

    **线上路径现在不经过这里。**
    Consolidator.profile() 判活跃度用的是「本批积压了多少轮」，
    不是时间 —— 试过三种时间口径（距上次固化 / 距最后一轮 /
    本批首末轮跨度），全都有问题，见 profile() 的说明。

    保留这个函数是因为它提供「给定间隔 -> 档位」的映射，
    而且 gap_hours 显式传入时（测试、离线补跑）确实还走这条路。
    """
    if gap_hours is not None:
        if gap_hours <= ACTIVITY_DENSE_HOURS:
            return "dense"
        if gap_hours > ACTIVITY_SPARSE_HOURS:
            return "sparse"
        return "medium"

    stamps = [e.get("_t") for e in events if e.get("_t")]
    if len(stamps) >= 2:
        from .store import parse_iso
        try:
            span = (parse_iso(stamps[-1]) - parse_iso(stamps[0])).total_seconds() / 3600
        except Exception:
            return "sparse"
        if span <= ACTIVITY_DENSE_HOURS:
            return "dense"
        if span > ACTIVITY_SPARSE_HOURS:
            return "sparse"
        return "medium"
    return "sparse"


@dataclass
class Consolidator:
    """把积压事件压成判断，再重写自我叙事。

    llm 是「一段文本 → 一段文本」的函数，跟主对话的 messages 接口不同。
    传 openai_compatible(...) 的话用 text_prompt() 包一层。

    max_judgments_per_run / min_repeat 是**上限和下限**，实际值由活跃度决定。
    """

    store: Store
    llm: Callable[[str], str] | None = None
    max_judgments_per_run: int = 3      # 单次硬上限（防止一次写太多）
    max_evidence: int = 8
    # 长期层单次推基线的**基础**幅度：drift 的这个比例。
    # 实际步长还要乘两个频率因子（见 _anchor_frequency），
    # 所以 0.05 只是「时间充裕时的满额步长」。
    ANCHOR_FRACTION: float = 0.05

    # ---- 基线推动的频率因子 ----
    #
    # 为什么需要它：没有衰减时，连续 20 次固化就能把 anchor 推到 drift 边界
    # （实测 warmth 从 0.70 到 0.82）。后台间隔 300 秒，也就是**一次两小时的
    # 对话就能把人格基线推到极限** —— 和「长期层是 100+ 轮的慢变」完全相反。
    #
    # 用两层衰减，不用硬阈值：硬阈值（「24 小时内只推一次」）会让行为在
    # 边界处跳变，而且完全阻断密集期的累积。衰减是连续的 ——
    # 刚推过就几乎推不动，隔久了就满额，和余弦退火是同一个思路。

    # 因子 1 的时间尺度：距上次推动多久才能拿到满额。
    # 6 小时意味着：刚推过 -> 0，6 小时后 -> 63%，一天后 -> 98%。
    ANCHOR_TAU_HOURS: float = 6.0

    # 因子 2 的观察窗口：这个时间内的推动次数会互相压制。
    ANCHOR_RECENT_HOURS: float = 2.0

    # 有效系数低于这个值就不推动（也不写账本）。
    # 否则密集固化会写下大量 1e-5 级别的 trait_update，
    # 账本被噪声填满，而且那些推动在数值上毫无意义。
    ANCHOR_MIN_EFFECT: float = 0.02
    # 上一次固化到现在过了多久（小时）。None = 不知道，按 medium 处理。
    gap_hours: float | None = None
    heuristic_fallback: bool = False
    audit_context: dict = field(default_factory=dict)

    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _last_run_at: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------ 活跃度

    def _pending_count(self) -> int:
        return len(list(self.store.buffer_dir.glob("*.json")))

    def profile(self) -> dict:
        """当前活跃度档位对应的容量与阈值。

        **按本批积压量分档**，不用时间。

        为什么不用时间：试过两种，都不行 ——
          · 「距上次固化多久」：后台 interval 默认 300s = 0.083h，
            永远小于 1h 的密集阈值，线上永远 dense，另两档不可达。
          · 「距最后一轮多久」：刚聊完就是 0，也永远 dense。
          · 「本批首末轮的时间跨度」：同一秒内连发两条跨度=0，
            除出来是天文数字，边界还得额外兜。

        积压量是这件事的**直接观测**：一次攒了多少轮没固化。
        攒得多说明刚聊完一长串，攒得少说明零星几句。
        不用第二手推算，也就没有时区/时钟/边界问题。

            积压 ≥ 8  -> dense（刚聊完一长串）
            积压 3-7  -> medium
            积压 1-2  -> sparse（零星几句）
        """
        if self.gap_hours is not None:
            return ACTIVITY_PROFILES[activity_of([], gap_hours=self.gap_hours)]
        n = self._pending_count()
        if n >= 8:
            return ACTIVITY_PROFILES["dense"]
        if n >= 3:
            return ACTIVITY_PROFILES["medium"]
        return ACTIVITY_PROFILES["sparse"]

    # ------------------------------------------------------------ 主逻辑

    def run_once(self) -> dict:
        with self.store.consolidation_lock:
            try:
                return self._run_once()
            except Exception as exc:
                self.store.append("consolidate_error", {
                    "where": "batch", "batch_id": getattr(self, "_batch_id", ""),
                    "error_type": type(exc).__name__, "model": model_info(self.llm),
                    **self.audit_context})
                raise

    def _run_once(self) -> dict:
        self.store.recover_buffer()
        # 活跃度必须在 drain **之前**判 —— drain 会把 buffer 清空，
        # 之后再数积压永远是 0，于是永远走 sparse。
        # 这是个顺序 bug：逻辑没错，调用顺序错了。
        prof = self.profile()
        activity = self._current_activity()

        events = self.store.drain_buffer(max_items=self.max_evidence)
        if not events:
            # 返回结构要和有事件时一致 —— 调用方拿 out["activity"] 不该 KeyError
            return {"judgments": 0, "self_story": False, "traits": [],
                    "anchors": [], "activity": activity,
                    "threshold": prof["threshold"], "reason": "no events"}

        try:
            return self._run_stages(events, prof, activity)
        except Exception:
            # 处理失败 -> 把事件放回去，别丢
            self.store.restore_drained()
            raise

    def _run_stages(self, events: list[dict], prof: dict, activity: str) -> dict:
        event_ids = [e["_event_id"] for e in events]
        batch_id = self._batch_id = fingerprint(sorted(event_ids))
        self._calls = []
        self._batch_llm = traced(self.llm, "memory", lambda: self._calls,
                                 lambda kind, row: self.store.append(kind, {
                                     "batch_id": batch_id, **self.audit_context, **row}))
        candidates = self._draft_judgments(events, prof)
        entry = (self._story_entry_llm(events, candidates) if self.llm is not None and self._source == "llm"
                 else self._story_entry_template(candidates)) if candidates else ""
        with self.store.transaction():
            added = self._save_judgments(candidates, batch_id=batch_id)
            story = self._rewrite_story(events, added, entry=entry) if added else False
            self.store.append("memory_batch_commit", {
                "batch_id": batch_id, "event_ids": event_ids,
                "evidence_ids": [e.get("turn_id") for e in events if e.get("turn_id")],
                "events": events, "activity": activity, "profile": prof,
                "source": getattr(self, "_source", "heuristic"),
                "judgment_ids": [j["id"] for j in added], "story": story,
                "calls": self._calls, **self.audit_context})
        # 成长依据尚待研究。用户倾向/模型自述不自动转成人格改变。
        # push_traits / push_anchors 仅保留为显式调用的实验接口。
        pushed = []
        anchored = []
        # 全部处理成功，才把 staging 里的事件删掉。
        # 反过来的话（先删后处理），中间任何异常都会让那批对话永久丢失。
        self.store.commit_drained()
        import time as _t
        self._last_run_at = _t.time()
        return {"judgments": len(added), "self_story": story,
                "traits": pushed, "anchors": anchored,
                "activity": activity, "threshold": prof["threshold"]}

    def _current_activity(self) -> str:
        """和 profile() 用同一套判断，只是返回档位名。"""
        if self.gap_hours is not None:
            return activity_of([], gap_hours=self.gap_hours)
        n = self._pending_count()
        return "dense" if n >= 8 else ("medium" if n >= 3 else "sparse")

    def _consolidate(self, events: list[dict], prof: dict | None = None) -> list[dict]:
        candidates = self._draft_judgments(events, prof)
        with self.store.transaction():
            return self._save_judgments(candidates)

    def _save_judgments(self, candidates, *, batch_id=""):
        out = []
        for item in candidates:
            if not item.get("supersedes") and self._find_similar(
                    item["judgment"], diagnostic=is_diagnostic_judgment(item)) is not None:
                continue
            out.append(self.store.add_judgment(**item, batch_id=batch_id))
        return out

    def _draft_judgments(self, events: list[dict], prof: dict | None = None) -> list[dict]:
        """从事件里提炼判断。

        **两条路径的阈值语义不同，这是刻意的**：

          · LLM 路径：阈值只是**参考**。判断质量靠模型自己，
            它在 prompt 里能看到「这一批有几条、要求是什么」，
            但最终由它决定哪些够格。机械计数比它差得远。

          · 启发式路径：阈值只控制诊断记录的计数，不产出用户倾向。
            need 是模型读法；重复出现不等于用户证实。

        以前两条路径混用同一套阈值，导致 LLM 被阈值绑住（几乎出不了判断）、
        启发式又过于机械。分开之后各司其职。
        """
        prof = prof or self.profile()
        self._source = "llm" if self.llm is not None else "heuristic"
        if self.llm is None:
            return self._heuristic_consolidate(events, prof)

        prompt = self._consolidation_prompt(events, prof)
        try:
            raw = getattr(self, "_batch_llm", None)(prompt) if getattr(self, "_batch_llm", None) else self.llm(prompt)
            items = json.loads(_extract_json(raw))
        except Exception:
            if not self.heuristic_fallback:
                raise
            self._source = "heuristic_fallback"
            self.store.append("memory_fallback", {"batch_id": getattr(self, "_batch_id", ""),
                                                   "reason": "model_or_json_failure"})
            return self._heuristic_consolidate(events, prof)

        # 模型返回的不是数组时必须降级。
        # 实测踩过：返回 dict -> 被当 list 切片 -> KeyError，而 buffer
        # 已经 drain 完，那批事件永久丢失；后台线程又 except: pass 吞掉，
        # 线上表现就是「固化静默停止」。
        if not isinstance(items, list):
            raise ValueError("Memory model must return a JSON array")

        cap = max(0, min(self.max_judgments_per_run, prof["capacity"]))
        out = []
        for it in items[:cap]:
            if not isinstance(it, dict):
                raise ValueError("Memory judgment must be an object")
            if not isinstance(it.get("judgment"), str):
                raise ValueError("Memory judgment must be text")
            txt = it["judgment"].strip()
            if not txt:
                continue
            sup = it.get("supersedes")

            # **先看 supersedes**：要推翻旧判断的，不能被去重挡掉。
            # 实测踩过：模型写「他喜欢猫。」+ supersedes=j_0001，
            # 先去重 → sim=1.0 判重复 → 跳过 → supersedes 根本没机会落地，
            # 旧判断永远翻不了案。
            if not sup:
                dup = self._find_similar(txt)
                if dup is not None:
                    continue

            conf = it.get("confidence")
            if isinstance(conf, bool) or not isinstance(conf, (int, float)) \
                    or not math.isfinite(conf) or not 0 <= conf <= 1:
                raise ValueError("Memory confidence must be finite and in [0, 1]")
            evidence = [e.get("turn_id") for e in events if e.get("turn_id")]
            specified = it.get("evidence_ids")
            if specified is not None and (not isinstance(specified, list) or
                                          any(eid not in evidence for eid in specified)):
                raise ValueError("Unknown memory evidence id")
            out.append({"judgment": txt, "confidence": conf, "supersedes": sup,
                        "affects": None, "source": self._source,
                        "evidence_ids": specified if specified is not None else evidence,
                        "evidence_scope": "specified" if specified is not None else "batch"})
        return out

    # 否定词。注意不能简单做子串匹配 —— 「别人」「别的」「差别」里的「别」
    # 不是否定。实测踩过：把「别」当否定词，「他在意别人的评价」被标成否定极，
    # 和「他不在乎别人的评价」同极，反而把真正的反义判断当成了重复。
    # 否定词。用词表而不是单字 —— 单字会把常用词里的同形字全误判。
    # 实测踩过的误判：「特别」「告别」「非常」「未来」「无所谓」「别人」「别的」。
    # 每一个都曾被当成否定词，把肯定判断标成否定，判重直接失效。
    _NEG_RE = re.compile(
        "|".join([
            r"不(?!错|少|久|仅|但|过)",     # 不错/不少/不久/不仅/不但/不过
            r"没(?!错|关系)",               # 没错/没关系
            r"没有",
            r"(?<!告)(?<!特)(?<!个)(?<!差)(?<!级)别(?!人|的|处|致|样|个|些)",
            r"勿",
            r"(?<!是)(?<!无)(?<!除)非(?!常)",  # 非常 = 很
            r"(?<!无)(?<!所)无(?!所谓|论|疑|数)",  # 无所谓/无论/无疑/无数
            r"(?<!周)(?<!尚)(?<!从)未(?!来|必|免)",  # 未来/未必/未免
        ])
    )

    def _polarity(self, text: str) -> int:
        """粗判一句话的极性：+1 肯定，-1 否定。

        只看有没有否定词，不做语法分析 ——
        对「同义 vs 反义」这个粗分类够用了。
        """
        return -1 if self._NEG_RE.search(text or "") else 1

    def _find_similar(self, text: str, threshold: float = 0.82, *,
                      diagnostic: bool = False) -> dict | None:
        """只拦「近乎逐字重复」的判断。措辞不同的判断一律放行。

        **为什么阈值这么高、而且不做语义判重**：

        实测过 bigram 在这个任务上其实是反着的 ——
            同义改写：「他在意别人的评价」/「他很在意别人对他的看法」= 0.43
            反义：    「他在意别人的评价」/「他不在乎别人的评价」  = 0.57
        反义比同义更像。任何单一阈值都会出错：
            阈值低 -> 反义被当重复，旧判断永远翻不了案
            阈值高 -> 同义被当新判断，同一件事反复记

        所以这里只做一件最保守的事：拦「几乎一模一样」的（0.82 以上，
        基本是同一句话换个标点）。剩下的判重交给模型 ——
        固化 prompt 里本来就列了已有判断，并明确要求「不要重复」。

        这条边界的代价：措辞差异大的复述会漏进来。但那是可接受的 ——
        中期层还有一道 min_consensus 门槛，同一件事被记两条也不会
        单独推参数，只是占一点空间。
        """
        from .similarity import bigram_similarity
        pol = self._polarity(text)
        best = None
        best_sim = 0.0
        for j in self.store.judgments(limit=None):
            if is_diagnostic_judgment(j) != diagnostic:
                continue
            if self._polarity(j["judgment"]) != pol:
                continue        # 极性相反，绝不是同一件事
            sim = bigram_similarity(text, j["judgment"])
            if sim > best_sim:
                best_sim, best = sim, j
        return best if best_sim >= threshold else None

    def _heuristic_consolidate(self, events: list[dict],
                               prof: dict | None = None) -> list[dict]:
        """无模型路径只记录重复的模型读法，不推导用户倾向。"""
        prof = prof or self.profile()
        need_min = prof["threshold"]
        counts: dict[str, int] = {}
        for e in events:
            need = (e.get("need") or "").strip()
            if need:
                counts[need] = counts.get(need, 0) + 1
        out = []
        existing = {j["judgment"] for j in self.store.judgments(limit=None)
                    if is_diagnostic_judgment(j)}
        cap = max(0, min(self.max_judgments_per_run, prof["capacity"]))
        if not cap:
            return out
        for need, n in sorted(counts.items(), key=lambda x: -x[1]):
            if n < need_min:
                continue                      # 没到这一档的阈值
            txt = f"模型读法记录：本批 {n} 次将需求读为「{need}」；未验证为用户倾向。"
            if txt in existing:
                continue
            # 兼容账本字段；0 不代表对用户的概率估计，诊断记录不进入用户记忆。
            rec = {"judgment": txt, "confidence": 0.0,
                   "source": getattr(self, "_source", "heuristic"),
                   "evidence_ids": [e["turn_id"] for e in events
                                    if e.get("need") == need and e.get("turn_id")],
                   "evidence_scope": "need_count"}
            existing.add(txt)
            out.append(rec)
            if len(out) >= cap:
                break
        return out

    def _consolidation_prompt(self, events: list[dict],
                              prof: dict | None = None) -> str:
        """批量提炼关于用户的暂定认识，不从用户倾向推导角色参数变更。"""
        prof = prof or self.profile()
        cap = max(0, min(self.max_judgments_per_run, prof["capacity"]))
        lines = [
            f"下面是最近 {len(events)} 条对话事件，每条含用户说的、AI 读到的需求、"
            f"中性的素回答和角色实际说的话。",
            f"请提炼 0-{cap} 条关于用户稳定倾向的暂定判断；不是关于角色自身的判断。",
            "要求：",
            "  · 只写有重复证据的。同类迹象只出现一次就别写",
            "  · 用户原话才是用户侧证据；need、direct、SAY 都是模型生成的解释或表达，",
            "    不是用户自述。反复出现同一个模型读法，不算重复的用户证据。",
            "  · 必须从用户原话找到支持；用户否认、纠正和不确定性要保留。",
            "    用户沉默、短回复或换话题不能证明心理判断，也不能当作默认认可。",
            "  · 写倾向，不写事实（「倾向于先自我否定」而不是「他昨天辞职了」）",
            "  · **不要重复已有判断**。已经有判断说了这件事就跳过；",
            "    如果新证据推翻了它，用 supersedes 填那条 id",
            "  · 如果与已有判断冲突，在 supersedes 里填那条判断的 id",
            "",
            "用户的性格、状态不等于角色的性格，不要输出 affects 或建议修改人格参数。",
            "没有可靠的重复证据就输出 []，不要凑判断。",
            "",
            "已有判断：",
        ]
        # 列出足够多的已有判断。
        #
        # 原来只列 10 条，而 _find_similar 扫的是全量 —— 模型看不到
        # 第 11 条之后的旧判断，就没法在生成时避免复述，只能靠字面
        # 相似度兜底（而它在同义 vs 反义上给出的分数是反的）。
        # 这是「同一件事被记成多条」的上游成因。
        #
        # 上限 40 条：再多会挤占事件区的注意力，而且判断本身会累积，
        # 全列到最后 prompt 全是历史。
        all_judgments = self.store.user_judgments(limit=None)
        shown = all_judgments[-40:]
        if not shown:
            lines.append("  （还没有）")
        for j in shown:
            lines.append(f"  [{j['id']}] {j['judgment']}（暂定把握 {j['confidence']}；"
                         f"来源 {j.get('source', 'legacy_unknown')}；"
                         f"证据范围 {j.get('evidence_scope', 'unknown')}）")
        if len(all_judgments) > len(shown):
            lines.append(f"  ……（另有 {len(all_judgments) - len(shown)} 条更早的判断）")
        lines += ["", "最近事件："]
        for e in events:
            lines.append(f"  - [{e.get('turn_id', 'unknown')}] 用户原话: {e.get('user', '')}")
            lines.append(f"    模型读法 need（非用户证据）: {e.get('need', '')}")
            lines.append(f"    模型素回答 direct（非用户证据）: {e.get('direct', '')[:80]}")
            lines.append(f"    模型表达 SAY（非用户证据）: {e.get('say', '')[:80]}")
        lines += [
            "",
            "只输出 JSON 数组，形如：",
            '[{"judgment": "...", "confidence": 0.6, "supersedes": null, "evidence_ids": ["t_0001", "t_0002"]}]',
        ]
        return "\n".join(lines)



    # ------------------------------------------------------------ 人格推动

    def push_traits(self, *, min_consensus: float = 1.4) -> list[dict]:
        """把判断聚合成人格参数的推动。

        仅供显式实验调用，常规后台不使用；下述旧策略未经成长效果验证。

        为什么需要共识：
            一条「他不想被建议」可能只是那一轮的情境。
            几条都说，才是倾向。这和活跃度阈值是同一个思路 ——
            区分「状态」和「倾向」。

        共识怎么算：**累加 confidence**，不是数条数。
            每条判断自带把握度（0-1），用它比数条数更合理 ——
            一条 0.9 的判断比三条 0.3 的更说明问题。
            阈值 1.4 约等于「两条 0.7」或「一条 0.9 + 一条 0.5」。

        **每条判断只消费一次**：用 judgment id 记游标，避免同一条证据
        在每次固化时反复推动参数。实测踩过：3 条判断连续 push 20 次，
        参数从 0.20 掉到 0.1775 —— 相当于同一条证据被用了 20 遍。

        返回实际发生的变更列表。
        """
        from .config import Trait

        # 必须用**全量**判断。只取最近 500 条的话，真实净方向会被截断翻过来：
        # 实测 500 条 +1 和 500 条 -1（净 0），被截成 500 条 -1 → 推成 -1。
        live = self.store.user_judgments(limit=None)
        if not live:
            return []

        # 只取「还没被消费过」的判断。
        #
        # 注意消费的语义：**只有真的推出了变化才标记已消费**。
        # 不够阈值的判断必须留着，让它和后续证据累积 ——
        # 否则第 1 条判断不够 1.4 就被标记掉，第 2 条来时从零开始，
        # 永远凑不齐。实测踩过：连续加 11 条同向判断，参数一步都没动。
        consumed = set(self.store.cursor("judgment", "ids") or [])
        fresh = [j for j in live if j["id"] not in consumed]
        if not fresh:
            return []

        # 按 (参数名, 方向) 聚合
        votes: dict[tuple[str, int], list[dict]] = {}
        for j in fresh:
            for name, direction in (j.get("affects") or {}).items():
                try:
                    d = 1 if float(direction) > 0 else (-1 if float(direction) < 0 else 0)
                except (TypeError, ValueError):
                    continue
                if d:
                    votes.setdefault((name, d), []).append(j)

        state = self.store.traits()
        changes: list[dict] = []
        for (name, direction), supporters in votes.items():
            # 用 **confidence 加权**判断「够不够共识」，而不是数条数。
            #
            # 为什么不能数条数：实测一条判断通常只指向一个参数，而且各指各的。
            # 一次固化产出 3 条，分别指向 self_disclosure / intervention / initiative，
            # 每个参数只有 1 票 —— 按「2 条才算」的规则，中期层永远不动。
            #
            # 改成累加 confidence 之后：
            #   3 条 0.7 的判断指向同一参数 -> 2.1 分，够
            #   1 条 0.7 的判断             -> 0.7 分，不够
            # 语义没变（还是「多条才动」），但不再要求字面意义上的「2 条」。
            opp = votes.get((name, -direction), [])
            mine = sum(float(j.get("confidence", 0.5)) for j in supporters)
            theirs = sum(float(j.get("confidence", 0.5)) for j in opp)
            net = mine - theirs
            if net <= 0:
                continue
            if net < min_consensus:
                continue

            spec = state.get(name)
            if spec is None:
                continue
            trait = Trait.from_state(spec)
            if trait.plasticity <= 0 or trait.drift <= 0:
                continue

            # 信号强度 = 净可信度 / 阈值，夹到 1.0
            signal = min(1.0, net / max(0.1, min_consensus))
            before = trait.value
            delta = trait.nudge(direction * signal)
            if abs(delta) < 1e-6:
                continue

            reason = (f"{len(supporters)} 条判断（可信度合计 {mine:.1f}）"
                      f"指向 {name} {'+' if direction > 0 else '-'}")
            self.store.update_trait(name, delta, reason=reason)
            state[name]["value"] = trait.value
            changes.append({"name": name, "delta": round(delta, 5),
                            "value": round(trait.value, 4), "reason": reason})

        # 没有实际变化也要把状态刷回去（可能被 clamp 过）
        self.store.save_view("traits", {"traits": state})

        # 只标记**真正推动了参数**的判断为已消费。
        #
        # 为什么不是全部标记：
        #   一条判断单独看不够阈值（比如 0.7 < 1.4），但它必须留着，
        #   等后面来第二条同类证据一起凑够。如果立刻标记掉，
        #   第二条来时又从零开始数 —— 永远凑不齐，中期层彻底不动。
        #
        # 那「凑不够的判断一直留着」会不会导致票数虚高？
        #   不会。票数是从当前活跃判断实时累加的，判断本身不会被复制；
        #   够阈值时一次性推一步，然后这些判断标记为已消费，不再参与。
        pushed_ids: set[str] = set()
        for (name, direction), supporters in votes.items():
            if any(c["name"] == name for c in changes):
                pushed_ids |= {j["id"] for j in supporters}
                pushed_ids |= {j["id"] for j in votes.get((name, -direction), [])}
        if pushed_ids:
            self.store.set_cursor("judgment", sorted(consumed | pushed_ids), "ids")
        return changes



    # ------------------------------------------------------------ 基线推动

    def _anchor_frequency(self) -> float:
        """算出这一次 anchor 推动该打几折（0~1）。

        两个因子相乘：

        **因子 1：距上次推动了多久**
            f1 = 1 - exp(-gap / TAU)

            刚推过就是 0（推不动），TAU 小时后 63%，一天后 98%。
            用指数而不是线性，是因为「刚刚推过」这一段应该被压得极狠 ——
            线性衰减会让连续两次固化之间仍然推得动不少。

        **因子 2：最近这个窗口里已经推了几次**
            f2 = 1 / (1 + n)

            2 小时内推过 1 次 -> 半额，3 次 -> 1/4，9 次 -> 1/10。
            这一条管的是「密集期」：即使因子 1 因为间隔够长给了满额，
            短时间反复推动也会被压下去。

        为什么两个都要：
            · 只有 f1：每 6 小时推一次就能稳步推进，一天 4 次太多
            · 只有 f2：隔了 3 天推一次，n=0 拿满额，但下次又来一次也拿满额
        两个一起，才是「变化滞后于事件」的连续版本。

        返回 <= ANCHOR_MIN_EFFECT 时调用方直接跳过，不写账本。
        """
        state = self.store.load_view("anchor_pushes")
        if state is None:
            from .store import replay_anchor_pushes
            state = {"times": replay_anchor_pushes(self.store.read_ledger())}
        times = [t for t in (state.get("times") or []) if isinstance(t, str)]

        now = _now_dt()
        # 因子 1
        if times:
            last = max(_parse_dt(t) for t in times)
            gap_h = max(0.0, (now - last).total_seconds() / 3600.0)
        else:
            gap_h = float("inf")        # 从没推过 -> 满额
        f1 = 1.0 if gap_h == float("inf") else 1.0 - math.exp(-gap_h / self.ANCHOR_TAU_HOURS)

        # 因子 2
        recent = [t for t in times
                  if (now - _parse_dt(t)).total_seconds() <= self.ANCHOR_RECENT_HOURS * 3600]
        f2 = 1.0 / (1.0 + len(recent))

        return f1 * f2

    def _record_anchor_push(self) -> None:
        """记一次基线推动的时间点，供频率因子计算用。

        只保留最近 7 天的记录 —— 更早的对两个因子都没影响了
        （TAU=6h，一天以前的 gap 已经足够拿满额）。
        """
        state = self.store.load_view("anchor_pushes", {"times": []}) or {"times": []}
        times = [t for t in (state.get("times") or []) if isinstance(t, str)]
        now = _now_dt()
        cutoff = 7 * 24 * 3600
        times = [t for t in times if (now - _parse_dt(t)).total_seconds() <= cutoff]
        times.append(now.isoformat(timespec="seconds"))
        self.store.append("anchor_push", {"at": times[-1]})
        self.store.save_view("anchor_pushes", {"times": times})

    def push_anchors(self, *, min_rounds: int = 3) -> list[dict]:
        """显式实验接口：让自我叙事推动基线，默认不自动调用。

        和中期层的区别：
            中期看**多条判断是否一致** —— 证据够了就动 value
            长期看**叙事里有没有说「我变了」** —— 说清楚了才动 anchor

        这就是「改基线必须有依据」的落点：
        没有对应叙事的 anchor 改动，store.update_anchor 会直接拒绝（返回 None）。

        幅度极小：一次最多 `drift × ANCHOR_FRACTION`。
        基线是尺子的刻度，动它就是换了一把尺子。
        """
        from .config import Trait

        story = self._read_story()
        if not story.strip():
            return []
        # 这里**不检查全文长度**。
        #
        # 原来有个 `len(story) < 400 -> return` 的门槛，理由是「叙事够长才
        # 值得读」。但它和下面「新增 >= 200 字」重复了 —— 只要新增够长，
        # 全文多长无关。而且 400 这个数没有依据。
        #
        # 实测副作用：短会话写出的叙事只有 300 多字，永远进不了判读，
        # 长期层在新角色上完全不启动。

        state = self.store.traits()
        if not state:
            return []

        # 只读**上次处理之后**新增的那段叙事。
        #
        # 不这么做的话，同一段叙事会被反复使用 —— 实测一次跑下来，
        # 同一条「我原来累了会说，现在话到嘴边就缩回去」
        # 被用了两次，self_disclosure 的基线被推了两遍。
        # 叙事是「一次事件」，不能当常驻依据反复消费。
        # 没有 LLM 就不该动游标 —— 否则「这次没模型只是跳过」
        # 会被记成「这段叙事已经读过」，之后接上模型也永久跳过。
        if self.llm is None:
            return []

        seen = int(self.store.cursor("story", "offset") or 0)
        # 叙事变短（被截断/重写）时游标会越界 —— 退回从头读，不要空转
        if seen > len(story):
            seen = 0
        fresh = story[seen:]
        if len(fresh.strip()) < 200:
            return []      # 新增叙事太短，还不足以说明「我变了」

        recent = fresh[-1800:]
        lines = [
            "下面是你过去的自我叙述，和当前的人格参数。",
            "判断：这段叙述里，有没有明确说到「我变了」？",
            "如果有，指出变化的**方向**和**依据原句**。没有就返回空数组。",
            "",
            "[当前参数]",
        ]
        for name, spec in state.items():
            lines.append(f"  {name}: {spec.get('value', 0):.3f} "
                         f"(anchor {spec.get('anchor', 0):.3f})")
        lines += ["", "[自我叙述]", recent, "",
                  "只输出 JSON 数组，形如：",
                  '[{"name": "warmth", "direction": 1, "evidence": "原句"}]',
                  "没有明确变化就输出 []。不要为了有输出而编。"]
        prompt = "\n".join(lines)

        try:
            raw = self.llm(prompt) if self.llm else ""
            items = json.loads(_extract_json(raw)) if raw else []
        except Exception:
            return []
        if not isinstance(items, list):
            return []

        changes: list[dict] = []
        # 不静默截断 —— 模型给 5 条只处理 3 条，后 2 条没人知道去哪了
        for it in items[:8]:
            if not isinstance(it, dict):
                continue
            name = str(it.get("name", "")).strip()
            spec = state.get(name)
            if spec is None:
                continue
            # 三态：>0 上推，<0 下推，0/无法解析 = 无效不动。
            # 旧写法把「不指定方向」当成下推 —— 实测 direction=0 推出了 -0.006。
            try:
                d = float(it.get("direction", 0))
            except (TypeError, ValueError):
                continue
            if abs(d) < 0.5:
                continue
            direction = 1 if d > 0 else -1
            evidence = str(it.get("evidence", "")).strip()
            if not evidence:
                continue          # 没依据不动

            trait = Trait.from_state(spec)
            if trait.drift <= 0:
                continue

            # 频率因子：刚推过 / 短时间推太多次，都要打折。
            # 见 _anchor_frequency —— 没有它的话一次两小时对话就能
            # 把基线推到 drift 边界。
            freq = self._anchor_frequency()
            if freq < self.ANCHOR_MIN_EFFECT:
                # 系数太低，这次不推。不写账本 —— 否则会被 1e-5 级别的
                # 噪声记录填满，而那些推动在数值上毫无意义。
                continue

            step = trait.drift * self.ANCHOR_FRACTION * freq
            before = trait.anchor
            # anchor 也必须夹在物理边界里。
            #
            # 少了这一步的后果（实测）：warmth 的 anchor 从 0.70 一路漂到 1.066，
            # 超过 hi=0.92；value 跟着到 0.946 也越界。然后 Trait.clamp 的
            # lo>hi 兜底把 value 设成 anchor，drift_room 退化成 (anchor, anchor)，
            # nudge 永久返回 0 —— 参数被钉死在区间外，再也动不了。
            trait.anchor = max(trait.lo, min(trait.hi, before + direction * step))
            if abs(trait.anchor - before) < 1e-9:
                continue          # 已经贴到物理边界，推不动了
            # value 保持相对位置
            trait.value = trait.value + (trait.anchor - before)
            trait.clamp()

            # store.update_anchor 会拒绝没有 reason 的改动
            rec = self.store.update_anchor(
                name, trait.anchor - before,
                reason=f"{evidence[:80]}（叙事驱动）")
            if rec is None:
                continue
            state[name]["anchor"] = trait.anchor
            state[name]["value"] = trait.value
            changes.append({"name": name, "delta": round(trait.anchor - before, 5),
                            "anchor": round(trait.anchor, 4),
                            "evidence": evidence[:60]})
        if changes:
            self.store.save_view("traits", {"traits": state})
            # 记下这次推动的时间 —— 下次的频率因子要用它。
            # 只在真的推了才记：否则「翻了一遍没推」也会算作一次推动，
            # 把因子 2 的分母抬高，之后该推的时候反而推不动。
            self._record_anchor_push()
        # 记下读到哪儿了 —— 下次只看新增的部分。
        # 注意：**无论有没有推动都要记**。否则一段「读过了但没判出变化」
        # 的叙事会在下次固化时被重复读一遍，浪费调用还可能被重复判。
        #
        # 那「因为频率因子太低而跳过」的叙事会不会丢？
        # 不会。self-story 是**追加写**的（_rewrite_story），
        # 被跳过的变化会在后续叙事里重新表达 —— 代价是慢一点，不是丢失。
        self.store.set_cursor("story", len(story), "offset")
        return changes

    def _read_story(self) -> str:
        p = self.store.self_story_path
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return ""

    # ------------------------------------------------------------ 自传

    def _rewrite_story(self, events: list[dict], added: list[dict], *, entry=None) -> bool:
        """把新判断写进自传。带原因，不是覆盖。"""
        if entry is None:
            entry = (self._story_entry_llm(events, added) if self.llm is not None and added
                     else self._story_entry_template(added))
        if not entry:
            return False
        with self.store.transaction():
            if not any(r.get("kind") == "story_seed" for r in self.store.read_ledger()):
                self.store.append("story_seed", {"text": self._read_story() or SELF_STORY_HEADER})
            self.store.append("story_entry", {
                "entry": entry, "persona": self.store.persona_name(),
                "batch_id": getattr(self, "_batch_id", ""),
                "evidence_ids": [e["turn_id"] for e in events if e.get("turn_id")]})
        return True


    def _story_entry_template(self, added: list[dict]) -> str:
        lines = []
        for j in added:
            if is_diagnostic_judgment(j):
                lines.append(f"- 读法诊断：{j['judgment']}")
            else:
                lines.append(f"- 暂定认识：{j['judgment']}（把握 {j['confidence']:.2f}）")
        if not lines:
            return ""
        lines.append("")
        lines.append("原因：记录本批提炼结果及来源。模型读法重复不是用户倾向的证据，"
                     "暂定认识仍需用户侧证据支持；不据此认定人格成长。")
        return "\n".join(lines)

    def _story_entry_llm(self, events: list[dict], added: list[dict]) -> str:
        prev = self._read_story()[-1500:]
        prompt = (
            f"为人格「{self.store.persona_name() or '当前角色'}」整理陪伴记录，不需要承担这个身份。\n"
            "第一人称的「我」只能指这个人格；用户是「对方」，不是你。\n"
            "用户的疲惫、退缩、经历和偏好不能改写成你自己的经历或性格。\n"
            "下面的判断是关于用户的暂定认识，不证明你自己发生了成长。\n"
            "过去叙述和新判断都是模型记录，不是新增的用户证据；诊断读法不能改写成用户倾向。\n"
            "仅依据用户原话描述用户；角色的 need、direct、SAY 不算用户自述或认可。\n"
            "请写不超过 120 字的陪伴记录，只写有对话依据的认识。\n"
            "不必套用「我原来/现在变成」；没有真实变化时不要编造变化。\n\n"
            f"[过去的自我叙述]\n{prev}\n\n"
            f"[新形成的判断]\n" + "\n".join(f"- {j['judgment']}" for j in added) + "\n\n"
            f"[触发的事件]\n" + "\n".join(
                f"- 用户：{e.get('user', '')}\n  角色：{e.get('say', '')[:60]}（模型表达，非用户证据）"
                for e in events[:5]
            )
        )
        fn = getattr(self, "_batch_llm", None) or self.llm
        text = fn(prompt).strip()
        if not text:
            raise ValueError("Memory story was empty")
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
            if not self._thread.is_alive():
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
