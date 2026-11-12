"""The selected five-example prompt, adapted to an ongoing, visible-only chat.

The candidate text is read verbatim from the earlier development selection.
Both arms use the caller's current repository character, rather than the pilot's
abridged research profile. The plain-prompt arm receives the whole character
phrase/special catalogue as optional references, not a framework-only need label.
No store, protocol history, assessment task or future user turn enters the input.
"""
import json
from pathlib import Path

from persona import prompt


PILOT = Path(__file__).resolve().parent.parent / "noninferiority_pilot_2026_11_10"
SOURCE_FILES = (PILOT / "selection.json", PILOT / "prompts.json")


def selected_system():
    selection, options = [json.loads(p.read_bytes()) for p in SOURCE_FILES]
    chosen = selection["selected_id"]
    if chosen != "examples":
        raise ValueError("The registered baseline must remain the pilot's selected examples arm")
    system, = [c["system"] for c in options["candidates"] if c["id"] == chosen]
    return system


def role_spec(cfg):
    r = cfg.role
    def plain(text):
        # The historical specification asks to *write* HEAD.confidence, even for
        # a visible-only baseline. Preserve its semantic instruction and intervals.
        return (text.replace("先根据本轮输入写 HEAD.confidence，再据此选择站位",
                             "仅在心里根据本轮输入估计把握度，再据此选择站位")
                .replace("baseline/SAY", "实际回复").replace("direct", "基础判断"))
    ref = dict(name=r.name, spec=r.identity, voice=r.voice, attention=r.attention,
               attention_hint=prompt._ATTENTION_HINT[r.attention], bias=r.bias,
               legibility_max=r.legibility_max, trait_guidance=plain(prompt._trait_block(cfg)),
               stance_guidance=plain(prompt.stance_policy(cfg)))
    if r.phrases:
        ref["分场景语感示例（按需参考，不照抄）"] = r.phrases
    if r.special:
        ref["条件性表达参考（有对应线索时使用）"] = r.special
    return json.dumps(ref, ensure_ascii=False)


def make_messages(cfg, history, user):
    """Only completed visible user/SAY exchanges may be replayed."""
    if not isinstance(user, str) or not user.strip() or len(history) % 2:
        raise ValueError("Expected a user message and an even number of visible history entries")
    previous = []
    for n, item in enumerate(history):
        role = "user" if n % 2 == 0 else "assistant"
        if set(item) != {"role", "content"} or item["role"] != role or not isinstance(item["content"], str):
            raise ValueError("Invalid visible conversation history")
        previous.append(dict(role=role, content=item["content"]))
    return [dict(role="system", content=selected_system() + "\n\n[共享人格规格]\n" + role_spec(cfg)),
            *previous, dict(role="user", content=user)]
