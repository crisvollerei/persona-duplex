"""命令行入口。

    chat      普通对话（默认流式）
    trace     带 trace 的对话（能看到 HEAD / 结算 / 修复）
    demo      离线演示，不需要 API key
    rebuild   从账本重建 state/
    stats     从账本算指标
    calibrate 从账本反推阈值该怎么调
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PersonaConfig
from .llm import (
    ScriptedLLM,
    make_reply,
    openai_compatible,
    openai_stream,
    openai_tool_call,
    text_prompt,
)
from .session import Session, SessionConfig
from .store import Store, rebuild_views
from .term import bold, cyan, dim, red, yellow

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 载入


def _config_dirs() -> list[Path]:
    """人格配置的搜索路径。

    装成包之后 personas 是可导入的数据包；直接跑源码时就在仓库根目录。
    两条都留着，哪种跑法都能找到文件。
    """
    dirs: list[Path] = []
    try:
        import personas as pkg          # type: ignore[import-not-found]
        dirs.append(Path(pkg.__file__).resolve().parent)
    except Exception:
        pass
    dirs.append(ROOT / "personas")
    dirs.append(Path(__file__).resolve().parent / "personas")
    return dirs


def _load_cfg(name: str) -> PersonaConfig:
    p = Path(name)
    if not p.exists():
        for d in _config_dirs():
            cand = d / f"{name}.toml"
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise SystemExit(f"找不到人格配置: {name}")
    return PersonaConfig.load(p)


def _make_session(args, *, streaming: bool) -> Session:
    cfg = _load_cfg(args.persona)
    store = Store(args.data)
    llm = openai_compatible(args.model)
    stream_llm = openai_stream(args.model) if streaming else None

    # 话题判定单独走一个轻模型：不开思考、temperature=0、输出极短。
    # 它异步跑，和主生成并行，所以永远不会拖慢对话。
    #
    # 注意这里给的是 HTTP 超时，不是 grace：
    #   grace   = 主链路愿意等它多久（很短，0.35s）
    #   timeout = 这个请求本身允许跑多久（可以长一些）
    # 判定慢一点没关系，只要主链路不等它就行。
    fast_llm = None
    if not args.no_judge:
        fast_llm = openai_compatible(
            args.judge_model or args.model,
            temperature=0.0,
            max_tokens=120,
            timeout=max(8.0, cfg.judge_grace * 10),
        )

    # 慢层固化：没有模型时只会数 need 出现几次，那不是判断，是计数。
    # 用同一个端点，但温度低、输出短 —— 它做的是提炼，不是创作。
    consolidate_llm = None
    if not args.no_judge:
        consolidate_llm = text_prompt(openai_compatible(
            args.judge_model or args.model,
            temperature=0.3,
            max_tokens=400,
            timeout=30.0,
        ))

    # tool 协议需要「填 schema」的调用方式，跟纯文本的 llm 接口不同。
    tool_llm = None
    if args.protocol in ("tool", "auto", "persona"):
        tool_llm = openai_tool_call(args.model, temperature=0.9,
                                    max_tokens=600, timeout=60.0)

    return Session(
        cfg, store, llm,
        stream_llm=stream_llm,
        fast_llm=fast_llm,
        consolidate_llm=consolidate_llm,
        tool_llm=tool_llm,
        protocol=args.protocol,
        options=SessionConfig(stream=streaming),
    )


# ---------------------------------------------------------------- 展示


def _print_trace(res) -> None:
    h = res.parsed.head
    print(f"\n{dim('── ' + res.turn_id + ' ──')}")
    if h:
        print(f"{cyan('need')}      {h.need}   {cyan('conf')} {h.confidence}")
        print(f"{cyan('direct')}    {h.direct}")
        print(f"{cyan('displace')}  {h.displace}")
    print(f"{cyan('stance')}    {res.stance}")
    for s in res.settled:
        label = s.get("label", "")
        print(yellow(f"settle    {s['id']} → {s['outcome']}（{label}）"))
    if res.pending_added:
        p = res.pending_added
        print(yellow(f"pending   {p['claim']} ({p['confidence']})"))
    if res.repair_opened:
        print(red(f"repair    发起修复（{res.repair_opened.get('action')}）"))
    if res.repair_closed:
        print(red(f"repair    结果 {res.repair_closed.get('verdict')}"))
    for w in res.warnings:
        print(red(f"⚠  {w}"))


def _banner(sess: Session, online: bool, model: str = "") -> None:
    cfg = sess.engine.cfg
    r = cfg.role
    line = "─" * 54
    print(dim(line))
    print(f"人格  {r.name}   bias={r.bias:+.2f}  "
          f"attention={r.attention}  legibility={r.legibility_max}")
    print(f"说明  {r.voice}")
    if online and model:
        print(f"模型  {model}")
    print(f"模式  {'在线' if online else '离线演示'}   数据 {sess.store.root}")
    print(dim("输入空行或 Ctrl-C 退出"))
    print(dim(line))


def _repl(sess: Session, show_trace: bool) -> None:
    streaming = sess.opts.stream
    try:
        while True:
            try:
                text = input(f"\n{cyan('你')}  ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                return

            try:
                if show_trace:
                    res = sess.say(text)
                    _print_trace(res)
                    print(f"\n{bold(res.say)}")
                elif streaming:
                    print(f"{bold('它')}  ", end="", flush=True)

                    def sink(piece: str) -> None:
                        print(piece, end="", flush=True)

                    sess.say(text, on_delta=sink)
                    print()
                else:
                    res = sess.say(text)
                    print(f"{bold('它')}  {res.say}")
            except Exception as e:              # 生成失败不崩
                print(red(f"生成失败：{e}"))
    finally:
        sess.close()


# ---------------------------------------------------------------- 子命令


def cmd_chat(args) -> int:
    sess = _make_session(args, streaming=not args.no_stream)
    _banner(sess, online=True, model=args.model)
    _repl(sess, show_trace=False)
    return 0


def cmd_trace(args) -> int:
    sess = _make_session(args, streaming=False)
    _banner(sess, online=True, model=args.model)
    _repl(sess, show_trace=True)
    return 0


def cmd_demo(args) -> int:
    """离线跑通：不联网，验证协议、结算、修复、后台、指标。"""
    cfg = _load_cfg(args.persona)
    store = Store(args.data)

    replies = [
        make_reply(
            head="need: 只想被听\nconfidence: 0.72\n"
                 "direct: 你根本不是在问我，你只是想让人知道你有多累\n"
                 "displace: 降级",
            say="今天是不是特别长啊。",
            tail="settle: -\npending:\n  claim: 用户此刻只想被听\n"
                 "  confidence: 0.7\n  predicts: 下一句会继续描述细节，不会问怎么办",
        ),
        make_reply(
            head="need: 要被推一把\nconfidence: 0.62\n"
                 "direct: 你其实早就想辞了，只是在等一个人说可以\n"
                 "displace: 悬置",
            say="那你是想听我说「辞了吧」，还是想让我帮你算算还能撑多久？",
            tail="settle: p_0001 | refuted | 用户直接问怎么办，不是想被听\n"
                 "pending:\n  claim: 用户已经有倾向，需要的是被允许\n"
                 "  confidence: 0.6\n  predicts: 下一句会说出某个具体方案",
        ),
        make_reply(
            head="need: 想被拦住\nconfidence: 0.45\n"
                 "direct: 你每次说到这一步就跑\n"
                 "displace: 截断",
            say="……嗯。吃过了，你呢。",
            tail="settle: -",
        ),
        make_reply(
            head="need: 要被推一把\nconfidence: 0.66\n"
                 "direct: 你看，你又绕回来了\ndisplace: 多余",
            say="吃了。不过你刚才那句「先不说这个了」，我记着。",
            tail="settle: p_0002 | confirmed | 用户绕了一轮还是回到这件事\n"
                 "delta:\n  closeness: 0.03\n  unresolved: 辞职这件事还没说完",
        ),
        make_reply(
            head="need: 只是想说说话\nconfidence: 0.85\n"
                 "direct: 那挺好的\ndisplace: none",
            say="两家都是做哪块的？",
            tail="settle: -",
        ),
        make_reply(
            head="need: 要被推一把\nconfidence: 0.55\n"
                 "direct: 你就是不敢跟人正面说话\ndisplace: none",
            say="你每次都这样，话说一半就缩回去。",
            tail="settle: -\npending:\n  claim: 用户需要被推一把\n"
                 "  confidence: 0.55\n  predicts: 会反驳，或者沉默",
        ),
        # 修复不在受伤的当轮发。所以第 7 轮先说别的，第 8 轮才道歉 ——
        # 这个间隔本身就是设计的一部分，演示里也要演出来。
        make_reply(
            head="need: 只是想说说话\nconfidence: 0.8\n"
                 "direct: 那就好\ndisplace: none",
            say="嗯，那两家都是做什么的？",
            tail="settle: -",
        ),
        make_reply(
            head="need: 认同\nconfidence: 0.7\n"
                 "direct: 我可能说得太直了\ndisplace: 降级",
            say="刚才那句……是不是有点重。",
            tail="settle: -",
        ),
        make_reply(
            head="need: 只是想说说话\nconfidence: 0.8\n"
                 "direct: 她没事了\ndisplace: none",
            say="嗯，那就好。",
            tail="settle: -",
        ),
    ]
    scripted = ScriptedLLM(replies)

    with Session(cfg, store, scripted,
                 options=SessionConfig(consolidate_interval=999,
                                       consolidate_on_exit=True)) as sess:
        _banner(sess, online=False)
        turns = [
            "今天好累啊，从早忙到晚，路上还堵了一个半小时。",
            "算了不说了。你觉得我该怎么办。",
            "……先不说这个了。你吃饭了吗。",
            "其实我已经投了两家简历了。",
            "有一家是做工业软件的，感觉还行。",
            "……算了，说这些干嘛。",
            # 语域变正式 → 判成受伤。当轮不修复（当场道歉是在处理自己的尴尬）
            "谢谢，我自己想想吧。",
            # 第二拍：这才是该修复的时候
            "嗯，两家都还行。",
            # 接住修复 → 销账
            "没事，我也没往心里去，哈哈。",
        ]
        for text in turns:
            print(f"\n{cyan('你')}  {text}")
            res = sess.say(text)
            print(f"{bold('它')}  {res.say}")
            h = res.parsed.head
            if h:
                print(dim(f"   [need={h.need} conf={h.confidence} "
                          f"displace={h.displace}]"))
            for s in res.settled:
                print(yellow(f"   [结算 {s['id']} → {s['outcome']}"
                             f"（{s.get('label', '')}）]"))
            if res.pending_added:
                print(yellow(f"   [挂起 {res.pending_added['id']}: "
                             f"{res.pending_added['claim']}]"))
            if res.repair_opened:
                print(red("   [发起修复]"))
            if res.repair_closed:
                print(red(f"   [修复结果: {res.repair_closed.get('verdict')}]"))
            for w in res.warnings:
                print(red(f"   [⚠ {w}]"))

    print(dim("\n—— 会话已关闭，后台已跑完最后一次固化 ——"))
    print(dim("\n—— 关系账本 ——"))
    print("  ", json.dumps(store.relation(), ensure_ascii=False))
    print(dim("\n—— 修复账本 ——"))
    for r in store.load_view("repair", {}).get("items", []):
        print(f"   {r['state']:<10} attempts={r['attempts']}  {r['note']}")

    print(dim("\n—— 从账本重建视图 ——"))
    print("  ", rebuild_views(store))

    print(dim("\n—— 指标 ——"))
    from .metrics import compute, format_report
    print(format_report(compute(store.read_ledger())))
    print(dim("\n注：这个脚本刻意演示了五种位移和一次受伤-修复，"
              "\n    所以位移率和受伤数都偏高。"
              "\n    真实对话里绝大多数轮次是 displace: none。"))
    return 0


def cmd_rebuild(args) -> int:
    store = Store(args.data)
    print(json.dumps(rebuild_views(store), ensure_ascii=False, indent=2))
    return 0


def cmd_stats(args) -> int:
    from .metrics import compute, format_report

    store = Store(args.data)
    ledger = store.read_ledger()
    if not ledger:
        print(f"{store.root} 里还没有账本。先跑 python -m persona demo")
        return 1
    print(format_report(compute(ledger)))
    return 0


def cmd_lift(args) -> int:
    """关系增益：有历史 vs 无历史，命中率差多少。

    无历史那一侧不能靠「清空数据再跑一遍」—— 那样样本永远凑不齐，
    而且两次对话不是同一段经历，差值没有意义。

    这里用同一份账本做**首见 vs 非首见**的切分：同一个 claim 第一次出现时
    系统还没有关于它的任何积累，之后才逐步有。差值就是积累的增量。
    """
    from .metrics import compare_with_without_history, compute

    store = Store(args.data)
    ledger = store.read_ledger()
    if not ledger:
        print(f"{store.root} 里还没有账本。先跑 python -m persona demo")
        return 1

    conf_by_id = {r["pending_id"]: r for r in ledger
                  if r.get("kind") == "pending_add"}
    seen_claims: set[str] = set()
    first: list[dict] = []
    later: list[dict] = []

    for rec in ledger:
        if rec.get("kind") != "pending_settle":
            continue
        add = conf_by_id.get(rec.get("pending_id"))
        if add is None:
            continue
        claim = (add.get("claim") or "").strip()
        bucket = later if claim in seen_claims else first
        bucket.append({"kind": "pending_add", "pending_id": rec["pending_id"],
                       "confidence": add.get("confidence", 0.0)})
        bucket.append(rec)
        if claim:
            seen_claims.add(claim)

    full = compute(first + later)
    bare = compute(first)
    out = compare_with_without_history(full, bare)
    print()
    print("关系增益")
    print("─" * 54)
    print(f"  有积累（全部）  {out['with_history']:.0%}  （{out['n_with']} 条）")
    print(f"  首见（无积累）  {out['without_history']:.0%}  （{out['n_without']} 条）")
    print(f"  增益            {out['lift']:+.0%}")
    if not out["reliable"]:
        print("  · 样本不足（两侧各需 ≥10 条），这个数字还不能当结论")
    print()
    return 0


def cmd_calibrate(args) -> int:
    """从账本反推阈值该往哪调。只读，不改。"""
    from .calibrate import collect, format_report

    store = Store(args.data)
    ledger = store.read_ledger()
    if not ledger:
        print(f"{store.root} 里还没有账本。先聊几轮再来。")
        return 1
    print(format_report(collect(ledger)))
    return 0


# ---------------------------------------------------------------- 入口


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="persona-duplex",
        description="双向响应性 · 人格框架",
        epilog="环境变量：PERSONA_API_KEY / PERSONA_BASE_URL / PERSONA_COLOR",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-p", "--persona", default="default",
                       help="人格配置名或 toml 路径")
        p.add_argument("-d", "--data", default="data", help="数据目录")
        p.add_argument("-m", "--model", default=None,
                       help="主模型（默认读 PERSONA_MODEL，否则 gpt-4o-mini）")
        p.add_argument("-j", "--judge-model", default=None,
                       help="话题判定用的轻模型（默认跟随主模型）")
        p.add_argument("--no-judge", action="store_true",
                       help="关掉 LLM 话题判定，退回字面")
        p.add_argument("--protocol", default="text",
                       choices=["text", "tool", "auto", "persona"],
                       help="text=单次生成三段（默认）；"
                            "tool=两阶段，direct 结构上先于 say；"
                            "persona=三层，素回复无人设、人设化单独一步；"
                            "auto=文本失效时自动升级")
        return p

    chat = common(sub.add_parser("chat", help="普通对话"))
    chat.add_argument("--no-stream", action="store_true", help="关掉流式")
    chat.set_defaults(func=cmd_chat)

    common(sub.add_parser("trace", help="带 trace 的对话")).set_defaults(
        func=cmd_trace)
    common(sub.add_parser("demo", help="离线演示")).set_defaults(func=cmd_demo)

    rb = sub.add_parser("rebuild", help="从账本重建视图")
    rb.add_argument("-d", "--data", default="data")
    rb.set_defaults(func=cmd_rebuild)

    st = sub.add_parser("stats", help="从账本算指标")
    st.add_argument("-d", "--data", default="data")
    st.set_defaults(func=cmd_stats)

    lf = sub.add_parser("lift", help="关系增益：有积累 vs 首见")
    lf.add_argument("-d", "--data", default="data")
    lf.set_defaults(func=cmd_lift)

    ca = sub.add_parser("calibrate", help="从账本反推阈值该怎么调")
    ca.add_argument("-d", "--data", default="data")
    ca.set_defaults(func=cmd_calibrate)

    args = ap.parse_args(argv)

    # 模型：命令行 > 环境变量 > 默认
    if getattr(args, "model", None) is None:
        args.model = _default_model()

    return args.func(args)


def _default_model() -> str:
    import os

    env = os.environ.get("PERSONA_MODEL", "").strip()
    if env:
        return env
    # 按 base_url 猜一个合理的默认值，省得用户每次都传 -m
    base = os.environ.get("PERSONA_BASE_URL", "").lower()
    if "deepseek" in base:
        return "deepseek-chat"
    return "gpt-4o-mini"


if __name__ == "__main__":
    sys.exit(main())
