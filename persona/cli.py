"""命令行入口。

    chat     普通对话（只显示 say）
    trace    带 trace 的对话（开发用，能看到 HEAD/TAIL）
    demo     离线演示：脚本模型跑一遍全流程，不需要 API key
    rebuild  从账本重建 state/
    stats    从账本算指标
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PersonaConfig
from .llm import ScriptedLLM, make_reply, openai_compatible, openai_stream
from .session import Session, SessionConfig
from .store import Store, rebuild_views

ROOT = Path(__file__).resolve().parent.parent


def _load_cfg(name: str) -> PersonaConfig:
    p = Path(name)
    if not p.exists():
        p = ROOT / "personas" / f"{name}.toml"
    if not p.exists():
        raise SystemExit(f"找不到人格配置: {name}")
    return PersonaConfig.load(p)


def _print_trace(res) -> None:
    h = res.parsed.head
    print(f"\n\033[90m── {res.turn_id} ──\033[0m")
    if h:
        print(f"\033[34mneed\033[0m      {h.need}   \033[34mconf\033[0m {h.confidence}")
        print(f"\033[34mdirect\033[0m    {h.direct}")
        print(f"\033[34mdisplace\033[0m  {h.displace}")
    print(f"\033[34mstance\033[0m    {res.stance}")
    for s in res.settled:
        print(f"\033[33msettle\033[0m    {s['id']} → {s['outcome']}（{s.get('label', '')}）")
    if res.pending_added:
        p = res.pending_added
        print(f"\033[33mpending\033[0m   {p['claim']} ({p['confidence']})")
    if res.repair_opened:
        print(f"\033[31mrepair\033[0m    发起修复（{res.repair_opened.get('action')}）")
    if res.repair_closed:
        print(f"\033[31mrepair\033[0m    结果 {res.repair_closed.get('verdict')}")
    for w in res.warnings:
        print(f"\033[31m⚠\033[0m  {w}")


def _make_session(args, *, streaming: bool) -> Session:
    cfg = _load_cfg(args.persona)
    store = Store(args.data)
    llm = openai_compatible(args.model)
    stream_llm = openai_stream(args.model) if streaming else None
    return Session(
        cfg, store, llm,
        stream_llm=stream_llm,
        options=SessionConfig(stream=streaming),
    )


def cmd_chat(args) -> int:
    sess = _make_session(args, streaming=not args.no_stream)
    _banner(sess, online=True)
    _repl(sess, show_trace=False)
    return 0


def cmd_trace(args) -> int:
    sess = _make_session(args, streaming=False)
    _banner(sess, online=True)
    _repl(sess, show_trace=True)
    return 0


def cmd_demo(args) -> int:
    """离线跑通：不联网，验证协议、结算、修复、后台。"""
    cfg = _load_cfg(args.persona)
    store = Store(args.data)

    replies = [
        # 1 读到潜台词，选择位移
        make_reply(
            head="need: 只想被听\nconfidence: 0.72\n"
                 "direct: 你根本不是在问我，你只是想让人知道你有多累\n"
                 "displace: 降级",
            say="今天是不是特别长啊。",
            tail="settle: -\npending:\n  claim: 用户此刻只想被听\n"
                 "  confidence: 0.7\n  predicts: 下一句会继续描述细节，不会问怎么办",
        ),
        # 2 用户直接问「怎么办」，把预言推翻 —— 当场结算
        make_reply(
            head="need: 要被推一把\nconfidence: 0.62\n"
                 "direct: 你其实早就想辞了，只是在等一个人说可以\n"
                 "displace: 悬置",
            say="那你是想听我说「辞了吧」，还是想让我帮你算算还能撑多久？",
            tail="settle: p_0001 | refuted | 用户直接问怎么办，不是想被听\n"
                 "pending:\n  claim: 用户已经有倾向，需要的是被允许\n"
                 "  confidence: 0.6\n  predicts: 下一句会说出某个具体方案",
        ),
        # 3 用户把话题收回去 —— 三种含义不可区分，挂起
        make_reply(
            head="need: 想被拦住\nconfidence: 0.45\n"
                 "direct: 你每次说到这一步就跑\n"
                 "displace: 截断",
            say="……嗯。吃过了，你呢。",
            tail="settle: -",
        ),
        # 4 话题回来了，延迟结算
        make_reply(
            head="need: 要被推一把\nconfidence: 0.66\n"
                 "direct: 你看，你又绕回来了\ndisplace: 多余",
            say="吃了。不过你刚才那句「先不说这个了」，我记着。",
            tail="settle: p_0002 | confirmed | 用户绕了一轮还是回到这件事\n"
                 "delta:\n  closeness: 0.03\n  unresolved: 辞职这件事还没说完",
        ),
        # 5 普通闲聊，无位移。这是常态。
        make_reply(
            head="need: 只是想说说话\nconfidence: 0.85\n"
                 "direct: 那挺好的\ndisplace: none",
            say="两家都是做哪块的？",
            tail="settle: -",
        ),
        # 6 这一句伤到人了 —— 用户开始变客气
        make_reply(
            head="need: 要被推一把\nconfidence: 0.55\n"
                 "direct: 你就是不敢跟人正面说话\ndisplace: none",
            say="你每次都这样，话说一半就缩回去。",
            tail="settle: -\npending:\n  claim: 用户需要被推一把\n"
                 "  confidence: 0.55\n  predicts: 会反驳，或者沉默",
        ),
        # 7 修复：短，不解释
        make_reply(
            head="need: 认同\nconfidence: 0.7\n"
                 "direct: 我可能说得太直了\ndisplace: 降级",
            say="刚才那句……是不是有点重。",
            tail="settle: -",
        ),
        # 8 修复被接受
        make_reply(
            head="need: 只是想说说话\nconfidence: 0.8\n"
                 "direct: 那就好\ndisplace: none",
            say="嗯，那两家都是做什么的？",
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
            "谢谢，我自己想想吧。",
            "嗯，两家都还行。",
        ]
        for text in turns:
            print(f"\n\033[36m你\033[0m  {text}")
            res = sess.say(text)
            print(f"\033[1m它\033[0m  {res.say}")
            if res.parsed.head:
                h = res.parsed.head
                print(f"   \033[90m[need={h.need} conf={h.confidence} "
                      f"displace={h.displace}]\033[0m")
            for s in res.settled:
                print(f"   \033[33m[结算 {s['id']} → {s['outcome']}"
                      f"（{s.get('label', '')}）]\033[0m")
            if res.pending_added:
                print(f"   \033[33m[挂起 {res.pending_added['id']}: "
                      f"{res.pending_added['claim']}]\033[0m")
            if res.repair_opened:
                print(f"   \033[31m[发起修复]\033[0m")
            if res.repair_closed:
                print(f"   \033[31m[修复结果: {res.repair_closed.get('verdict')}]\033[0m")
            for w in res.warnings:
                print(f"   \033[31m[⚠ {w}]\033[0m")

    print("\n\033[90m—— 会话已关闭，后台已跑完最后一次固化 ——\033[0m")
    print("\n\033[90m—— 关系账本 ——\033[0m")
    print("  ", json.dumps(store.relation(), ensure_ascii=False))
    print("\n\033[90m—— 修复账本 ——\033[0m")
    for r in store.load_view("repair", {}).get("items", []):
        print(f"   {r['state']:<10} attempts={r['attempts']}  {r['note']}")

    print("\n\033[90m—— 从账本重建视图 ——\033[0m")
    print("  ", rebuild_views(store))

    print("\n\033[90m—— 指标 ——\033[0m")
    from .metrics import compute, format_report
    print(format_report(compute(store.read_ledger())))
    print("\n\033[90m注：这个脚本刻意演示了五种位移和一次受伤-修复，"
          "\n    所以位移率和受伤数都偏高。真实对话里绝大多数轮次是 displace: none。\033[0m")
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


def _banner(sess: Session, online: bool) -> None:
    cfg = sess.engine.cfg
    r = cfg.role
    print("\033[90m" + "─" * 54 + "\033[0m")
    print(f"人格  {r.name}   bias={r.bias:+.2f}  "
          f"attention={r.attention}  legibility={r.legibility_max}")
    print(f"说明  {r.voice}")
    print(f"模式  {'在线' if online else '离线演示'}   数据 {sess.store.root}")
    print("\033[90m输入空行或 Ctrl-C 退出\033[0m")
    print("\033[90m" + "─" * 54 + "\033[0m")


def _repl(sess: Session, show_trace: bool) -> None:
    streaming = sess.opts.stream
    try:
        while True:
            try:
                text = input("\n\033[36m你\033[0m  ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                return

            try:
                if show_trace:
                    res = sess.say(text)
                    _print_trace(res)
                    print(f"\n\033[1m{res.say}\033[0m")
                elif streaming:
                    print("\033[1m它\033[0m  ", end="", flush=True)
                    collected = []

                    def sink(piece: str) -> None:
                        collected.append(piece)
                        print(piece, end="", flush=True)

                    sess.say(text, on_delta=sink)
                    print()
                else:
                    res = sess.say(text)
                    print(f"\033[1m它\033[0m  {res.say}")
            except Exception as e:              # 生成失败不崩
                print(f"\033[31m生成失败：{e}\033[0m")
    finally:
        sess.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="persona-duplex",
                                 description="双向响应性 · 人格框架")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-p", "--persona", default="default",
                       help="人格配置名或 toml 路径")
        p.add_argument("-d", "--data", default="data", help="数据目录")
        p.add_argument("-m", "--model", default="gpt-4o-mini")
        return p

    common(sub.add_parser("chat", help="普通对话")).set_defaults(func=cmd_chat)
    chat = sub.choices["chat"]
    chat.add_argument("--no-stream", action="store_true", help="关掉流式")

    common(sub.add_parser("trace", help="带 trace 的对话")).set_defaults(func=cmd_trace)
    common(sub.add_parser("demo", help="离线演示")).set_defaults(func=cmd_demo)

    rb = sub.add_parser("rebuild", help="从账本重建视图")
    rb.add_argument("-d", "--data", default="data")
    rb.set_defaults(func=cmd_rebuild)

    st = sub.add_parser("stats", help="从账本算指标")
    st.add_argument("-d", "--data", default="data")
    st.set_defaults(func=cmd_stats)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
