"""Prepare one perspective at a time. No provider, transcript or answer key is required."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OTHER = {"tang": "yu", "yu": "tang"}


def card(world, side, scene):
    """Only this side's private information may pass this boundary."""
    me, partner = world["roles"][side], world["roles"][OTHER[side]]
    detail = scene[side]
    return {
        "name": me["name"], "partner": partner["name"],
        "bio": me["public"], "background": me["private"],
        "partner_bio": partner["public"], "when": scene["when"],
        "public": scene["public"], "entry": detail["entry"], "private": detail["private"],
    }


def player_text(world, side, scene):
    c = card(world, side, scene)
    return (f'**{scene["when"]}｜{scene["title"]}**\n\n'
            f'{c["entry"]}\n\n现场：{c["public"]}\n\n'
            f'我还知道：{c["private"]}\n\n'
            f'可以这样开口：“{scene[side]["opening"]}”\n')


def book(world, side):
    me, partner = world["roles"][side], world["roles"][OTHER[side]]
    scenes = "\n".join(
        f'<details>\n<summary>第 {int(s["id"])} 场</summary>\n\n'
        f'{player_text(world, side, s)}\n</details>\n' for s in world["scenes"])
    return (f'# {me["name"]}的视角 · 对面是{partner["name"]}\n\n'
            f'你来做{me["name"]}，AI 接{partner["name"]}的戏。{me["public"]}\n\n'
            f'出门之前，你自己心里有数：{me["private"]}\n\n'
            '第一次玩只开第一场。每场可以多说几轮，也可以什么都不答。开场台词随意改；'
            '内心独白和私情不必告诉对方。有人已经另作了主意，就沿着实际发生的事接，'
            '相冲突的场景可以跳过。不要把这本玩家册发给对面的 AI。\n\n' + scenes)


def actor_text(world, player, scene):
    actor = OTHER[player]
    c = card(world, actor, scene)
    return (f'虚构对话。你写{c["name"]}的言行，玩家决定{c["partner"]}的言行。\n'
            f'本人：{c["bio"]}\n自己的处境：{c["background"]}\n'
            f'你对对方已知的背景：{c["partner_bio"]}\n'
            f'时间：{c["when"]}\n两人都能看到或听到：{c["public"]}\n'
            f'此刻从你的位置看：{c["entry"]}\n你自己还知道：{c["private"]}\n'
            '等对方真正开口再接。此前有无约定以这条对话中实际发生的事为准。')


def actor_config(world, player, scene):
    """Build a real project config for the fixed-character condition. No growth by default."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from persona.config import PersonaConfig, Trait
    actor = world["roles"][OTHER[player]]
    cfg = PersonaConfig.load(ROOT / "personas/default.toml")
    cfg.role.name, cfg.role.voice = actor["name"], actor["voice"]
    cfg.role.identity = actor_text(world, player, scene)
    cfg.role.traits = {key: Trait(anchor=value) for key, value in actor["traits"].items()}
    cfg.validate()
    return cfg


def checks(world):
    scenes = world["scenes"]
    assert [s["id"] for s in scenes] == [f"{i:02}" for i in range(1, 10)]
    assert len(re.findall(r"^## ", (HERE / "novel.md").read_text(encoding="utf8"), re.M)) == len(scenes)
    for side in OTHER:
        assert (HERE / f"{side}_view.md").read_text(encoding="utf8") == book(world, side)
    # Tag every field separately: a copy of a wrong lane cannot hide in shared vocabulary.
    w = deepcopy(world)
    for side, role in w["roles"].items():
        role["private"] += f" [bio-{side}]"
    for s in w["scenes"]:
        s["public"] += f' [event-{s["id"]}]'
        for side in OTHER:
            for key in ("entry", "private", "opening"):
                s[side][key] += f' [{s["id"]}-{side}-{key}]'
    from sys import path
    if str(ROOT) not in path:
        path.insert(0, str(ROOT))
    from persona.llm import make_reply
    from persona.session import Session, SessionConfig
    from persona.store import Store
    turns = 0
    for s in w["scenes"]:
        for player, actor in OTHER.items():
            cfg = actor_config(w, player, s)
            seen = []

            def scripted(messages):
                seen.append(messages)
                return make_reply("need: 只是想说说话\nconfidence: 0.5\ndirect: 在。\nbaseline: 我在。\ndisplace: none", "我在。")

            with TemporaryDirectory() as tmp, Session(cfg, Store(tmp), scripted,
                    options=SessionConfig(memory_mode="frozen", consolidate_on_exit=False)) as session:
                assert session.chat("我刚进门。") == "我在。"
            assert len(seen) == 1
            prompt = json.dumps(seen[0], ensure_ascii=False)
            page = player_text(w, player, s)
            assert f"[bio-{actor}]" in prompt and f"[bio-{player}]" not in prompt
            for t in w["scenes"]:
                for lane in OTHER:
                    for k in ("private", "entry", "opening"):
                        tag = f'[{t["id"]}-{lane}-{k}]'
                        assert (tag in prompt) == (t is s and lane == actor and k != "opening")
                        assert (tag in page) == (t is s and lane == player)
                tag = f'[event-{t["id"]}]'
                assert (tag in prompt) == (tag in page) == (t is s)
            turns += 1
    print(f"{len(scenes)} scenes, 2 player books, {turns} single-call prompt captures; no live API.")


def main():
    ap = argparse.ArgumentParser(description="《人站到海里去》：分视角发场景，不调用模型。")
    ap.add_argument("mode", choices=("player", "actor", "build", "check"))
    ap.add_argument("--player", choices=tuple(OTHER), help="真人的角色：tang 唐荔，yu 俞川")
    ap.add_argument("--scene", choices=[f"{i:02}" for i in range(1, 10)])
    args = ap.parse_args()
    world = json.loads((HERE / "world.json").read_bytes())
    if args.mode == "check":
        checks(world)
    elif args.mode == "build":
        for side in OTHER:
            (HERE / f"{side}_view.md").write_bytes(book(world, side).encode())
    else:
        if not (args.player and args.scene):
            ap.error("请指定 --player 和 --scene")
        s = next(x for x in world["scenes"] if x["id"] == args.scene)
        if args.mode == "player":
            me = world["roles"][args.player]
            print(f'你是{me["name"]}。{me["public"]}\n{me["private"]}\n')
            print(player_text(world, args.player, s))
        else:
            print(actor_text(world, args.player, s))
            print("说话习惯：" + world["roles"][OTHER[args.player]]["voice"])


if __name__ == "__main__":
    main()
