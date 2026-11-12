"""Real free-turn Shorekeeper trial. Private local state only; never part of the study archive."""
import argparse
import getpass
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona.config import PersonaConfig
from persona.llm import openai_compatible
from persona.session import Session, SessionConfig
from persona.store import Store


def main():
    default = Path(os.getenv("LOCALAPPDATA") or (Path.home()/".local"/"share"))/"persona-duplex"/"private-trials"/"shorekeeper-continuity"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=default)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", help="Send one user turn and close; later invocations restore this private session.")
    mode.add_argument("--input-file", type=Path, help="One UTF-8 utterance from a private local file.")
    p.add_argument("--context-file", type=Path, help="Private scene background, saved once as context rather than spoken dialogue.")
    a = p.parse_args()
    if a.store.resolve().is_relative_to(ROOT) or a.store.resolve().is_relative_to(HERE):
        p.error("Choose a private store outside the repository.")
    a.store.mkdir(parents=True, exist_ok=True)
    context_path = a.store / "scene-context.txt"
    if a.context_file:
        ref = a.context_file.read_text(encoding="utf-8-sig").strip()
        if context_path.exists() and context_path.read_text(encoding="utf-8") != ref:
            p.error("Existing scene context differs. Start a separate private session for a new setting.")
        if not context_path.exists():
            context_path.write_bytes(ref.encode())
    secret = os.getenv("PERSONA_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("Provider credential: ")
    api = openai_compatible("deepseek-chat", base_url="https://api.deepseek.com/v1", api_key=secret,
                            temperature=0.7, max_tokens=1100, timeout=90)
    cfg = PersonaConfig.load(ROOT/"personas/shorekeeper.toml")
    if context_path.exists():
        cfg.role.identity += "\n\n[本场故事背景：用户提供的剧情资料，不是本轮说出口的话]\n" + context_path.read_text(encoding="utf-8")
    with Session(cfg, Store(a.store), api, options=SessionConfig(memory_mode="frozen", consolidate_on_exit=False)) as s:
        if a.input_file:
            a.once = a.input_file.read_text(encoding="utf-8-sig").strip()
        if a.once is not None:
            if not a.once.strip(): p.error("Expected a nonempty user turn")
            print(s.say(a.once).say, flush=True)
            return
        print("守岸人 · 私有试聊｜输入 /quit 结束；下次使用同一私有目录可续聊。", flush=True)
        while True:
            try:
                line = input("你：").strip()
            except EOFError:
                break
            if line == "/quit": break
            if line: print("守岸人："+s.say(line).say, flush=True)


if __name__ == "__main__":
    main()
