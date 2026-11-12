"""Research-only snapshot for the three authored branches in ../scenes.json.

The three versions share this timeline. These are author-supplied facts, not a
knowledge estimator. render_card never reads a source name, hidden note or chat.
"""
from hashlib import sha256
import json
from pathlib import Path


HEADER = "[此刻状态：这段回音的留音者身份]"
FIELDS = ("守岸人是否知情", "漂泊者是否知情", "关于该身份的保密约定")
STATES = (
    ("未确认", "未确认", "无生效约定"),
    ("已核验并确认", "尚不知情", "生效中；至留音者本人向漂泊者说明为止"),
    ("已核验并确认", "已听本人说明并确认", "已到期"),
)


def render_card(act: int) -> str:
    """Return just the current facts, with no names, act numbers or response instructions."""
    if type(act) is not int or not 1 <= act <= len(STATES):
        raise ValueError("Expected an act numbered 1 through 3")
    facts = (f"{field}：{value}" for field, value in zip(FIELDS, STATES[act - 1], strict=True))
    return "\n".join((HEADER, *facts))


def selfcheck():
    scene_bytes = (Path(__file__).resolve().parent.parent / "scenes.json").read_bytes()
    # If the authored timeline changes, manually verify alignment before using these snapshots.
    assert sha256(scene_bytes).hexdigest() == "e6dfa84145e2ab680132cb4dadcbfd74d6a5662e8c136a7cb53f6be3a33a0b0f"
    scenes = json.loads(scene_bytes)
    cards = [render_card(n) for n in (1, 2, 3)]
    assert len(set(cards)) == 3
    for variant in scenes["variants"]:
        assert len(variant["events"]) == 3
        for card in cards:
            assert len(card.splitlines()) == 4
            assert all(x not in card for x in (variant["source"], *variant["aliases"], variant["private_nonce"]))
        assert variant["source"] in variant["events"][1]["shorekeeper"]
    for invalid in (False, True, 0, 4, 2.0, "2"):
        try:
            render_card(invalid)
        except ValueError:
            continue
        raise AssertionError(f"Accepted invalid act: {invalid!r}")


if __name__ == "__main__":
    selfcheck()
    print("Three authored snapshots checked. No names, aliases or author canaries in the cards.")
