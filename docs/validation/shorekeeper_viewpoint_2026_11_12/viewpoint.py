"""Project already-authored events into a fictional observer's view; no engine or I/O."""
from copy import deepcopy
import json


FIELDS = ("shorekeeper", "traveler")
HEADER = "[本场虚构剧情：守岸人的经历与可确认的告知记录，按发生顺序]"


def _fact(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty authored observation")
    return value


def compile_view(variant, act):
    """Read only the allowed observations in the event prefix, never the author ledger.

    The author must put only things accessible to Shorekeeper in these two fields.
    'traveler' means what she can confirm has been disclosed, not his private thoughts.
    Event IDs and the rest of the variant are intentionally not model inputs.
    """
    events = variant["events"]
    if type(act) is not int or not 1 <= act <= 3:
        raise ValueError("act must be 1, 2 or 3")
    if not isinstance(events, list) or len(events) < act:
        raise ValueError("Event prefix is missing")
    return {"events": [{field: _fact(event[field]) for field in FIELDS}
                       for event in events[:act]]}


def render_view(view):
    """Serialize only a compiled view; an unprojected ledger fails closed."""
    if (not isinstance(view, dict) or set(view) != {"events"} or
            not isinstance(view["events"], list) or not 1 <= len(view["events"]) <= 3):
        raise ValueError("Expected a compiled event view")
    rows = []
    for event in view["events"]:
        if not isinstance(event, dict) or set(event) != set(FIELDS):
            raise ValueError("Unexpected event fields in compiled view")
        rows.append({"守岸人所知或亲历": _fact(event["shorekeeper"]),
                     "漂泊者已获知的情况（守岸人能确认的范围）": _fact(event["traveler"])})
    return HEADER + "\n" + json.dumps(rows, ensure_ascii=False, indent=2)


def role_for_view(base_cfg, view):
    """Return a deep copy of a pristine role config; do not compound prior views."""
    if HEADER in base_cfg.role.identity:
        raise ValueError("Use the base role, not a previously projected role")
    context = render_view(view)
    cfg = deepcopy(base_cfg)
    cfg.role.identity += "\n\n" + context
    return cfg


def selfcheck():
    from types import SimpleNamespace
    v = {"source": "AUTHOR_NAME", "aliases": ["AUTHOR_ALIAS"], "private_nonce": "HIDDEN",
         "events": [{"id": f"id_{i}", "world": f"WORLD_{i}", "shorekeeper": f"ROLE_{i}",
                     "traveler": f"HEARD_{i}"} for i in (1, 2, 3)]}
    old = deepcopy(v)
    base = SimpleNamespace(role=SimpleNamespace(identity="BASE", traits=[1]))
    for act in (1, 2, 3):
        view = compile_view(v, act)
        cfg = role_for_view(base, view)
        changed = deepcopy(v)
        changed["source"] = changed["private_nonce"] = "OTHER_SECRET"
        for i, event in enumerate(changed["events"], 1):
            event["world"] = event["id"] = "OTHER_SECRET"
            event["extra"] = "OTHER_SECRET"
            if i > act:
                event.clear()
        assert compile_view(changed, act) == view
        assert render_view(view) in cfg.role.identity
        assert all(x not in cfg.role.identity for x in ("AUTHOR", "HIDDEN", "WORLD", "id_"))
        assert all((token+str(i) in cfg.role.identity) == (i <= act)
                   for i in (1, 2, 3) for token in ("ROLE_", "HEARD_"))
        cfg.role.traits.append(2)
        view["events"][0]["shorekeeper"] = "edit"
        assert old == v and base.role.identity == "BASE" and base.role.traits == [1]
        try:
            role_for_view(cfg, compile_view(v, 1))
        except ValueError:
            pass
        else:
            raise AssertionError("A role from a later act could leak into an earlier one")
    for bad in (0, 4, 1.0, True):
        try:
            compile_view(v, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid act was accepted")
    try:
        render_view({"events": v["events"]})
    except ValueError:
        pass
    else:
        raise AssertionError("Unprojected fields could enter the rendered prompt")


if __name__ == "__main__":
    selfcheck()
    print("Viewpoint isolation and non-mutation checks passed.")
