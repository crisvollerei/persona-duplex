"""Study-only: admission of one dated arrival-memory record, given structured task facts."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Gate:
    action: str
    visible: str = ""

def admit(actor, subject, scope, requires_arrival, read):
    # No ledger callback before context, subject, and domain are established.
    if not actor or not subject or scope != (actor,subject,"arrival"):
        return Gate("omit")
    if requires_arrival is None:
        return Gate("confirm_arrival")
    if requires_arrival is False:
        return Gate("omit")
    if requires_arrival is not True:
        raise ValueError("requires_arrival must be True, False or None")
    record=read("V")
    live=next((r for r in record["versions"] if r["id"]==record["active"]),{})
    if live.get("memory_scope")!=".".join(scope) or not live.get("evidence_ids"):
        return Gate("omit")
    return Gate("use",record["visible"])

def check_gate():
    def forbidden(_):
        raise AssertionError("unnecessary memory lookup")
    scope=("tang","yu","arrival")
    assert admit("tang","yu",scope,False,forbidden)==Gate("omit")
    assert admit("tang","yu",scope,None,forbidden)==Gate("confirm_arrival")
    for actor,person,sc in (("yu","yu",scope),("tang","other",scope),("tang","yu",("tang","yu","onsite"))):
        assert admit(actor,person,sc,True,forbidden)==Gate("omit")
    empty=lambda m:{"active":None,"versions":[],"visible":"不得带出"}
    assert admit("tang","yu",scope,True,empty)==Gate("omit")
    wrong=lambda m:{"active":"j2","versions":[dict(id="j2",memory_scope="yu.tang.arrival",evidence_ids=["x"])],"visible":"不得带出"}
    assert admit("tang","yu",scope,True,wrong)==Gate("omit")
    try:admit("tang","yu",scope,"unknown",forbidden)
    except ValueError:pass
    else:raise AssertionError("invalid state accepted")
