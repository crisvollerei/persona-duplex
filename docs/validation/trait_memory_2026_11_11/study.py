"""Offline comparison of memory semantics; all attribution labels are authored fixtures."""
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona.config import PersonaConfig, Trait
from persona.consolidate import Consolidator
from persona.store import Store, replay_traits


@dataclass(frozen=True)
class Note:
    id: str
    episode: str
    relation: str
    reaction: str
    votes: tuple
    replaces: str = ""


I, W = "intervention", "warmth"
EARLY = "那个小办法我试了，帮我把散在桌上的材料分好了。"
LATER = "你提醒我留个余量，我这次真来得及收拾。"
VOID = "刚才那两次整理的效果是同事的方法，不是我们试的办法。"
BASE = PersonaConfig.load(ROOT/"personas/shorekeeper.toml").role.traits_state()


def note(id, episode, reaction, votes, relation="甲", replaces=""):
    return Note(id, episode, relation, reaction, tuple(votes), replaces)


A = note("a","e1",EARLY,[(I,1)])
B = note("b","e2",LATER,[(I,1)])
DUP = note("a2","e1",EARLY,[(I,1)])
EDIT_A = note("r1","e3",VOID,[],replaces="a")
EDIT_B = note("r2","e3",VOID,[],replaces="b")
NEG_A = note("c","e4","这次多给的安排打断了我本来要说的事。",[(I,-1)])
NEG_B = note("d","e5","第二次也是，那个额外步骤让我停下来了。",[(I,-1)])
B_OTHER = note("b","e2",LATER,[(I,1)],relation="乙")
MULTI = note("a","e1",EARLY,[(I,1),(W,1)])
WARM = note("w","e6","你记得给我留一盏灯，我有被照顾到。",[(W,1)])

# Each tuple is an explicit processing boundary. The same episode key denotes the
# same originating experience, not just similar text. Confidence fixed at 0.75.
CASES = {
    "independent_then_idle": [(A,), (B,), ()],
    "two_notes_one_episode": [(A,), (DUP,)],
    "correction_after_push": [(A,), (B,), (EDIT_A, EDIT_B)],
    "later_opposite_experiences": [(A,), (B,), (NEG_A,), (NEG_B,)],
    "two_relations_one_store": [(A,), (B_OTHER,)],
    "one_episode_two_traits": [(MULTI,), (B,), (WARM,)],
    "four_independent_one_flush": [(A,B,note("c","e4","第三次这个提醒也派上用场。",[(I,1)]),
                                    note("d","e5","第四次先想一步省下了收尾时间。",[(I,1)]))],
    "four_independent_four_flushes": [(A,),(B,),(note("c","e4","第三次这个提醒也派上用场。",[(I,1)]),),
                                      (note("d","e5","第四次先想一步省下了收尾时间。",[(I,1)]),)],
}


def rounded(state):
    return {k:round(state[k]["value"],8) for k in (I,W)}


def existing(batches):
    """Use public, opt-in update APIs; no consolidator worker or model invoked."""
    with tempfile.TemporaryDirectory() as tmp:
        store=Store(tmp)
        store.init_traits(BASE,persona="守岸人")
        known={}; turns=set(); steps=[]
        for batch in batches:
            for x in batch:
                if x.episode not in turns:
                    store.append("turn",dict(turn_id=x.episode,user=x.reaction))
                    turns.add(x.episode)
                j=store.add_judgment(f"合成的经历归因 {x.id}",.75,[x.episode],
                    affects=dict(x.votes),memory_scope=x.relation,supersedes=known.get(x.replaces),
                    evidence_quotes=[dict(turn_id=x.episode,quote=x.reaction,evidence_scope="specified")])
                if x.replaces:assert j["supersedes"]==known[x.replaces]
                known[x.id]=j["id"]
            Consolidator(store).push_traits()
            steps.append(rounded(store.traits()))
            store=Store(tmp)   # Reopen between processing boundaries.
        replay=rounded(replay_traits(store.read_ledger()))
        assert replay==rounded(store.traits())
        return dict(steps=steps,replayed=replay,
                    consumed_judgments=store.cursor("judgment","ids") or [],
                    active_judgments=[r["id"] for r in store.user_judgments(limit=None)])


def project(batches):
    """Research reference: recompute current state from valid attributions in event order.

    The source episode order, not the batch/interpretation order, sets the clock.
    An ambiguous direction on one source gets no vote. No automatic global promotion.
    """
    notes={}; inactive=set(); source_order={}; traces={}; all_steps=[]
    for batch in batches:
        for x in batch:
            if x.id in notes and notes[x.id]!=x:raise ValueError("conflicting note id")
            if x.replaces and (x.replaces not in notes or notes[x.replaces].relation!=x.relation):
                raise ValueError("missing or cross-relationship correction target")
            notes[x.id]=x
            source_order.setdefault((x.relation,x.episode),len(source_order))
            if x.replaces:inactive.add(x.replaces)
        by_source=defaultdict(set)
        for k,x in notes.items():
            if k not in inactive:
                for trait,direction in x.votes:
                    by_source[x.relation,trait,x.episode].add(direction)
        state={}; pending=defaultdict(list); changes=defaultdict(list)
        for (relation,trait,ep),votes in sorted(by_source.items(),key=lambda item:source_order[item[0][0],item[0][2]]):
            if len(votes)!=1:continue
            key=(relation,trait)
            t=state.setdefault(key,Trait.from_state(BASE[trait]))
            evidence=pending[key]
            evidence.append((ep,next(iter(votes))))
            score=.75*sum(d for _,d in evidence)
            if abs(score)>=1.4:
                delta=t.nudge(1 if score>0 else -1)
                if delta:
                    changes[key].append(dict(episodes=[e for e,_ in evidence],direction=1 if score>0 else -1))
                    evidence.clear()
        traces={rel:{k:changes[rel,k] for k in (I,W)} for rel in ("甲","乙")}
        all_steps.append({rel:{k:round(state.get((rel,k),Trait.from_state(BASE[k])).value,8)
                           for k in (I,W)} for rel in ("甲","乙")})
    return dict(steps=all_steps,active_change_sources=traces)


def main():
    data={"base":rounded(BASE),"scope":"offline authored attribution labels; no generated dialogue",
          "cases":{}}
    for name,batches in CASES.items():
        data["cases"][name]=dict(existing=existing(batches),reference=project(batches))
    ref=lambda case: data["cases"][case]["reference"]
    current=lambda case: data["cases"][case]["existing"]
    assert current("independent_then_idle")["steps"][-2:]==[current("independent_then_idle")["replayed"]]*2
    assert current("two_notes_one_episode")["replayed"][I]>BASE[I]["value"]
    assert current("correction_after_push")["active_judgments"]==["j_0003","j_0004"]
    assert current("correction_after_push")["replayed"][I]>BASE[I]["value"]
    assert current("two_relations_one_store")["replayed"][I]>BASE[I]["value"]
    assert current("one_episode_two_traits")["replayed"][W]==BASE[W]["value"]
    for case in ("two_notes_one_episode","correction_after_push","two_relations_one_store"):
        assert all(s==data["base"] for s in ref(case)["steps"][-1].values())
    assert ref("one_episode_two_traits")["steps"][-1]["甲"][W]>BASE[W]["value"]
    assert [c["direction"] for c in ref("later_opposite_experiences")["active_change_sources"]["甲"][I]]==[1,-1]
    assert ref("four_independent_one_flush")["steps"][-1]==ref("four_independent_four_flushes")["steps"][-1]
    assert current("four_independent_one_flush")["replayed"]!=current("four_independent_four_flushes")["replayed"]
    out=json.dumps(data,ensure_ascii=False,indent=2)+"\n"
    path=HERE/"results.json"
    if "--check" in sys.argv:
        assert json.loads(out)==json.loads(path.read_bytes())
    else:path.write_bytes(out.encode())
    for case,v in data["cases"].items():
        print(case,v["existing"]["replayed"],v["reference"]["steps"][-1]["甲"])


if __name__=="__main__":
    main()
