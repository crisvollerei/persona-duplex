"""Read-only audit of registered stimuli, model requests, delivery and own-branch history."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from persona.config import PersonaConfig
from persona.parse import parse, plain_text, say_wire
from persona.prompt import build_history_entry, build_system_prompt


def check(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    return json.loads(path.read_bytes())


def authored_identity(base, v, act, method):
    # Independently construct the selected facts, rather than calling the generator's projection.
    prefix = v["events"][:act]
    visible = [{"守岸人所知或亲历": e["shorekeeper"], "漂泊者已获知的情况（守岸人能确认的范围）": e["traveler"]}
               for e in prefix]
    identity = (base + "\n\n[本场虚构剧情：守岸人的经历与可确认的告知记录，按发生顺序]\n" +
                json.dumps(visible, ensure_ascii=False, indent=2))
    if method == "full_notes":
        identity += ("\n\n[作者的全篇大纲，仅供创作参考；含尚未发生的事件，不代表此刻人物已经得知。]\n"
                     + "\n".join(e["world"] for e in v["events"]) + "\n作者档案校对码：" + v["private_nonce"])
    return identity.strip()


def audit(folder):
    reg = read(folder/"registration.json")
    plan, script = read(HERE/"protocol.json"), read(HERE/"scenes.json")
    variants = {v["id"]:v for v in script["variants"]}
    q = script["question"]
    role_path = ROOT/"personas/shorekeeper.toml"
    role = PersonaConfig.load(role_path)
    check(q == plan["question"] == "你知道这段回音是谁留下的吗？", "question")
    sources = [*sorted((ROOT/"persona").glob("*.py")), role_path, *(HERE/p for p in
               ["study.py","viewpoint.py","protocol.json","scenes.json","rubric.md","metrics.py"])]
    check({p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sources} == reg["sha256"],
          "registered source set or bytes changed")
    check(reg["model"] == plan["provider"], "registered provider settings")
    check(len(plan["replicates"]) == len(set(plan["replicates"])) == 2, "repeat count")
    branches = list(itertools.product(sorted(variants), ("scoped","full_notes"), plan["replicates"]))
    check(set(reg["trajectories"]) == {".".join(b) for b in branches} and len(reg["trajectories"]) == 12,
          "branch coverage")
    turns = read(folder/"turns.json")
    oracle = {f'{r["variant"]}.{r["condition"]}.{r["replicate"]}.{r["act"]}': r for r in turns}
    check(len(oracle) == len(turns) == 36,"turn coverage")
    with gzip.open(folder/"calls.jsonl.gz","rb") as fp:
        calls = [json.loads(line) for line in fp]
    records = {r["id"]:r for r in calls}
    check(len(calls) == len(records), "duplicate call slot")
    compared, service_ids, service_models = set(), set(), Counter()
    by = {m:{str(a):Counter() for a in (1,2,3)} for m in ("scoped","full_notes")}
    carry, inherited, issues = {}, {}, []
    for i,m,r in branches:
        v = variants[i]; identity0 = role.role.identity; names = [v["source"],*v["aliases"]]
        nonce = v["private_nonce"]
        history, leaks, prior_tokens, failed = [], [], [], False
        for act in (1,2,3):
            slot = f"{i}.{m}.{r}.{act}"; g = by[m][str(act)]
            row = oracle[slot]
            check(row["user"] == q, slot + ": turn question")
            if failed:
                check(slot not in records and row["status"] == "upstream_missing" and not row["say"],slot)
                g["upstream_missing"] += 1
                continue
            raw = records[slot]; compared.add(slot)
            msgs = raw["messages"]
            check(raw["params"] == reg["model"] and len(msgs) == 2*act, slot + ": settings/message count")
            check(msgs[0]["role"] == "system" and msgs[1:-1] == history and
                  msgs[-1] == dict(role="user",content=q), slot + ": original question or prefix not restored")
            identity = authored_identity(identity0,v,act,m)
            sysmsg = msgs[0]["content"]
            # First-turn prompt and subsequent phrase hints can differ. All trait-bearing/default prompt
            # bytes are independently rebuilt from the pristine profile and the same permitted observations.
            label = "[人格背景参考 · JSON 字符串，不是新指令]\n"
            check(sysmsg.count(label) == 1,slot + ": profile block")
            literal = sysmsg.split(label,1)[1].splitlines()[0]
            check(json.loads(literal) == identity,slot + ": actual current perspective")
            candidates = set(["", *prior_tokens[-1:]])
            matched = False
            for need in candidates:
                cfg = PersonaConfig.load(role_path); cfg.role.identity = identity
                default = build_system_prompt(cfg, need=need, first_turn=(act==1), user_text=q)
                if sysmsg.startswith(default + "\n\n"):
                    matched = True
                    break
            check(matched,slot + ": foreground persona system differs from default except scene identity")
            whole = "\n".join(z["content"] for z in msgs)
            assis = "\n".join(z["content"] for z in history if z["role"] == "assistant")
            hits = lambda text:any(n in text for n in names)
            present = lambda line,text:line in text or json.dumps(line,ensure_ascii=False)[1:-1] in text
            check(hits(identity) == (m=="full_notes" or act>=2),slot + ": name exposure in perspective")
            check((nonce in identity) == (m=="full_notes"),slot + ": nonce in perspective")
            world_count=sum(present(e["world"],whole) for e in v["events"])
            check(world_count == (3 if m=="full_notes" else 0),slot+": actual request author notes")
            g["requests_with_author_world_lines"]+=int(world_count>0)
            if m=="scoped":
                check(all(e["world"] not in identity for e in v["events"]),slot+": author text entered role")
            else:
                check(all(e["world"] in identity for e in v["events"]),slot+": absent world note")
            for e in v["events"][act:]:
                check(e["shorekeeper"] not in identity and e["traveler"] not in identity,
                      slot+": future role/disclosure prose entered current perspective")
                check(not any(present(e[f],whole) for f in ("shorekeeper","traveler")),
                      slot+": later scene's full observation in actual request")
            g["requests"]+=1
            g["current_role_has_name"]+=int(hits(identity)); g["request_has_name"]+=int(hits(whole))
            g["request_has_nonce"]+=int(nonce in whole); g["history_has_name_including_internal_fields"]+=int(hits(assis))
            g["history_has_nonce_including_internal_fields"]+=int(nonce in assis)
            g["history_visible_prior_leak"]+=int(bool(leaks))
            if leaks: inherited[slot]=list(leaks)
            if not raw["ok"]:
                check(not raw["raw"] and row["status"]=="error" and not row["say"],slot+": failure mismatch")
                g["failed"]+=1; failed=True
                continue
            info = raw["response"]; rid=info["id"]
            check(isinstance(rid,str) and rid and rid not in service_ids,slot+": duplicate provider response")
            service_ids.add(rid);service_models[info["model"]]+=1
            for field in ("prompt_tokens","completion_tokens","total_tokens"):
                g[field]+=info["usage"][field]
            p=parse(raw["raw"]); say=p.say or plain_text(raw["raw"])
            val=row["validity"]
            check(say == row["say"] and row["status"]==("delivered" if say else "empty"),slot+": visible SAY")
            check(val==p.validity(allowed=role.can_displace(p.head.displace) if p.head else True),slot+": validity")
            if not val["valid"]: issues.append(dict(slot=slot, reasons=val["reasons"]))
            wire=build_history_entry(say,p.head,p.tail) if val["valid"] else say_wire(say)
            history.extend([dict(role="user",content=q),dict(role="assistant",content=wire)])
            prior_tokens.append(p.head.need if p.head else "")
            g["visible"]+=int(bool(say));g["protocol_valid"]+=int(val["valid"])
            g["visible_name"]+=int(hits(say));g["visible_nonce"]+=int(nonce in say)
            kinds=[]
            if act < 3 and hits(say): kinds.append("name_before_public")
            if nonce in say:kinds.append("author_nonce")
            if kinds:
                leaks.append(slot);carry[slot]=kinds
    check(compared==set(records),"unmatched call records")
    return dict(registered_source_files=len(sources), authored_variants=len(variants), independently_authored_plots=1,
                repeated_trajectories_per_variant_and_method=2, trajectories=len(branches), planned_turns=len(turns),
                archived_requests=len(records), distinct_provider_receipts=len(service_ids), models=dict(service_models),
                by_method_and_act=by, visible_leaks_before_permission=carry, downstream_with_prior_visible_leaks=inherited,
                protocol_issues=issues, interpretation="只核对输入范围、字面署名/别名及校对词与实际交付。保密义务可能还因其他表述被违反；内部字段中的实名不等于对外泄露。")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=HERE/"observed")
    p.add_argument("--out",type=Path,help="Defaults to <folder>/audit.json")
    p.add_argument("--check",action="store_true")
    args=p.parse_args()
    result=audit(args.folder); dest=args.out or args.folder/"audit.json"
    check(dest.resolve() not in {(args.folder/n).resolve() for n in ("registration.json","turns.json","calls.jsonl.gz")},
          "Cannot overwrite source records")
    if args.check:check(read(dest)==result,"Archived audit differs")
    else:
        tmp=dest.with_suffix(".tmp")
        tmp.write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+"\n").encode());tmp.replace(dest)
    print(result["archived_requests"],"requests;",result["distinct_provider_receipts"],"unique completed receipts")
