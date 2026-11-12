"""Local human-listener study. Serving never exposes the source tree or scored keys."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import secrets
import threading
from uuid import uuid4

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "reference_growth"
PRIVATE = HERE.parents[3] / "data" / "listener_reference_growth_v1"
SCENES = json.loads((PREV / "scenes.json").read_bytes())
SAMPLES = json.loads((PREV / "observed/probes.json").read_bytes())
OPTIONS = json.loads((HERE / "options.json").read_bytes())
NAMES = {"tang": "唐荔", "yu": "俞川"}
PHASES = ("free", "action", "motive")
BASE_ROLES = ("yu", "yu", "tang", "tang", "yu", "tang")
CHOICES = ("A", "B", "unclear", "other")
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + "." + uuid4().hex + ".tmp")
    tmp.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode())
    tmp.replace(p)


def digest(x):
    return hashlib.sha256(json.dumps(x, separators=(",", ":"), ensure_ascii=False,
                                     sort_keys=True).encode()).hexdigest()


def form(number):
    if not 1 <= number <= 12:
        raise ValueError("卷号应为 1 至 12")
    f = number - 1
    counts = Counter()
    items = []
    for i, (q, first) in enumerate(zip(SCENES, BASE_ROLES, strict=True)):
        role = first if f < 6 else ("yu" if first == "tang" else "tang")
        arm = ("G", "F", "S")[(counts[role] + f) % 3]
        counts[role] += 1
        rep = 1 + (f // 3 + i) % 2
        k = f"{role}.{arm}.{q['id']}.{rep}"
        raw = SAMPLES[k]
        item = dict(source=k, scene=q["id"], role=role, arm=arm, public=q["public"],
                    partner=q["line"], say=raw["say"], valid=raw["validity"]["valid"])
        for j, phase in enumerate(("action", "motive")):
            pair = OPTIONS[q["id"]][role][j]
            # Distinct concealed patterns, balanced within each form and for each scene across forms.
            order = ((0, 1, 0, 1, 1, 0), (1, 1, 0, 0, 1, 0))[j]
            flip = (order[i] + f // (1 if j == 0 else 2)) % 2
            item[phase] = {"choices": dict(zip(("A", "B"), pair[::-1] if flip else pair, strict=True)),
                           "target": "B" if flip else "A"}
        items.append(item)
    random.Random(831950 + number).shuffle(items)
    return items


def register():
    paths = [HERE / f for f in ("listener.py", "options.json", "protocol.md", "review.html")]
    paths += [PREV / "scenes.json", PREV / "observed/probes.json"]
    return {"version": 1, "files": {os.path.relpath(p, HERE).replace("\\","/"):
                hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths},
            "forms": {str(n): digest(form(n)) for n in range(1, 13)}}


def validate_material():
    used = []
    for n in range(1, 13):
        items = form(n)
        assert len({x["scene"] for x in items}) == 6
        assert len({(x["role"], x["arm"]) for x in items}) == 6
        assert all(x["say"].strip() for x in items)
        for phase in ("action","motive"):
            assert sum(x[phase]["target"]=="A" for x in items)==3
            assert all(len(set(x[phase]["choices"].values()))==2 for x in items)
        used += [x["source"] for x in items]
    assert len(used)==len(set(used))==72 and set(used)=={k for k in SAMPLES if k.split(".")[1] in ("G","F","S")}
    assert not any(x["role"]=="tang" and x["scene"] in ("platform","names") for x in form(1))
    assert all(sum(next(x for x in form(n) if x["scene"]==q["id"])[phase]["target"]=="A"
                       for n in range(1,13))==6 for q in SCENES for phase in ("action","motive"))


def path_for(root, number):
    return root / f"form-{number:02}.json"


def load(root, number):
    path = path_for(root, number)
    d = json.loads(path.read_bytes()) if path.exists() else dict(form=number, fingerprint=digest(form(number)), answers=[])
    if d["form"]!=number or d["fingerprint"]!=digest(form(number)):
        raise ValueError("材料版本与原答卷不一致")
    return d


def view(number, state):
    answers = state["answers"]
    n = len(answers)
    if n==18:
        return dict(done=True, receipt=state["receipt"])
    phase = PHASES[n//6]
    item = form(number)[n%6]
    public = dict(actor=NAMES[item["role"]], context=item["public"], partner=item["partner"], say=item["say"])
    if phase != "free":
        public["choices"] = item[phase]["choices"]
    return dict(done=False, step=n, phase=phase, index=n%6+1, item=public)


def clean(s, maxlen):
    if not isinstance(s,str) or len(s)>maxlen:
        raise ValueError("回答格式或长度不符合要求")
    return s.strip()


def submit(root, number, payload):
    with LOCK:
        d=load(root,number)
        if not isinstance(payload,dict) or type(payload.get("step")) is not int:
            raise ValueError("缺少题号")
        step=payload["step"]
        if len(d["answers"])!=step or not 0<=step<18:
            raise ValueError("这一题已经保存，请刷新后继续")
        phase=PHASES[step//6]; item=form(number)[step%6]
        choice=payload.get("choice")
        if phase=="free":
            if choice not in ("heard","unclear") or type(payload.get("familiar")) is not bool:
                raise ValueError("请选择自己的读法")
            note=clean(payload.get("note",""),350)
            if (choice=="heard") != bool(note):
                raise ValueError("听出内容时请用自己的话简短写下；听不出来可以直接选")
            answer=dict(choice=choice,note=note,familiar=payload["familiar"])
        else:
            if choice not in CHOICES:
                raise ValueError("请选择一个选项")
            quote=clean(payload.get("quote",""),600)
            if choice=="unclear" and quote:
                raise ValueError("弃判不需要引文")
            if choice!="unclear" and (not quote or quote not in item["say"]):
                raise ValueError("请从角色回复中原样选一小段作为依据")
            answer=dict(choice=choice,quote=quote)
        d["answers"].append(dict(step=step,phase=phase,position=step%6,**answer))
        if len(d["answers"])==18:
            d["receipt"]=secrets.token_hex(4);d["submitted_at"]=now()
        write(path_for(root,number),d)
        return view(number,d)


def report(root):
    """Treat abstaining, alternate explanation, other, and missing as distinct outcomes."""
    groups={group:{} for group in ("novel","novel_valid","familiar","partial")}
    complete=0;partial=0
    for n in range(1,13):
        if not path_for(root,n).exists():continue
        d=load(root,n);ds=d["answers"];finished=len(ds)==18 and bool(d.get("receipt"))
        complete+=int(finished);partial+=int(not finished)
        for i,item in enumerate(form(n)):
            fam=ds[i]["familiar"] if len(ds)>i else None
            gs=["partial"] if not finished else (["familiar"] if fam else ["novel"]+(["novel_valid"] if item["valid"] else []))
            for g in gs:
                key=item["role"]+"."+item["arm"]
                cell=groups[g].setdefault(key,{phase:Counter() for phase in PHASES})
                for j,phase in enumerate(PHASES):
                    pos=i+j*6
                    ans=ds[pos]["choice"] if len(ds)>pos else "missing"
                    if ans in ("A","B"):
                        ans="target" if item[phase]["target"]==ans else "foil"
                    cell[phase][ans]+=1
    return {"completed_forms":complete,"started_incomplete_forms":partial,"groups":groups}


def serve(root,number,port,ready=None):
    cookie=secrets.token_urlsafe(28)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def reply(self,status,body,kind="application/json; charset=utf-8"):
            if not isinstance(body,bytes):body=json.dumps(body,ensure_ascii=False).encode()
            self.send_response(status);self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff");self.end_headers();self.wfile.write(body)
        def local(self):return self.headers.get("Host")==f"127.0.0.1:{self.server.server_port}"
        def do_GET(self):
            if not self.local():return self.reply(403,{})
            if self.path=="/":return self.reply(200,(HERE/"review.html").read_bytes(),"text/html; charset=utf-8")
            if self.path=="/state":return self.reply(200,dict(**view(number,load(root,number)),nonce=cookie))
            return self.reply(404,{})
        def do_POST(self):
            if not self.local() or self.headers.get("X-Review-Key")!=cookie:return self.reply(403,{})
            if self.path!="/answer":return self.reply(404,{})
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=12000:raise ValueError("回答长度异常")
                payload=json.loads(self.rfile.read(length))
                v=submit(root,number,payload)
            except (ValueError,TypeError) as exc:return self.reply(400,dict(error=str(exc)))
            return self.reply(200,dict(**v,nonce=cookie))
    with ThreadingHTTPServer(("127.0.0.1",port),Handler) as server:
        info=dict(url=f"http://127.0.0.1:{server.server_port}/",pid=os.getpid(),form=number)
        if ready:write(ready,info)
        print(info["url"],flush=True)
        server.serve_forever()


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("register","check","serve","report"))
    ap.add_argument("--form",type=int,choices=range(1,13),default=1)
    ap.add_argument("--root",type=Path,default=PRIVATE);ap.add_argument("--port",type=int,default=0)
    args=ap.parse_args();validate_material();regfile=HERE/"registration.json"
    if args.command=="register" and not regfile.exists():write(regfile,register())
    if json.loads(regfile.read_bytes())!=register():raise SystemExit("Registered materials differ.")
    if args.command in ("register","check"):print("12 balanced forms, 72 original replies, exposure exclusions and fingerprints verified.")
    elif args.command=="report":print(json.dumps(report(args.root),ensure_ascii=False,indent=2))
    else:serve(args.root,args.form,args.port,args.root/f"server-{args.form:02}.json")
