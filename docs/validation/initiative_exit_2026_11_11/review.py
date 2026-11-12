"""Build a self-contained, offline, method-blind reader form or summarize returned forms."""
import argparse
from collections import Counter
from pathlib import Path
import json
from analyze import joined
from study import HERE, digest, load

PROFILE="岚明快直接，在有空间时愿意给自己的看法；用户限定范围时收住。角色贴合不要求每轮追问。"
OPTIONS=("A","B","两条都合适，无偏好","两条都不合适","无法判断")


def pairs():
    rr=joined("short")
    keys=[(c["id"],"open",1) for c in load(HERE/"cases.json")]+[("thread","bounded",i) for i in (1,2)]
    result=[]
    for case,mode,rep in keys:
        pair=sorted((r for r in rr if (r["case"],r["mode"],r["rep"])==(case,mode,rep)),key=lambda r:r["blind_id"])
        assert len(pair)==2 and pair[0]["user"]==pair[1]["user"]
        result.append(dict(id=digest([case,mode,rep])[:12],user=pair[0]["user"],
                           answers=[dict(id=r["blind_id"],text=r["say"]) for r in pair]))
    return result


def build():
    data=pairs()
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>岚的接话 · 匿名阅读</title><style>body{max-width:48rem;margin:auto;padding:2rem 1rem;font:17px/1.65 system-ui;color:#222}
article{border-top:1px solid #bbb;margin:2rem 0}blockquote{margin:1rem 0;background:#f3f5f6;padding:.8rem;white-space:pre-wrap}
select,button{font:inherit;margin:.4rem;padding:.2rem}small{color:#53616a}label{display:block}</style>
<h1>你愿意收到哪条回复？</h1><p>六组，约五分钟。方法名隐藏，不需要猜。请按你对这个情境的实际判断填写，可以两条都不选。匿名文件仅在你点击时下载到本机，不会上传。无需填写姓名或私人经历。</p><main></main><button id="save">下载我的匿名评价</button>
<script>
const pairs=DATA, role=ROLE, opts=OPTIONS, seal=SEAL;
const main=document.querySelector('main'), fields=[];
function el(tag,text,parent){const e=document.createElement(tag);e.textContent=text;if(parent)parent.append(e);return e}
function select(parent,title,options){const l=el('label',title,parent), s=el('select','',l);for(const x of ['—请选择—',...options])el('option',x,s);return s}
function swap(){return (crypto.getRandomValues(new Uint8Array(1))[0]&1)===1}
for(const p of pairs){const ar=el('article','',main),order=swap()?[...p.answers].reverse():p.answers;
 el('h2','情境 '+(fields.length+1),ar);el('small','角色参考：'+role,ar);el('p',p.user,ar);
 const evidence=order.map((a,i)=>{el('blockquote',(i?'B：':'A：')+a.text,ar);
 return {id:a.id,fit:select(ar,'放在这里是否合适？',['1','2','3','4','5','不确定']),role:select(ar,'符合岚的角色吗？',['符合','不符合','不确定'])};});
 const pref=select(ar,'如果是你，想收到哪条？',opts);fields.push({id:p.id,order,evidence,pref});}
document.querySelector('#save').onclick=()=>{if([...document.querySelectorAll('select')].some(x=>!x.selectedIndex)){alert('还有未选项；不确定也可以。');return}
 const rows=fields.map(f=>({id:f.id,display:f.order.map(a=>a.id),scores:f.evidence.map(e=>({id:e.id,fit:e.fit.value,role:e.role.value})),preference:f.pref.value}));
 const blob=new Blob([JSON.stringify({schema:1,seal,rows},null,2)],{type:'application/json'});let a=el('a','');a.href=URL.createObjectURL(blob);a.download='lan-reader.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),10000);}
</script></html>"""
    # Only synthetic prompts and anonymized visible replies travel with the form.
    enc=lambda x:json.dumps(x,ensure_ascii=False).replace("<","\\u003c")
    page=page.replace("RO"+"LE",enc(PROFILE),1).replace("OPTIONS",enc(OPTIONS),1).replace("SEAL",enc(digest(data)),1).replace("DATA",enc(data),1)
    (HERE/"reader.html").write_text(page,encoding="utf-8")


def tally(paths):
    data=pairs(); lookup={p["id"]:{a["id"] for a in p["answers"]} for p in data}
    methods={r["blind_id"]:r["route"] for r in joined("short")}
    results=[];seen=set()
    for path in paths:
        obj=load(path)
        if digest(obj) in seen: raise ValueError("duplicate form")
        seen.add(digest(obj))
        if obj.get("schema")!=1 or obj["seal"]!=digest(data) or len(obj["rows"])!=len(data):raise ValueError("wrong form")
        if {r["id"] for r in obj["rows"]}!=set(lookup):raise ValueError("case mismatch")
        counts=Counter()
        for r in obj["rows"]:
            ids=lookup[r["id"]];order=r["display"]
            if len(order)!=2 or set(order)!=ids or r["preference"] not in OPTIONS:raise ValueError("invalid comparison")
            scores=r["scores"]
            if len(scores)!=2 or {s["id"] for s in scores}!=ids or any(s["fit"] not in ('1','2','3','4','5','不确定')
                  or s["role"] not in ('符合','不符合','不确定') for s in scores):raise ValueError("invalid scores")
            pref=r["preference"]; counts[methods[order["AB".index(pref)]] if pref in ("A","B") else pref]+=1
        results.append(dict(counts))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("forms",nargs="*",type=Path)
    forms=ap.parse_args().forms
    if forms:tally(forms)
    else:build()
