"""Build a standalone reader for paired one-step answers with a shared archive history."""
import argparse
from html import escape
from itertools import product
import json
from pathlib import Path

from statecard import render_card


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent
CONDITIONS = {"control": "无状态卡", "state_card": "附当前状态卡"}


def load(path):
    return json.loads(path.read_bytes())


def utter(row):
    if row["status"] == "delivered":
        if not row["say"].strip():
            raise ValueError("Empty delivered SAY")
        return escape(row["say"])
    if row["say"]:
        raise ValueError("Undelivered output has non-empty SAY")
    descriptions = {"empty": "无可见回复", "error": "请求失败，无可见回复",
                    "upstream_missing": "前文缺失，无可见回复"}
    return f'<span class="missing">（{descriptions[row["status"]]}）</span>'


def render():
    material = load(PREVIOUS / "scenes.json")
    variants = {v["id"]: v for v in material["variants"]}
    question = material["question"]
    rows = load(HERE / "observed/turns.json")
    index = {(r["variant"],r["history"],r["act"],r["sample"],r["condition"]):r for r in rows}
    keys = list(product(("V1", "V2", "V3"), "AB", (2, 3), (1, 2)))
    if len(rows) != 48 or set(index) != {(*k,c) for k in keys for c in CONDITIONS}:
        raise ValueError("Expected every one of the 48 assigned current answer slots, including errors")
    old = {(r["variant"],r["replicate"],r["act"]):r for r in load(PREVIOUS/"observed/turns.json")
           if r["condition"] == "scoped"}
    cases = []
    for variant, history, act, sample in keys:
        prior = []
        for before in range(1, act):
            row = old[variant, history, before]
            if row["user"] != question:
                raise ValueError("Changed previously archived question")
            prior.append(f'<li><p><b>漂泊者：</b><span>{escape(row["user"])}</span></p>'
                         f'<p><b>守岸人：</b><span>{utter(row)}</span></p></li>')
        event = variants[variant]["events"][act-1]
        pair = []
        for condition, caption in CONDITIONS.items():
            row = index[variant,history,act,sample,condition]
            if row["parent_id"] != f"{variant}.scoped.{history}.{act}":
                raise ValueError("Displayed history does not match archived request")
            pair.append(f'<section><h3>{caption}</h3><blockquote data-answer="{escape(row["id"])}">'
                        f'{utter(row)}</blockquote></section>')
        cid = f"{variant}-{history}-{act}-{sample}"
        cases.append(f'<article id="{cid}"{"" if cid=="V1-A-2-1" else " hidden"}>'
                     f'<h2>固定前史</h2><ol>{"".join(prior)}</ol>'
                     f'<h2>此刻发生的事</h2><p>{escape(event["shorekeeper"])}</p><p>{escape(event["traveler"])}</p>'
                     f'<details><summary>展开：有卡条件多出的三行状态</summary><pre>{escape(render_card(act))}</pre></details>'
                     f'<h2>漂泊者：{escape(question)}</h2><div class="pair">{"".join(pair)}</div></article>')
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>旧话还在，此刻呢？｜守岸人状态卡对照</title>
<style>*{box-sizing:border-box}body{max-width:59rem;margin:2rem auto;padding:0 1rem;background:#f4f7f7;color:#182f3b;font:17px/1.65 system-ui,sans-serif}
h1{line-height:1.15;font-size:clamp(1.7rem,5vw,2.8rem)}small{color:#476670}label{display:inline-flex;flex-direction:column;gap:.15rem}select{font:inherit;border:1px solid #8da7a6;border-radius:.3rem;padding:.2rem .45rem;background:white}
.filters,.pair{display:flex;flex-wrap:wrap;gap:.7rem}.filters{margin:1.2rem 0}article{background:white;border:1px solid #c8d8d6;padding:.7rem 1.25rem 1.25rem;border-radius:.8rem}h2{font-size:1.12em;margin:1rem 0 .35rem}ol{padding-left:1.4rem}li{border-left:3px solid #b7c8c4;padding-left:.7rem}li p{margin:.25rem 0;white-space:pre-wrap}
.pair>section{flex:1;min-width:0;background:#edf4f2;border-radius:.45rem;padding:.65rem .85rem}.pair h3{font-size:1rem;margin:0;color:#365b56}
blockquote{white-space:pre-wrap;margin:.5rem 0}pre{white-space:pre-wrap}details{color:#436258}summary{cursor:pointer}.missing{color:#794e49}
@media(max-width:650px){body{margin:1.2rem auto}.pair{flex-direction:column}article{padding:.5rem .8rem .8rem}}
</style><main><small>黑海岸 · 原创故事 · 实际回复回放</small><h1>旧话还在，此刻呢？</h1>
<p>两条回复读到相同的剧情资料、相同的旧对话。区别是有没有一份单列的当前状态。第三幕的历史来自上一轮真实对话，不接入本轮任何一条新生成的第二幕回复。</p>
<div class="filters"><label>支线版本<select id="v"><option>V1</option><option>V2</option><option>V3</option></select></label>
<label>旧历史<select id="h"><option>A</option><option>B</option></select></label>
<label>当前幕<select id="a"><option value="2">第二幕</option><option value="3">第三幕</option></select></label>
<label>本轮采样<select id="s"><option>1</option><option>2</option></select></label></div>
<noscript>当前仅显示默认组合，启用 JavaScript 可切换。全部回复也见 observed/turns.json。</noscript>
""" + "\n".join(cases) + """
<p><small>从存档逐字回放，不联网，不重新生成。第三幕无需复述姓名才算理解双方已经知情。</small></p></main>
<script>const controls=['v','h','a','s'].map(id=>document.getElementById(id));
function show(){const key=controls.map(x=>x.value).join('-');document.querySelectorAll('article').forEach(a=>a.hidden=a.id!==key)}
controls.forEach(x=>x.addEventListener('change',show));show();</script></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    page = render().encode()
    dest = HERE / "read.html"
    if args.check:
        if dest.read_bytes() != page:
            raise ValueError("Reader differs from the archived answer pairs")
    else:
        dest.write_bytes(page)
