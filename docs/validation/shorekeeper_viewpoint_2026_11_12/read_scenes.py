"""Make a standalone replay of actual delivered SAY, without exposing later acts by default."""
import argparse
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def render():
    scenes = json.loads((HERE/"scenes.json").read_bytes())
    turns = json.loads((HERE/"observed/turns.json").read_bytes())
    esc = html.escape
    blocks = []
    for v in scenes["variants"]:
        for rep in ("A", "B"):
            for n, event in enumerate(v["events"], 1):
                replies = []
                for method, label in (("scoped", "只给当幕视角"), ("full_notes", "另附作者全稿")):
                    row, = [r for r in turns if (r["variant"],r["replicate"],r["act"],r["condition"]) ==
                            (v["id"], rep, n, method)]
                    replies.append(f'<section><small>{label}</small><blockquote>{esc(row["say"]) if row["say"] else "（无可见回复）"}</blockquote></section>')
                blocks.append(f'<article data-v="{esc(v["id"])}" data-r="{rep}" data-n="{n}" hidden>'
                    f'<h2>第{"一二三"[n-1]}幕</h2><p>{esc(event["shorekeeper"])}</p><p>{esc(event["traveler"])}</p>'
                    f'<h3>漂泊者：{esc(scenes["question"])}</h3><div>{"".join(replies)}</div></article>')
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>谁留下了这段回音？｜三幕回放</title><style>
*{box-sizing:border-box}body{margin:3em auto;padding:0 1em;max-width:59rem;color:#183041;background:#f6f8f7;font:17px/1.73 system-ui,sans-serif}
h1{font-size:2em;line-height:1.3}h2{font-size:1.5em}small{color:#43616d}select,button{font:inherit;padding:.2em .5em}label,button{margin:.2em .6em .2em 0}section,article{padding:.45em 1em}article{background:white;border:1px solid #c6d6d7;border-radius:10px;margin:1em 0}article>div{display:grid;grid-template-columns:1fr 1fr;gap:.4em}
section{background:#f2f6f5;border-left:3px solid #537e86;margin:.6em 0}blockquote{margin:.45em 0;white-space:pre-wrap}button:disabled{opacity:.48}@media(max-width:660px){article>div{grid-template-columns:1fr}body{margin:1.2em auto}}
</style><main><small>守岸人 · 原创支线 · 实际生成回放</small><h1>谁留下了这段回音？</h1>
<p>同一句问话，两条独立的对话。在每一幕先读守岸人亲历的事，再读她实际说的话。切换复采可看同一设定是否得到同样的回答。</p>
<label>支线版本 <select id="v"><option>V1</option><option>V2</option><option>V3</option></select></label>
<label>采样 <select id="r"><option>A</option><option>B</option></select></label>
<button type="button" id="prev">上一幕</button><button type="button" id="next">下一幕</button>
<noscript>需要 JavaScript 翻页；完整文本也在同目录 observed/turns.json 中。</noscript>
""" + "\n".join(blocks) + """
<p><small>此页离线读取已归档回复，不调用模型，不是重新生成。每条路线承接自己的前文；展示文本未润色。</small></p></main>
<script>
const $=id=>document.getElementById(id);let n=1;
function show(){for(const a of document.querySelectorAll('article'))a.hidden=!(a.dataset.v===$('v').value&&a.dataset.r===$('r').value&&Number(a.dataset.n)===n);$('prev').disabled=n===1;$('next').disabled=n===3}
$('prev').onclick=()=>{if(n>1)n--;show()};$('next').onclick=()=>{if(n<3)n++;show()};
for(const x of ['r','v'])$(x).onchange=()=>{n=1;show()};show();
</script></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data = render().encode()
    dest = HERE/"scenes.html"
    if args.check:
        if dest.read_bytes() != data: raise ValueError("changed scene reader")
    else:
        dest.write_bytes(data)
