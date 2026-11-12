"""Build a self-contained, answer-free form from the dialogue corpus. Standard library only."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>对话中的到岗状态｜独立盲评</title>
<style>
:root{color-scheme:light;--ink:#202f31;--muted:#506461;--green:#195e52;--line:#d5dfd9;--pale:#eff5f1}*{box-sizing:border-box}body{margin:0;background:#f6f6f0;color:var(--ink);font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:56rem;margin:0 auto;padding:2rem 1rem 5rem}h1{font-size:clamp(1.65rem,4vw,2.4rem);line-height:1.25;margin:.25rem 0 1rem}h2{margin:.15rem 0;font-size:1.2rem}p{margin:.6rem 0}.eyebrow{color:var(--green);font-weight:700;font-size:.85rem}.guide{background:#ecf2ed;padding:1rem;border-left:4px solid var(--green);border-radius:5px}.meta{margin:1rem 0}.meta input{max-width:14rem}.privacy,.hint{color:var(--muted);font-size:.92rem}.counter{margin:1.2rem 0 .4rem;font-weight:600}.case{background:#fff;border:1px solid var(--line);border-radius:12px;padding:1.1rem;margin:1rem 0}.case.skip{background:#f5f8f5}.turns{padding:.65rem 1rem;background:#f5f8f6;border-left:2px solid #bccdc2;margin:.8rem 0}.turns p{margin:.3rem 0}.people{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}fieldset{border:1px solid var(--line);margin:0;padding:.5rem .8rem .85rem;min-width:0;border-radius:7px}legend{font-weight:700}fieldset label{display:block;margin:.25rem 0}.line-label{display:block;font-size:.88rem;color:var(--muted)}select,input[type=text]{font:inherit;min-height:2.55rem;padding:.3rem .5rem;border-radius:4px;border:1px solid #81968e;background:#fff;color:var(--ink);max-width:100%}select{width:100%}input[type=checkbox]{width:1.05rem;height:1.05rem;vertical-align:middle}.skip-label{display:inline-block;margin:.7rem 0 0}.choices{display:flex;align-items:center;flex-wrap:wrap;gap:.75rem}.choices button{font:inherit;font-weight:bold;min-height:2.65rem;border:0;border-radius:7px;background:var(--green);color:white;padding:.5rem 1rem;cursor:pointer}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #ba6f34;outline-offset:2px}#msg{color:#7e410a}.invalid{border-color:#a74e12}.footer{border-top:1px solid var(--line);padding-top:1rem}noscript{display:block;color:#822} @media(max-width:550px){main{padding-top:1.15rem}.people{grid-template-columns:1fr}.case{padding:.9rem}}
</style></head>
<body><main><header><div class="eyebrow">独立盲评 · 社区放映室</div><h1>最后一句说完，两人在哪里？</h1>
<p>每段对话彼此独立。目标地点始终是<strong>社区放映室内的检票桌</strong>。请分别标注沈遥、许原，不借用其他题的信息，也不和另一位评审商量。</p></header>
<div class="guide"><p><b>状态</b>：已在场＝话里的意思是此人已在检票桌；还需到岗＝此人尚需过去，不要求已经上路；未知＝从对话无法判断。</p>
<p><b>判断依据</b>：直接定位＝话语直接给出此人的当前位置或行动；语境推知＝位置没有直接说出来，是你结合上下文作的推断；无可定位线索＝没有可用来定位此人的信息。</p>
<p>状态和依据分开选。如果你认真判断后觉得位置说不清，就选“未知”；不愿作答可跳过整题。</p></div>
<div class="meta"><label>匿名评审编号（选填） <input type="text" id="reviewer" placeholder="不填姓名或联系方式" maxlength="48" autocomplete="off"></label></div>
<p class="privacy">仅在本机填写，不会联网或自动保存。请在关闭前下载答案；另一位评审另开一份空白页面。</p>
<p class="counter" id="progress" aria-live="polite"></p><noscript>此表单需要在浏览器中启用 JavaScript。</noscript>
<div id="cases"></div><div class="footer"><div class="choices"><button type="button" id="download">下载我的标注（JSON）</button><span id="msg" role="status" aria-live="polite"></span></div></div>
</main><script type="application/json" id="corpus">__CORPUS__</script>
<script>
"use strict";
const corpus=JSON.parse(document.getElementById("corpus").textContent);
const persons=["沈遥","许原"];
const options={state:[["present","已在场"],["future","还需到岗"],["unknown","未知"]],basis:[["explicit","直接定位"],["pragmatic","语境推知"],["none","无可定位线索"]]};
const cards=[];
function make(tag,txt="",cls=""){const el=document.createElement(tag);if(txt)el.textContent=txt;if(cls)el.className=cls;return el;}
function choice(type){const s=make("select");s.add(new Option("请选择",""));for(const [v,t] of options[type])s.add(new Option(t,v));return s;}
for(const [i,item] of corpus.items.entries()){
  const box=make("section","","case"), head=make("h2","对话 "+String(i+1).padStart(2,"0")), dialog=make("div","","turns"), peopleBox=make("div","","people");
  for(const turn of item.turns){const line=make("p");line.append(make("strong",turn.speaker+"："),document.createTextNode(turn.text));dialog.append(line);}
  const controls={};
  for(const p of persons){const f=make("fieldset");f.append(make("legend",p));controls[p]={};
    for(const t of ["state","basis"]){const label=make("label");label.append(make("span",t==="state"?"状态":"判断依据","line-label"));
      const sel=choice(t);sel.addEventListener("change",()=>{box.classList.remove("invalid");refresh();});label.append(sel);f.append(label);controls[p][t]=sel;}
    peopleBox.append(f);}
  const skip=make("input");skip.type="checkbox";const skipLabel=make("label","","skip-label");skipLabel.append(skip,document.createTextNode(" 跳过整题"));
  skip.addEventListener("change",()=>{for(const f of peopleBox.children)f.disabled=skip.checked;box.classList.toggle("skip",skip.checked);box.classList.remove("invalid");refresh();});
  box.append(head,dialog,peopleBox,skipLabel);document.getElementById("cases").append(box);cards.push({id:item.id,box,controls,skip});
}
function read(card){if(card.skip.checked)return null;
  const result={};for(const p of persons){const state=card.controls[p].state.value,basis=card.controls[p].basis.value;if(!state||!basis)return undefined;result[p]={state,basis};}return result;}
function refresh(){const n=cards.filter(c=>read(c)!==undefined).length;const s=cards.filter(c=>c.skip.checked).length;
  document.getElementById("progress").textContent="已处理 "+n+" / "+cards.length+" 段（跳过 "+s+" 段）";document.getElementById("msg").textContent="";}
function compile(){const labels={};for(const c of cards){const x=read(c);if(x===undefined)return {missing:c};labels[c.id]=x;}
  return {answer:{reviewer:document.getElementById("reviewer").value.trim(),source:"human_self_report",corpus_sha256:corpus.sha256,labels}};}
document.getElementById("download").addEventListener("click",()=>{const v=compile(),msg=document.getElementById("msg");
  if(v.missing){v.missing.box.classList.add("invalid");msg.textContent="还有未完成的对话，请补齐或勾选跳过。";for(const s of v.missing.box.querySelectorAll("select"))if(!s.value){s.focus();break;}return;}
  const blob=new Blob([JSON.stringify(v.answer,null,2)+"\n"],{type:"application/json;charset=utf-8"}),url=URL.createObjectURL(blob);
  const link=make("a");link.href=url;link.download="arrival_review_"+(v.answer.reviewer.replace(/[^a-zA-Z0-9_-]/g,"").slice(0,24)||"anonymous")+".json";
  document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1500);msg.textContent="下载已发起，请保留导出的 JSON 文件。";});
refresh();
</script></body></html>
"""

def render(source: bytes) -> str:
    raw = json.loads(source)
    if not isinstance(raw, list) or not raw:
        raise ValueError("The corpus must be a nonempty list")
    ids = set()
    public = []
    for item in raw:
        number = item["id"]
        if not isinstance(number, str) or not number.isascii() or not number.isdigit() or number in ids:
            raise ValueError("Case IDs must be unique numeric strings")
        ids.add(number)
        turns = [{"speaker": turn["speaker"], "text": turn["text"]} for turn in item["turns"]]
        if not turns or not all(isinstance(v,str) and v.strip() for t in turns for v in t.values()):
            raise ValueError("Empty turn or speaker")
        public.append({"id":number,"turns":turns})
    digest = hashlib.sha256(source).hexdigest()
    # Same blind order across respondents. Do not embed pair metadata or external references.
    public.sort(key=lambda item: hashlib.sha256(("arrival-review-order/v1/"+item["id"]).encode()).digest())
    data = json.dumps({"sha256":digest,"items":public}, ensure_ascii=False, separators=(",",":"))
    data = data.replace("&", "\\u0026").replace("<","\\u003c").replace(">","\\u003e")
    return PAGE.replace("__CORPUS__", data)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=HERE/"cases.json")
    ap.add_argument("--out",type=Path,default=HERE/"review.html")
    args=ap.parse_args()
    args.out.write_bytes(render(args.source.read_bytes()).encode("utf-8"))
