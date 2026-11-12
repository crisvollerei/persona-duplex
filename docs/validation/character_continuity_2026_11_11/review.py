"""Render a standalone, offline human ballot from materials and blinded SAY only."""
import argparse
import hashlib
import html
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
STUDY = "character_continuity_2026_11_11"
STYLES = """
*{box-sizing:border-box}html{color-scheme:light}body{margin:auto;max-width:74rem;background:#f4f6f5;color:#20343a;
font:16px/1.72 system-ui,"Microsoft YaHei",sans-serif;padding:1.5rem}h1,h2,h3,p{margin:0 0 .55rem}
h1{font-size:clamp(1.6rem,4vw,2.35rem)}h2{font-size:1.4rem}h3{font-size:1.08rem}
.box,.scene,footer{margin:0 auto 1rem;background:#fff;border:1px solid #d1ddd8;border-radius:.75rem;padding:clamp(1rem,2.5vw,1.7rem)}
.ref{border-inline-start:4px solid #377366;background:#eef6f2;padding:.65rem .9rem;white-space:pre-wrap;margin:.65rem 0}
.reading{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1rem 0}.version{min-width:0;padding:1rem;border-radius:.45rem;background:#f5f7f6}
.version ol{list-style:none;padding:0;margin:0}.version li+li{border-top:1px solid #d7e0dc;margin-top:.55rem;padding-top:.55rem}
.turn{font-size:.82rem;color:#4d675f;font-weight:650}.utter{margin:.2rem 0 .45rem;white-space:pre-wrap;overflow-wrap:anywhere}
.utter b{color:#38554d}.none,.hint{color:#50645d}.none{font-style:italic}
fieldset{border:0;margin:.9rem 0;padding:0}legend{font-weight:650}.opts{display:flex;flex-wrap:wrap;gap:.35rem}
.opts label{display:flex;align-items:center;gap:.25rem;border:1px solid #bdcfc8;border-radius:.4rem;padding:.35rem .65rem;cursor:pointer}
input{accent-color:#236451}.opts label:has(:checked){background:#dcefe6;border-color:#39745f}
textarea{width:100%;padding:.65rem;border:1px solid #abbdb4;border-radius:.4rem;font:inherit}
button{font:inherit;border:1px solid #355e52;border-radius:.4rem;padding:.4rem .85rem;margin:.2rem .25rem .2rem 0;color:#264c3f;background:#ecf4ef;cursor:pointer}
button:focus-visible,a:focus-visible,textarea:focus-visible{outline:3px solid #206a55;outline-offset:2px}
nav{display:flex;flex-wrap:wrap;gap:1rem}a{color:#205b49}.ref small{display:block}.footnote{font-size:.88rem;color:#50645d}
@media(max-width:760px){body{padding:.65rem}.reading{grid-template-columns:1fr}.box,.scene,footer{padding:.85rem}.version{padding:.7rem}}
"""
SCRIPT = """
const byId=id=>document.getElementById(id), meta=JSON.parse(byId('meta').textContent);
function ballot(){
  const form=byId('ballot');if(!form.reportValidity())return null;
  const answers={};for(const id of meta.ids){
    const group=byId('case-'+id);
    answers[id]={fit:group.querySelector('input[name="fit-'+id+'"]:checked').value,
      continue_chat:group.querySelector('input[name="chat-'+id+'"]:checked').value,
      evidence:group.querySelector('textarea').value.trim()};
  }
  return JSON.stringify({study:meta.study,source:'human_self_report',packet_sha256:meta.hash,ratings:answers},null,2);
}
function expose(value){
  const area=byId('result');area.value=value;byId('fallback').hidden=false;area.focus();area.select();
  byId('status').textContent='答卷文本已选中，可复制并粘贴到聊天。';
}
byId('select').onclick=()=>{const v=ballot();if(v)expose(v)};
byId('copy').onclick=async()=>{const v=ballot();if(!v)return;try{
  await navigator.clipboard.writeText(v);byId('status').textContent='已复制。请粘贴到聊天，网页不会自动提交。';
}catch(e){expose(v)}};
byId('download').onclick=()=>{const v=ballot();if(!v)return;
  const url=URL.createObjectURL(new Blob([v],{type:'application/json;charset=utf-8'}));
  const link=document.createElement('a');link.href=url;link.download='character-continuity-ballot.json';
  document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1500);
  byId('status').textContent='已生成答卷文件。请保留或自行交回应答，网页不会自动提交。';
};
byId('ballot').addEventListener('input',()=>{byId('fallback').hidden=true;byId('status').textContent=''});
"""


def escape(value):
    return html.escape(value, quote=True)


def choice(ident, prefix, title, labels):
    options = zip(("A", "B", "both", "neither", "uncertain"), labels, strict=True)
    buttons = "".join(f'<label><input type="radio" required name="{prefix}-{ident}" value="{value}">{label}</label>'
                      for value, label in options)
    return f'<fieldset><legend>{title}</legend><div class="opts">{buttons}</div></fieldset>'


def version(label, users, replies):
    turns = []
    for n, (u, r) in enumerate(zip(users, replies, strict=True), 1):
        reply = escape(r) if r.strip() else '<span class="none">（没有可见回复）</span>'
        turns.append(f'<li><div class="turn">第 {n} 轮</div><p class="utter"><b>你：</b>{escape(u)}</p>'
                     f'<p class="utter"><b>对方：</b>{reply}</p></li>')
    return f'<section class="version"><h3>对话 {label}</h3><ol>{"".join(turns)}</ol></section>'


def render(materials_path, blind_path):
    materials = json.loads(materials_path.read_bytes())
    packet = blind_path.read_bytes()
    items = json.loads(packet)["scenes"]
    mrows = materials["scenes"]
    m = {s["id"]: s for s in mrows}
    ids = [s["id"] for s in items]
    if (not ids or len(set(ids)) != len(ids) or len(m) != len(mrows) or set(ids) != set(m)
            or any(not isinstance(i, str) or not i.isascii() or not i.isalnum() for i in ids)):
        raise ValueError("Scene IDs must match and be distinct ASCII letters/digits.")
    ref = escape(materials["role_reference"])
    if not ref.strip():
        raise ValueError("Missing role reference.")
    role = f'<aside class="ref"><strong>角色参考：守岸人</strong><div>{ref}</div></aside>'
    sections = []
    for scene in items:
        i = scene["id"]
        users, lines = m[i]["turns"], scene["lines"]
        if (set(lines) != {"A", "B"} or not all(isinstance(row, list) and len(row) == 4
                and all(isinstance(x, str) for x in row) for row in (users, lines["A"], lines["B"]))):
            raise ValueError(f"Require four actual turns per side in {i}.")
        parts = "".join(version(lab, users, lines[lab]) for lab in ("A", "B"))
        fit = choice(i, "fit", "① 连续读下来，哪段更让你觉得是这位角色在回应眼前这个人？",
                     ("A 更贴合", "B 更贴合", "两段都符合", "两段都不符合", "无法判断"))
        chat = choice(i, "chat", "② 如果你就是对话里的人，你更愿意和哪一段里的她接着聊？",
                      ("A", "B", "都愿意", "都不愿意", "无法判断"))
        sections.append(f'<article class="scene" id="case-{i}"><h2>{i} · {escape(m[i]["title"])}</h2>'
                        f'<div class="reading">{parts}</div>{role}{fit}{chat}'
                        f'<label for="note-{i}">哪一轮、哪一句让你这样选？有没有哪句换成普通助手也一样？'
                        f' <span class="hint">（可不填）</span></label>'
                        f'<textarea id="note-{i}" rows="2" placeholder="例如：B 第 3 轮，……"></textarea></article>')
    nav = "".join(f'<a href="#case-{i}">{i} · {escape(m[i]["title"])}</a>' for i in ids)
    meta = json.dumps(dict(study=STUDY, hash=hashlib.sha256(packet).hexdigest(), ids=ids),
                      ensure_ascii=True).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>守岸人 · 连续对话盲评</title><style>{STYLES}</style></head><body>
<header class="box"><h1>守岸人 · 连续对话盲评</h1>{role}
<p>每个场景有 A、B 两段完整对话。用户说的话相同；各自的回复一路延续。请整段读完再选，下一个场景的 A、B 顺序不一定与上一组相同。</p>
<p>只看她是否在听、有没有自己的看法、遇到新信息如何回应。正式或寡言都可以贴合；不要求像真人、一定反驳或说出私事。
两段都可能是她，也可能都不像。你愿意接着聊的那段，不一定是你觉得更像她的那段。</p>
<nav aria-label="场景目录">{nav}</nav></header><main><form id="ballot" onsubmit="return false">
{''.join(sections)}<footer><h2>带走答卷</h2><p>本页不联网、不自动提交，也不保存到浏览器。请在关闭前复制答卷回到聊天，或下载后自行交回。</p>
<noscript>本页需启用 JavaScript 才能导出；也可直接按场景号把两项选择和意见写回聊天。</noscript>
<button type="button" id="copy">复制答卷到聊天</button><button type="button" id="download">下载 JSON</button>
<button type="button" id="select">显示并全选文本</button><p id="status" role="status" aria-live="polite"></p>
<div id="fallback" hidden><label for="result">答卷原文（可 Ctrl/Cmd+C 或长按复制）</label>
<textarea id="result" rows="8" readonly></textarea></div></footer></form></main>
<script id="meta" type="application/json">{meta}</script><script>{SCRIPT}</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--materials", type=Path, default=HERE / "materials.json")
    ap.add_argument("--blind", type=Path, default=HERE / "observed/blind.json")
    ap.add_argument("--out", type=Path, default=HERE / "review.html")
    args = ap.parse_args()
    document = render(args.materials, args.blind)
    args.out.write_bytes((document + "\n").encode("utf-8"))
    print(f"Created {args.out} (offline blind review)")


if __name__ == "__main__":
    main()
