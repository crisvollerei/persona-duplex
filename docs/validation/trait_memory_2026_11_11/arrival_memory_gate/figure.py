"""Plot actual matched Z outcomes. Gray means an incomplete pair, never a reserve choice."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Patch
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
fonts={f.name for f in font_manager.fontManager.ttflist}
font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","Noto Sans CJK JP","SimHei","PingFang SC") if f in fonts),None)
if not font:raise SystemExit("A Chinese font is required.")
plt.rcParams["font.family"]=font
rows=json.loads((HERE/"observed/choices.json").read_bytes())
fig,axes=plt.subplots(1,2,figsize=(9.7,3.35))
colors={"均选搭档":"#357971","均选常鸣":"#bdcda8","人选不同":"#d99760","票不齐":"#d6d9dd"}
for ax,a,name in zip(axes,("tang","yu"),("唐荔","俞川")):
    for i,(j,n) in enumerate((j,n) for j in ("envelope","boards") for n in range(4)):
        for y,s in enumerate(("future","present","unknown")):
            p=[rows[f"{a}.{j}.{s}.{h}.Z.{n}"]["event"]["target"] for h in ("rise","fall")]
            label="票不齐" if None in p else "人选不同" if p[0]!=p[1] else ("均选搭档" if p[0] else "均选常鸣")
            x=i+(.30 if i>=4 else 0)
            ax.add_patch(FancyBboxPatch((x-.42,y-.25),.84,.5,boxstyle="round,pad=0,rounding_size=.06",ec="none",fc=colors[label]))
    ax.set_title(name+"排班",loc="left",pad=11)
    ax.set_yticks(range(3),["还要赶来","已经在场","状态未知"])
    ax.set_xticks([1.5,5.8],["封套清点","画板归还"]);ax.xaxis.tick_top()
    ax.set_xlim(-.5,7.8);ax.set_ylim(2.48,-.5);ax.tick_params(length=0,labelsize=10)
    for spine in ax.spines.values():spine.set_visible(False)
fig.subplots_adjust(left=.12,right=.99,wspace=.3,bottom=.23,top=.74)
fig.legend(handles=[Patch(fc=v,label=k) for k,v in colors.items()],ncol=4,frameon=False,loc="lower center")
fig.savefig(HERE/"same_input.png",dpi=160,facecolor="white")
