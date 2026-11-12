"""Plot each assignment, including undecided forms. Requires Matplotlib and a CJK font."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch,Patch
HERE=Path(__file__).resolve().parent
families={f.name for f in fm.fontManager.ttflist}
font=next((n for n in ("Microsoft YaHei","Noto Sans CJK SC","Noto Sans CJK JP","SimHei","PingFang SC") if n in families),None)
if not font:raise SystemExit("Install a Chinese font, e.g. Noto Sans CJK SC.")
plt.rcParams["font.family"]=font
rows=json.loads((HERE/"observed/choices.json").read_bytes())
fig,axes=plt.subplots(1,2,figsize=(9.4,3.8))
palette={True:"#156660",False:"#dfb46b",None:"#d4d9dd"}
hist=["early","rise","fall","flat"]; arms=["R","S","V","O"]
for ax,a,name in zip(axes,("tang","yu"),("唐荔排班","俞川排班")):
    for y,h in enumerate(hist):
        for j,m in enumerate(arms):
            if h=="early" and m!="R":
                ax.text(j+.40,y,"—",ha="center",va="center",color="#999999",fontsize=11);continue
            for n in range(4):
                value=rows[f"{a}.{h}.{m}.relief.{n}"]["event"]["target"]
                ax.add_patch(FancyBboxPatch((j+n*.20,y-.24),.17,.48,
                    boxstyle="round,pad=0,rounding_size=.035",facecolor=palette[value],edgecolor="none"))
    ax.set_xlim(-.07,3.91);ax.set_ylim(3.62,-.67)
    ax.set_yticks(range(4),["仅五月失约","失约 → 履约","履约 → 失约","一直失约"])
    ax.set_xticks([j+.38 for j in range(4)],["原条","原条复采","更新簿","五月簿"])
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=4,labelsize=9)
    ax.set_title(name,fontsize=13,pad=12,loc="left")
    for spine in ax.spines.values():spine.set_visible(False)
fig.subplots_adjust(left=.12,right=.99,wspace=.43,top=.77,bottom=.20)
fig.legend(handles=[Patch(color=c,label=s) for c,s in zip(palette.values(),("签给搭档","签给常鸣","岗位未定"))],
           loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.45,.01))
fig.savefig(HERE/"assignments.png",dpi=160,facecolor="white")
