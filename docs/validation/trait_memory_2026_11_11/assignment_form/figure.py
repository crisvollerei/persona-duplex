"""Plot the four prespecified raw-history swap pairs per actor from the actual chosen names."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch
HERE=Path(__file__).resolve().parent
fonts={f.name for f in fm.fontManager.ttflist}
font=next((x for x in ("Microsoft YaHei","Noto Sans CJK SC","Noto Sans CJK JP","PingFang SC","SimHei") if x in fonts),None)
if not font:raise SystemExit("A Chinese font is required.")
plt.rcParams["font.family"]=font
rows=json.loads((HERE/"observed/choices.json").read_bytes())
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
for ax,a,name in zip(axes,("tang","yu"),("唐荔","俞川")):
    for y,f in enumerate(("T","B","N")):
        for j,(h,start) in enumerate((("rise",0),("rise",2),("fall",0),("fall",2))):
            pair=[rows[f"{a}.{h}.R.relief.{f}.{n}"]["event"]["target"] for n in (start,start+1)]
            if None in pair:tag,bg="票不齐","#e6e7e8"
            elif pair==[True,True]:tag,bg="均选搭档","#c2e1d5"
            elif pair==[False,False]:tag,bg="均选常鸣","#e2edcf"
            elif pair==[False,True]:tag,bg="均选第二","#f6d4ad"
            else:tag,bg="均选第一","#ecd5c4"
            ax.add_patch(FancyBboxPatch((j-.45,y-.31),.90,.62,boxstyle="round,pad=0,rounding_size=.065",fc=bg,ec="none"))
            ax.text(j,y,tag,ha="center",va="center",fontsize=9,color="#273941")
    ax.set_title(name+"排班",loc="left",pad=15,fontweight="bold")
    ax.set_xticks(range(4),["改善 ①","改善 ②","恶化 ①","恶化 ②"])
    ax.xaxis.tick_top();ax.set_yticks(range(3),["选签色 T","带色填名 B","直接填名 N"])
    ax.set_xlim(-.52,3.52);ax.set_ylim(2.55,-.55);ax.tick_params(length=0,labelsize=10)
    for sp in ax.spines.values():sp.set_visible(False)
fig.subplots_adjust(left=.105,right=.99,wspace=.37,bottom=.22,top=.74)
fig.text(.105,.08,"每格交换一次两名候选人的先后。①②表示两种颜色映射；无色组下为独立复采。",fontsize=10,color="#42545d")
fig.savefig(HERE/"swaps.png",dpi=150,facecolor="white")
