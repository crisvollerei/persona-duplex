"""Six previously unseen situations. Dots are scene means, not confidence intervals."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

HERE=Path(__file__).resolve().parent
def render(src,dest):
    data=json.loads(src.read_bytes())["tables"];scenes=json.loads((HERE/"scenes.json").read_bytes())
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","Source Han Sans SC","PingFang SC","SimHei") if f in fonts),None)
    if not font:raise SystemExit("A Chinese font is required (e.g., Noto Sans CJK SC).")
    plt.rcParams.update({"font.family":font,"axes.unicode_minus":False,"font.size":10})
    fig,axs=plt.subplots(1,2,figsize=(9.2,4.8),sharey=True)
    fig.subplots_adjust(left=.16,right=.96,top=.73,bottom=.22,wspace=.18)
    fig.text(.06,.925,"角色多说的那一句，出现在哪些场景？",fontsize=18,weight="bold",color="#223d44")
    fig.text(.06,.855,"同一场景：成长线的披露等级 − 两组冻结回复的平均等级",color="#425c63")
    for ax,(a,name) in zip(axs,(("tang","唐荔"),("yu","俞川"))):
        ax.set_title(name,loc="left",color="#234951",weight="bold",fontsize=12)
        ax.axvline(0,lw=1,color="#76868a");ax.grid(axis="x",lw=.6,color="#d8dfe0",ls=":")
        for who,color,marker,off,label in (("reader","#096c74","o",-.11,"研究侧盲读"),("model","#9b5429","D",.11,"模型盲读")):
            vals=[data[who][a]["by_scene"][s["id"]]["vs_two_frozen"] for s in scenes]
            ax.scatter(vals,[i+off for i in range(len(vals))],color=color,s=37,marker=marker,label=label,zorder=3)
        ax.set_xlim(-1.15,1.65);ax.set_xticks([-1,-.5,0,.5,1,1.5]);ax.set_yticks(range(6),[s["name"] for s in scenes])
        ax.set_ylim(5.55,-.55);ax.tick_params(length=0,pad=5)
        for sp in ax.spines.values():sp.set_visible(False)
    axs[0].legend(loc="upper left",bbox_to_anchor=(-.03,-.17),ncol=2,frameon=False)
    fig.text(.06,.025,"每个点比较 2 条成长与 4 条冻结回复。右侧表示成长线说得更多；0 表示同分，不表示两条回答相同。",fontsize=9,color="#425c63")
    fig.savefig(dest,dpi=165,facecolor="white");plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,default=HERE/"observed/results.json")
    p.add_argument("--out",type=Path,default=HERE/"scenes.png");args=p.parse_args();render(args.source,args.out)
