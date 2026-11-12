"""Render the publication figure from observed scores (matplotlib, a Chinese font)."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
def render(src,dst):
    data=json.loads(src.read_bytes())
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","Source Han Sans SC","SimHei","PingFang SC")
               if f in available),None)
    if not font:raise SystemExit("Install a Chinese plot font (e.g. Noto Sans CJK SC).")
    plt.rcParams.update({"font.family":font,"axes.unicode_minus":False,"font.size":10,"axes.spines.right":False,
                         "axes.spines.top":False,"axes.labelcolor":"#263d45","xtick.color":"#43585e","ytick.color":"#43585e"})
    fig,ax=plt.subplots(2,2,figsize=(10.6,7.1))
    fig.subplots_adjust(left=.10,right=.97,bottom=.13,top=.84,hspace=.57,wspace=.27)
    fig.suptitle("同一段经历中，参数与表达怎样变化",x=.1,y=.972,ha="left",fontsize=19,fontweight="bold",color="#213d45")
    fig.text(.1,.907,"24 幕连续剧情  /  0、12、24 幕隔离测量  /  成长与冻结共享同一份经历",color="#48646b")
    colors=["#126b70","#83502b"]
    allvalues=[v for r in data["roles"].values() for v in r["values"]]
    limits=(min(allvalues)-.014,max(allvalues)+.022)
    for j,(a,name) in enumerate((("tang","唐荔"),("yu","俞川"))):
        c=colors[j];top,bot=ax[0,j],ax[1,j];vs=data["roles"][a]["values"]
        top.set_title(f"{name}｜自我披露参数",loc="left",fontweight="bold",color=c)
        top.step(range(len(vs)),vs,where="post",color=c,lw=2.2)
        top.axhline(vs[0],color="#727e82",ls="--",lw=1.3)
        for t in (0,12,24):
            top.scatter(t,vs[t],color=c,s=22,zorder=3)
            top.annotate(f"{vs[t]:.3f}",(t,vs[t]),xytext=(0,7),textcoords="offset points",ha="center",color=c,fontsize=9)
        top.set(xlim=(-1,25),xticks=[0,6,12,18,24],ylim=limits,xlabel="正史幕数")
        if not j:top.set_ylabel("参数值")
        top.grid(axis="y",alpha=.16)
        bot.set_title(f"{name}｜同题表达差值",loc="left",fontweight="bold",color=c)
        bot.axhline(0,color="#7b8589",ls="--",lw=1)
        points=[];secondary=[]
        for i,cp in enumerate(("0","12","24")):
            stage=data["primary"][a][cp]
            for n,row in enumerate(stage["pairs"]):
                g,f=row["scores"]
                if g is None or f is None:continue
                bot.scatter(i+(n-2.5)*.045,g-f,s=23,facecolors=c if row["valid"] else "none",
                            edgecolors=c,linewidths=.9,alpha=.68)
            points.append(stage["all"]["difference"]);secondary.append(data["secondary"][a][cp]["all"]["difference"])
        bot.plot(range(3),points,color=c,marker="D",ms=5,zorder=4)
        bot.plot(range(3),secondary,color="#586a72",ls=":",marker="x",ms=6,lw=1.2)
        bot.set(xlim=(-.35,2.35),ylim=(-2.3,2.3),xticks=[0,1,2],xticklabels=["起点","中段","后段"],yticks=[-2,-1,0,1,2])
        if not j:bot.set_ylabel("披露等级：成长 − 冻结")
        bot.grid(axis="y",alpha=.14)
    fig.text(.1,.04,"上：实线为成长，虚线为冻结。下：每点是一对回复，空心表示至少一端协议无效；菱形为主编码均差，叉号为第二编码均差。",fontsize=9,color="#40585f")
    fig.savefig(dst,dpi=160,facecolor="white");plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--source",type=Path,default=HERE/"observed/results.json")
    ap.add_argument("--output",type=Path,default=HERE/"trajectory.png");a=ap.parse_args();render(a.source,a.output)
