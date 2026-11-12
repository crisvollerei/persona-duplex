"""Show every primary ballot, including missing ones, rather than hide them in rates."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.patches import Patch, Rectangle
import trial as t

def render(dest):
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","Source Han Sans SC","PingFang SC","SimHei") if f in fonts),None)
    if not font:raise SystemExit("Chinese font required (e.g. Noto Sans CJK SC).")
    plt.rcParams.update({"font.family":font,"font.size":10,"axes.unicode_minus":False})
    fig,axs=plt.subplots(2,2,figsize=(11.2,5.6))
    fig.subplots_adjust(left=.145,right=.98,top=.78,bottom=.22,wspace=.31,hspace=.58)
    fig.text(.065,.94,"四种处境下，他们把自己的选择留给谁知道？",fontsize=19,weight="bold",color="#223d44")
    fig.text(.065,.865,"每格是独立的一次决定；空心表示场内装置没有收到可执行的落签。",color="#476069")
    colors={True:"#147773",False:"#e5b271",None:"white"}
    tasks=[q for q in t.SCENES if q["family"]=="relation"]
    for col,(phase,label) in enumerate((("observed","搭档自然交接"),("handoff","搭档递上操作单"))):
        records=t.read(t.HERE/phase/"choices.json")
        for row,(a,name) in enumerate((("tang","唐荔"),("yu","俞川"))):
            ax=axs[row,col]
            ax.set_title(name+" · "+label,loc="left",fontsize=12,weight="bold",color="#29474b")
            for i,q in enumerate(tasks):
                for j,b in enumerate(t.ARMS):
                    for n in range(4):
                        val=records[t.key((a,b,q,n))]["event"]["target"];x=j*4.7+n
                        ax.add_patch(Rectangle((x-.42,i-.37),.84,.74,facecolor=colors[val],
                                               edgecolor="#99a7aa" if val is None else colors[val],lw=.9))
            ax.set(ylim=(3.6,-.6),xlim=(-.6,13.1),xticks=[1.5,6.2,10.9],xticklabels=["成长","冻结","冻结复采"])
            ax.set_yticks(range(4),[q["name"] for q in tasks] if not col else [""]*4)
            ax.tick_params(length=0,pad=4)
            for spine in ax.spines.values():spine.set_visible(False)
    fig.legend([Patch(facecolor=colors[k],edgecolor="#8a999b" if k is None else colors[k]) for k in (True,False,None)],
               ["让搭档知道或看见","保留在自己一侧","未落定"],ncol=3,frameon=False,loc="lower left",bbox_to_anchor=(.06,.064))
    fig.text(.065,.035,"每组从左到右依次为：目标在前／青签、在后／青签、在前／白签、在后／白签。两种选择都合乎场景。",fontsize=9,color="#496067")
    fig.savefig(dest,dpi=155,facecolor="white");plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,default=t.HERE/"choices.png")
    render(ap.parse_args().out)
