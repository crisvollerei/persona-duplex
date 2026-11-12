"""Render the decisions behind the precision and recall counts."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.patches import Patch, Rectangle
import study as s

def render(dest):
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","Source Han Sans SC","PingFang SC","SimHei") if f in fonts),None)
    if not font:raise RuntimeError("Install a Chinese font (e.g. Noto Sans CJK SC).")
    plt.rcParams.update({"font.family":font,"font.size":10,"axes.unicode_minus":False})
    pairs,refs=s.exchanges()
    new=[s.read(s.HERE/f/"attribution.json") for f in ("observed","confirmed")]
    colors={"yes":"#167875","no":"#f6f5ef","wrong":"#c55b32","maybe":"#a197b6","veto":"#d8dde2"}
    fig,axs=plt.subplots(2,1,figsize=(10,4.6));fig.subplots_adjust(left=.155,right=.98,bottom=.28,top=.78,hspace=.72)
    fig.text(.075,.94,"同一句话，不同判法把经历记给了谁？",fontsize=17,weight="bold",color="#20383e")
    fig.text(.075,.865,"每列是一幕，每格是该角色一次记账机会。灰列不符合原有协议要求，不参与三种方法的比较。",color="#476069")
    for ax,a in zip(axs,s.prev.ROLES):
        ax.set_title(s.prev.NAME[a],loc="left",fontsize=11,weight="bold",pad=4)
        for ep,e in enumerate(s.hist,1):
            k=f"H{ep:02}.{a}";r=refs[k];vs=[r,e["evidence"][a]["verified"],*(u[k]["vote"] for u in new)]
            for i,v in enumerate(vs):
                if not pairs[k].protocol_ok:color="veto"
                elif not i:color="maybe" if r is None else ("yes" if r else "no")
                elif not v:color="no"
                else:color="maybe" if r is None else ("yes" if v==r else "wrong")
                ax.add_patch(Rectangle((ep-.485,i-.47),.97,.94,facecolor=colors[color],edgecolor="#ffffff",lw=.5))
        ax.set(xlim=(.5,24.5),ylim=(3.5,-.5),yticks=range(4),yticklabels=["文本参考","原判定","核对双方归属","先提取再看回应"])
        ax.set_xticks(range(1,25));ax.tick_params(length=0,pad=2,labelsize=9)
        for sp in ax.spines.values():sp.set_visible(False)
    axs[-1].set_xlabel("剧情幕次")
    labels={"yes":"参考正例／正确入账","wrong":"误记为成长","maybe":"参考有歧义／此类票入账","no":"参考无证据／未入账","veto":"协议不完整"}
    fig.legend([Patch(facecolor=colors[k],edgecolor="#ced4d3") for k in labels],list(labels.values()),
               ncol=3,loc="lower center",frameon=False,fontsize=9,bbox_to_anchor=(.54,.015))
    fig.savefig(dest,dpi=155,bbox_inches="tight",pad_inches=.18);plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,default=s.HERE/"evidence.png");render(p.parse_args().out)
