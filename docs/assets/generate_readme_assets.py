"""Render the repository's original vector cover and local raster previews.

Requires the optional matplotlib dependency used by the research figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PALETTES = {
    "light": {"paper": "#f6f3eb", "ink": "#183332", "muted": "#5d706a",
              "line": "#c7cfc5", "card": "#fffdf7", "teal": "#187c78",
              "wash": "#e0ede4", "amber": "#bb763e"},
    "dark": {"paper": "#182827", "ink": "#edece1", "muted": "#acbdb3",
             "line": "#435650", "card": "#20332f", "teal": "#78c8b3",
             "wash": "#304f43", "amber": "#e1ac78"},
}


def render(theme: str, preview_dir: Path | None):
    c = PALETTES[theme]
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "persona-duplex-readme",
                         "font.family": "DejaVu Sans",
                         "savefig.transparent": False})
    fig = plt.figure(figsize=(14, 4.4), dpi=100, facecolor=c["paper"])
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1400), ylim=(440, 0))
    ax.axis("off")
    for x in range(660, 1400, 36):
        ax.plot([x, x], [0, 440], color=c["line"], lw=.45, alpha=.28)
    for y in range(8, 440, 36):
        ax.plot([640, 1400], [y, y], color=c["line"], lw=.45, alpha=.28)
    ax.add_patch(Circle((1330, 10), 230, color=c["teal"], alpha=.06, lw=0))
    ax.add_patch(Circle((705, 442), 170, color=c["amber"], alpha=.05, lw=0))

    def text(x, y, value, size=12, color=None, family=None, **kwargs):
        ax.text(x, y, value, fontsize=size, color=color or c["ink"],
                family=family or "DejaVu Sans", va="top", **kwargs)

    def box(x, y, width, height, *, fill=None, edge=None, radius=14, lw=1):
        patch = FancyBboxPatch((x, y), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill or c["card"], edgecolor=edge or c["line"], linewidth=lw)
        ax.add_patch(patch)
        return patch

    text(72, 43, "PERSONA-DUPLEX  /  OPEN RESEARCH", 9.5,
         color=c["muted"], family="DejaVu Sans Mono")
    text(64, 87, "persona", 84, family="STIXGeneral", fontstyle="italic")
    text(65, 209, "duplex", 84, family="STIXGeneral", fontweight="bold")
    ax.plot([77, 135], [346, 346], color=c["amber"], lw=3)
    text(77, 365, "Expression, with structure.", 19, family="STIXGeneral", fontstyle="italic")

    text(684, 75, "01  /  ONE FOREGROUND GENERATION", 10,
         color=c["muted"], family="DejaVu Sans Mono")
    outer = box(663, 119, 666, 164, fill=c["paper"], radius=22)
    outer.set_alpha(.92)
    nodes = [(686, "HEAD", "reference", c["line"]),
             (907, "SAY", "visible reply", c["teal"]),
             (1128, "TAIL", "observation", c["amber"])]
    for i, (x, title, subtitle, edge) in enumerate(nodes):
        box(x, 148, 176, 107, fill=c["wash"] if title == "SAY" else c["card"],
            edge=edge, radius=15, lw=1.8 if title == "SAY" else 1)
        text(x+22, 170, title, 22, family="DejaVu Sans Mono", fontweight="bold")
        text(x+22, 215, subtitle, 10.5, color=c["muted"])
        if i < 2:
            ax.add_patch(FancyArrowPatch((x+184, 201), (x+213, 201),
                arrowstyle="-|>", mutation_scale=11, color=c["muted"], lw=1))
    ax.add_patch(FancyArrowPatch((1217, 284), (1217, 326), arrowstyle="-|>",
        mutation_scale=12, color=c["amber"], lw=1, linestyle=(0, (3, 4))))
    box(945, 327, 360, 49, fill=c["paper"], edge=c["amber"], radius=13)
    text(967, 340, "02  /  ASYNC MEMORY", 12, family="DejaVu Sans Mono")
    text(957, 390, "user evidence / append-only ledger", 9, color=c["muted"])

    target = HERE / f"readme-hero-{theme}.svg"
    fig.savefig(target, format="svg", metadata={"Date": None, "Creator": "persona-duplex"})
    svg = target.read_text(encoding="utf-8")
    target.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                      encoding="utf-8", newline="\n")
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"readme-hero-{theme}.png", dpi=100)
    plt.close(fig)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    for theme in PALETTES:
        render(theme, args.preview_dir)


if __name__ == "__main__": main()
