"""Reproducible figure generation script for docs/validation/validation_study_paper.md."""
from __future__ import annotations

import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "experiment_results.json"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_font(size: int, bold: bool = False):
    candidates = (
        [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc"]
        if bold
        else [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\segoeui.ttf"]
    )
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_rounded_rect(draw: ImageDraw.ImageDraw, box, radius: int, fill=None, outline=None, width: int = 1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def generate_fig1(data: dict):
    """图 1：不同人格参数配置下的表达分化与站位轨迹对比 (C1-C4)"""
    W, H = 1600, 760
    img = Image.new("RGB", (W, H), (250, 251, 253))
    draw = ImageDraw.Draw(img)

    f_title = load_font(26, bold=True)
    f_sub = load_font(20, bold=True)
    f_label = load_font(16)
    f_small = load_font(14)

    draw.text((40, 24), "图 1  四组受控人格在相同 6 轮基准对话下的表达长度分离与把握度-站位轨迹", fill=(25, 32, 48), font=f_title)

    conds = {c["condition"]["id"]: c for c in data["conditions"]}
    p_keys = ["C1_shorekeeper_text", "C2_ache_text", "C3_guarded_text", "C4_service_text"]
    p_names = ["守岸人\n(bias=-0.10, subtle)", "阿澈\n(bias=+0.15, surface)", "疏离\n(bias=-0.28, subtle)", "客服\n(bias=+0.15, leg=0)"]
    colors = {
        "direct": (78, 121, 167),
        "baseline": (160, 203, 232),
        "say": (242, 142, 43),
    }

    draw_rounded_rect(draw, (35, 80, 775, 720), radius=12, fill=(255, 255, 255), outline=(222, 226, 235), width=2)
    draw.text((60, 100), "(a) 内部参考 (direct) 、人格基准 (baseline) 与实际回复 (SAY) 平均字数", fill=(35, 43, 62), font=f_sub)

    ax_l, ax_t, ax_r, ax_b = 110, 175, 740, 610
    max_y = 260
    for y_val in range(0, max_y + 1, 50):
        y_px = ax_b - int((y_val / max_y) * (ax_b - ax_t))
        draw.line([(ax_l, y_px), (ax_r, y_px)], fill=(238, 240, 245), width=1)
        draw.text((ax_l - 45, y_px - 10), str(y_val), fill=(100, 110, 125), font=f_small)
    draw.line([(ax_l, ax_t), (ax_l, ax_b)], fill=(140, 150, 165), width=2)
    draw.line([(ax_l, ax_b), (ax_r, ax_b)], fill=(140, 150, 165), width=2)

    group_w = (ax_r - ax_l) / len(p_keys)
    bar_w = 34
    for i, (pk, pname) in enumerate(zip(p_keys, p_names)):
        turns = conds[pk]["turns"]
        valid_t = [t for t in turns if t["direct_len"] > 0] or turns
        mean_direct = sum(t["direct_len"] for t in valid_t) / max(1, len(valid_t))
        mean_base = sum(t["baseline_len"] for t in valid_t) / max(1, len(valid_t))
        mean_say = sum(t["say_len"] for t in turns) / max(1, len(turns))

        cx = ax_l + group_w * (i + 0.5)
        for j, (metric_key, val, col) in enumerate([
            ("direct", mean_direct, colors["direct"]),
            ("baseline", mean_base, colors["baseline"]),
            ("say", mean_say, colors["say"]),
        ]):
            bx0 = int(cx + (j - 1.5) * (bar_w + 4))
            bx1 = bx0 + bar_w
            bh = int(min(val, max_y) / max_y * (ax_b - ax_t))
            by0 = ax_b - bh
            draw.rectangle([bx0, by0, bx1, ax_b], fill=col)
            draw.text((bx0 + 2, by0 - 20), f"{val:.0f}", fill=(45, 55, 72), font=f_small)

        draw.multiline_text((cx - 65, ax_b + 14), pname, fill=(45, 55, 72), font=f_small, align="center")

    leg_items = [("direct (内部中性参考)", colors["direct"]), ("baseline (人格常态基准)", colors["baseline"]), ("SAY (实际外显回复)", colors["say"])]
    lx = 115
    for label, col in leg_items:
        draw.rectangle([lx, 142, lx + 20, 158], fill=col)
        draw.text((lx + 26, 140), label, fill=(55, 65, 82), font=f_small)
        lx += 205

    draw_rounded_rect(draw, (810, 80, 1565, 720), radius=12, fill=(255, 255, 255), outline=(222, 226, 235), width=2)
    draw.text((835, 100), "(b) 各人格在 6 轮对话中的有效把握度 (conf + bias) 与站位决策带", fill=(35, 43, 62), font=f_sub)

    bx_l, bx_t, bx_r, bx_b = 900, 175, 1525, 610
    bands = [
        (0.80, 1.00, (232, 245, 233), "direct (≥0.80)"),
        (0.50, 0.80, (227, 242, 253), "guess_first (0.50–0.80)"),
        (0.25, 0.50, (255, 248, 225), "ask_only (0.25–0.50)"),
        (0.00, 0.25, (252, 228, 236), "admit_unsure (<0.25)"),
    ]
    for lo, hi, bcol, blabel in bands:
        y0 = bx_b - int(hi * (bx_b - bx_t))
        y1 = bx_b - int(lo * (bx_b - bx_t))
        draw.rectangle([bx_l, y0, bx_r, y1], fill=bcol)
        draw.text((bx_r - 175, y0 + 6), blabel, fill=(90, 100, 115), font=f_small)

    for v in [0.0, 0.25, 0.50, 0.80, 1.0]:
        y_px = bx_b - int(v * (bx_b - bx_t))
        draw.line([(bx_l, y_px), (bx_r, y_px)], fill=(180, 190, 205), width=1)
        draw.text((bx_l - 48, y_px - 10), f"{v:.2f}", fill=(80, 90, 105), font=f_small)

    draw.line([(bx_l, bx_t), (bx_l, bx_b)], fill=(120, 130, 145), width=2)
    draw.line([(bx_l, bx_b), (bx_r, bx_b)], fill=(120, 130, 145), width=2)

    p_line_colors = {
        "C1_shorekeeper_text": (31, 119, 180),
        "C2_ache_text": (214, 39, 40),
        "C3_guarded_text": (148, 103, 189),
        "C4_service_text": (44, 160, 44),
    }
    p_biases = {
        "C1_shorekeeper_text": -0.10,
        "C2_ache_text": +0.15,
        "C3_guarded_text": -0.28,
        "C4_service_text": +0.15,
    }
    p_short = {
        "C1_shorekeeper_text": "守岸人 (-0.10)",
        "C2_ache_text": "阿澈 (+0.15)",
        "C3_guarded_text": "疏离 (-0.28)",
        "C4_service_text": "客服 (+0.15)",
    }

    for t_idx in range(1, 7):
        x_px = bx_l + int((t_idx - 1) / 5 * (bx_r - bx_l - 40)) + 20
        draw.line([(x_px, bx_t), (x_px, bx_b)], fill=(220, 225, 232), width=1)
        draw.text((x_px - 12, bx_b + 14), f"T{t_idx}", fill=(45, 55, 72), font=f_label)

    for pk in p_keys:
        col = p_line_colors[pk]
        bias = p_biases[pk]
        pts = []
        for t in conds[pk]["turns"]:
            c = t["confidence"] if t["confidence"] is not None else 0.5
            eff = max(0.0, min(1.0, c + bias))
            x_px = bx_l + int((t["turn_seq"] - 1) / 5 * (bx_r - bx_l - 40)) + 20
            y_px = bx_b - int(eff * (bx_b - bx_t))
            pts.append((x_px, y_px))
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=3)
        for x_px, y_px in pts:
            draw.ellipse([x_px - 5, y_px - 5, x_px + 5, y_px + 5], fill=col, outline=(255, 255, 255), width=1)

    lx = 890
    for pk in p_keys:
        col = p_line_colors[pk]
        draw.line([(lx, 150), (lx + 24, 150)], fill=col, width=4)
        draw.ellipse([lx + 8, 146, lx + 16, 154], fill=col)
        draw.text((lx + 30, 140), p_short[pk], fill=(45, 55, 72), font=f_small)
        lx += 155

    out = FIGURES_DIR / "fig1_persona_differentiation.png"
    img.save(out)
    print("Saved", out)


def generate_fig2(data: dict):
    """图 2：三种生成协议 (text / tool / persona+fidelity) 的性能、调用开销与结构有效性对比"""
    W, H = 1600, 740
    img = Image.new("RGB", (W, H), (250, 251, 253))
    draw = ImageDraw.Draw(img)

    f_title = load_font(26, bold=True)
    f_sub = load_font(20, bold=True)
    f_label = load_font(16)
    f_small = load_font(14)

    draw.text((40, 24), "图 2  三种生成协议 (text / tool / persona+check_fidelity) 的单轮耗时与调用结构对比", fill=(25, 32, 48), font=f_title)

    conds = {c["condition"]["id"]: c for c in data["conditions"]}
    proto_keys = ["C1_shorekeeper_text", "C5_shorekeeper_tool", "C6_shorekeeper_persona"]
    proto_labels = ["text 协议\n(单次三段流式)", "tool 协议\n(inner + 流式SAY + tail)", "persona 协议\n(独立raw + 表达 + 一致性检查)"]
    proto_colors = [(57, 106, 177), (218, 124, 48), (62, 150, 81)]

    draw_rounded_rect(draw, (35, 80, 820, 700), radius=12, fill=(255, 255, 255), outline=(222, 226, 235), width=2)
    draw.text((60, 100), "(a) 守岸人在三种协议下各轮端到端响应耗时 (秒)", fill=(35, 43, 62), font=f_sub)

    ax_l, ax_t, ax_r, ax_b = 105, 175, 785, 600
    max_t = 16.0
    for y_v in range(0, 17, 4):
        y_px = ax_b - int((y_v / max_t) * (ax_b - ax_t))
        draw.line([(ax_l, y_px), (ax_r, y_px)], fill=(235, 238, 244), width=1)
        draw.text((ax_l - 40, y_px - 10), f"{y_v}s", fill=(90, 100, 115), font=f_small)

    draw.line([(ax_l, ax_t), (ax_l, ax_b)], fill=(120, 130, 145), width=2)
    draw.line([(ax_l, ax_b), (ax_r, ax_b)], fill=(120, 130, 145), width=2)

    for t_idx in range(1, 7):
        x_px = ax_l + int((t_idx - 1) / 5 * (ax_r - ax_l - 40)) + 20
        draw.line([(x_px, ax_t), (x_px, ax_b)], fill=(235, 238, 244), width=1)
        draw.text((x_px - 12, ax_b + 14), f"T{t_idx}", fill=(45, 55, 72), font=f_label)

    for pk, plab, col in zip(proto_keys, ["text (1 call)", "tool (3 calls)", "persona+check (4 calls)"], proto_colors):
        pts = []
        for t in conds[pk]["turns"]:
            x_px = ax_l + int((t["turn_seq"] - 1) / 5 * (ax_r - ax_l - 40)) + 20
            y_px = ax_b - int(min(t["elapsed_s"], max_t) / max_t * (ax_b - ax_t))
            pts.append((x_px, y_px, t["elapsed_s"]))
        draw.line([(x, y) for x, y, _ in pts], fill=col, width=3)
        for x, y, val in pts:
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col, outline=(255, 255, 255), width=1)
            draw.text((x - 14, y - 22), f"{val:.1f}s", fill=col, font=f_small)

    lx = 110
    for plab, col in zip(["text (1次前台调用)", "tool (3次前台调用)", "persona+核查 (4次前台调用)"], proto_colors):
        draw.line([(lx, 150), (lx + 24, 150)], fill=col, width=4)
        draw.text((lx + 30, 140), plab, fill=(45, 55, 72), font=f_small)
        lx += 220

    draw_rounded_rect(draw, (850, 80, 1565, 700), radius=12, fill=(255, 255, 255), outline=(222, 226, 235), width=2)
    draw.text((875, 100), "(b) 三种协议的平均耗时、前台调用数与协议有效率对比", fill=(35, 43, 62), font=f_sub)

    bx_l, bx_t, bx_r, bx_b = 930, 195, 1525, 580
    draw.line([(bx_l, bx_t), (bx_l, bx_b)], fill=(120, 130, 145), width=2)
    draw.line([(bx_l, bx_b), (bx_r, bx_b)], fill=(120, 130, 145), width=2)

    group_w = (bx_r - bx_l) / 3
    for i, (pk, plab, col) in enumerate(zip(proto_keys, proto_labels, proto_colors)):
        turns = conds[pk]["turns"]
        mean_lat = sum(t["elapsed_s"] for t in turns) / max(1, len(turns))
        mean_calls = sum(t.get("calls_count", 1) for t in turns) / max(1, len(turns))
        valid_rate = sum(1 for t in turns if t["validity"].get("valid")) / max(1, len(turns)) * 100.0

        cx = bx_l + group_w * (i + 0.5)
        bars = [
            (mean_lat / 15.0, f"{mean_lat:.2f}s", "平均耗时", (78, 121, 167)),
            (mean_calls / 6.0, f"{mean_calls:.1f}次", "前台调用", (242, 142, 43)),
            (valid_rate / 100.0, f"{valid_rate:.0f}%", "有效率", (89, 161, 79)),
        ]
        bw = 42
        for j, (ratio, txt, sub_l, bcol) in enumerate(bars):
            x0 = int(cx + (j - 1.5) * (bw + 8))
            x1 = x0 + bw
            bh = int(min(max(ratio, 0.02), 1.0) * (bx_b - bx_t))
            y0 = bx_b - bh
            draw.rectangle([x0, y0, x1, bx_b], fill=bcol)
            draw.text((x0 - 2, y0 - 22), txt, fill=(35, 45, 60), font=f_small)
        draw.multiline_text((cx - 75, bx_b + 16), plab, fill=(45, 55, 72), font=f_small, align="center")

    lx = 930
    for label, bcol in [("平均单轮耗时 (秒)", (78, 121, 167)), ("平均前台模型调用次数", (242, 142, 43)), ("本地协议有效率 (%)", (89, 161, 79))]:
        draw.rectangle([lx, 145, lx + 20, 161], fill=bcol)
        draw.text((lx + 26, 143), label, fill=(55, 65, 82), font=f_small)
        lx += 195

    out = FIGURES_DIR / "fig2_protocol_comparison.png"
    img.save(out)
    print("Saved", out)


def generate_fig3(data: dict):
    """图 3：双时间尺度人格参数可塑性限幅 (push_traits) 与长期锚点频率衰减曲线 (push_anchors)"""
    W, H = 1600, 740
    img = Image.new("RGB", (W, H), (250, 251, 253))
    draw = ImageDraw.Draw(img)

    f_title = load_font(26, bold=True)
    f_sub = load_font(20, bold=True)
    f_label = load_font(16)
    f_small = load_font(14)

    draw.text((40, 24), "图 3  双时间尺度参数动力学：中期可塑性漂移边界截断与长期基线锚点频率衰减", fill=(25, 32, 48), font=f_title)

    plast = data["plasticity_experiment"]
    init_t = plast["initial_traits"]
    step1_t = plast["after_step1"]
    clamp_t = plast["after_clamped_15_steps"]

    draw_rounded_rect(draw, (35, 80, 810, 700), radius=12, fill=(255, 255, 255), outline=(222, 226, 235), width=2)
    draw.text((60, 100), "(a) 中期参数推动 (push_traits) 与漂移半径 [anchor ± drift] 硬截断验证", fill=(35, 43, 62), font=f_sub)

    ax_l, ax_t, ax_r, ax_b = 110, 185, 775, 595
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y_px = ax_b - int(v * (ax_b - ax_t))
        draw.line([(ax_l, y_px), (ax_r, y_px)], fill=(235, 238, 244), width=1)
        draw.text((ax_l - 45, y_px - 10), f"{v:.1f}", fill=(90, 100, 115), font=f_small)
    draw.line([(ax_l, ax_t), (ax_l, ax_b)], fill=(120, 130, 145), width=2)
    draw.line([(ax_l, ax_b), (ax_r, ax_b)], fill=(120, 130, 145), width=2)

    t_names = ["learning_openness", "warmth", "intervention", "brevity"]
    t_labels = ["learning_openness\n(plast=0.70, drift=0.18)", "warmth\n(plast=0.20, drift=0.12)", "intervention\n(plast=0.10, drift=0.08)", "brevity\n(plast=0.15, drift=0.10)"]
    gw = (ax_r - ax_l) / len(t_names)

    for i, (tn, tlab) in enumerate(zip(t_names, t_labels)):
        cx = ax_l + gw * (i + 0.5)
        anc = init_t[tn]["anchor"]
        drf = init_t[tn]["drift"]
        lo_b = max(init_t[tn]["lo"], anc - drf)
        hi_b = min(init_t[tn]["hi"], anc + drf)

        y_hi = ax_b - int(hi_b * (ax_b - ax_t))
        y_lo = ax_b - int(lo_b * (ax_b - ax_t))
        draw.rectangle([int(cx - 56), y_hi, int(cx + 56), y_lo], fill=(232, 240, 254), outline=(180, 205, 240), width=1)

        vals = [
            (anc, (110, 120, 135)),
            (step1_t[tn]["value"], (57, 106, 177)),
            (clamp_t[tn]["value"], (214, 39, 40)),
        ]
        bw = 28
        for j, (v, col) in enumerate(vals):
            x0 = int(cx + (j - 1.5) * (bw + 14))
            x1 = x0 + bw
            y0 = ax_b - int(v * (ax_b - ax_t))
            draw.rectangle([x0, y0, x1, ax_b], fill=col)
            draw.text((x0 - 6, y0 - 20), f"{v:.3f}", fill=col, font=f_small)
        draw.multiline_text((cx - 72, ax_b + 14), tlab, fill=(45, 55, 72), font=f_small, align="center")

    lx = 65
    for lab, col, step_x in [
        ("初始基线 anchor", (110, 120, 135), 155),
        ("单批推动后 value", (57, 106, 177), 165),
        ("15批连续推动后 value", (214, 39, 40), 190),
        ("允许漂移带 [anchor±drift]", (200, 220, 250), 200),
    ]:
        draw.rectangle([lx, 145, lx + 18, 161], fill=col)
        draw.text((lx + 24, 143), lab, fill=(55, 65, 82), font=f_small)
        lx += step_x

    draw_rounded_rect(draw, (840, 80, 1565, 700), radius=12, fill=(255, 255, 255), outline=(222, 226, 235), width=2)
    draw.text((865, 100), "(b) 长期锚点推动有效系数 f(Δt, k) = (1 - e^{-Δt/6h}) / (1 + k) 衰减曲线", fill=(35, 43, 62), font=f_sub)

    bx_l, bx_t, bx_r, bx_b = 915, 185, 1530, 595
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y_px = bx_b - int(v * (bx_b - bx_t))
        draw.line([(bx_l, y_px), (bx_r, y_px)], fill=(235, 238, 244), width=1)
        draw.text((bx_l - 45, y_px - 10), f"{v:.1f}", fill=(90, 100, 115), font=f_small)
    draw.line([(bx_l, bx_t), (bx_l, bx_b)], fill=(120, 130, 145), width=2)
    draw.line([(bx_l, bx_b), (bx_r, bx_b)], fill=(120, 130, 145), width=2)

    for h_val in [0, 4, 8, 12, 16, 20, 24]:
        x_px = bx_l + int((h_val / 24.0) * (bx_r - bx_l))
        draw.line([(x_px, bx_t), (x_px, bx_b)], fill=(235, 238, 244), width=1)
        draw.text((x_px - 14, bx_b + 14), f"{h_val}h", fill=(45, 55, 72), font=f_small)
    draw.text((bx_l + 220, bx_b + 42), "距上次锚点推动的时间间隔 Δt (小时)", fill=(45, 55, 72), font=f_label)

    y_min = bx_b - int(0.02 * (bx_b - bx_t))
    draw.line([(bx_l, y_min), (bx_r, y_min)], fill=(214, 39, 40), width=1)
    draw.text((bx_r - 220, y_min - 20), "最小生效阈值 ANCHOR_MIN_EFFECT=0.02", fill=(214, 39, 40), font=f_small)

    curves = [
        (0, "k=0 (2h窗口内无近期推动)", (44, 160, 44)),
        (1, "k=1 (2h窗口内已推 1 次)", (57, 106, 177)),
        (3, "k=3 (2h窗口内已推 3 次)", (218, 124, 48)),
    ]
    for k, clab, col in curves:
        pts = []
        for step in range(0, 97):
            dt_h = step * 0.25
            f1 = 1.0 - math.exp(-dt_h / 6.0)
            f2 = 1.0 / (1.0 + k)
            eff = f1 * f2
            x_px = bx_l + int((dt_h / 24.0) * (bx_r - bx_l))
            y_px = bx_b - int(eff * (bx_b - bx_t))
            pts.append((x_px, y_px))
        draw.line(pts, fill=col, width=3)

    lx = 905
    for _, clab, col in curves:
        draw.line([(lx, 150), (lx + 22, 150)], fill=col, width=4)
        draw.text((lx + 28, 140), clab, fill=(45, 55, 72), font=f_small)
        lx += 215

    out = FIGURES_DIR / "fig3_plasticity_and_anchor_decay.png"
    img.save(out)
    print("Saved", out)


def generate_fig4(data: dict):
    """图 4：纯数值参数受控消融实验 (A1 vs A2) 与语域突变修复状态机 (B1 vs B2) 轨迹"""
    W, H = 1600, 760
    img = Image.new("RGB", (W, H), (250, 251, 253))
    draw = ImageDraw.Draw(img)

    f_title = load_font(26, bold=True)
    f_sub = load_font(20, bold=True)
    f_label = load_font(16)
    f_small = load_font(14)

    draw.text((40, 24), "图 4  纯数值参数受控消融实验 (identity 恒为空) 与语域突变修复闭环验证", fill=(25, 32, 48), font=f_title)

    abl = data["ablation_experiment"]
    a1_turns = abl["A1_LowInterv_NegBias"]["turns"]
    a2_turns = abl["A2_HighInterv_PosBias"]["turns"]

    draw_rounded_rect(draw, (35, 80, 810, 720), radius=12, fill=(255, 255, 255), outline=(222, 226, 235), width=2)
    draw.text((60, 100), "(a) 零身份文本 (identity=\"\") 下纯数值参数组的逐轮 SAY 长度与站位分离", fill=(35, 43, 62), font=f_sub)

    ax_l, ax_t, ax_r, ax_b = 105, 190, 775, 595
    max_len = 180
    for y_v in range(0, 181, 30):
        y_px = ax_b - int((y_v / max_len) * (ax_b - ax_t))
        draw.line([(ax_l, y_px), (ax_r, y_px)], fill=(235, 238, 244), width=1)
        draw.text((ax_l - 42, y_px - 10), str(y_v), fill=(90, 100, 115), font=f_small)
    draw.line([(ax_l, ax_t), (ax_l, ax_b)], fill=(120, 130, 145), width=2)
    draw.line([(ax_l, ax_b), (ax_r, ax_b)], fill=(120, 130, 145), width=2)

    gw = (ax_r - ax_l) / 6
    bw = 32
    col_a1 = (57, 106, 177)
    col_a2 = (218, 124, 48)
    short_stance = {
        "admit_unsure": "unsure",
        "ask_only": "ask",
        "guess_first": "guess",
        "direct": "direct",
    }

    for i in range(6):
        t1 = a1_turns[i]
        t2 = a2_turns[i]
        cx = ax_l + gw * (i + 0.5)

        x0_1 = int(cx - bw - 4)
        x1_1 = x0_1 + bw
        y0_1 = ax_b - int(min(t1["say_len"], max_len) / max_len * (ax_b - ax_t))
        draw.rectangle([x0_1, y0_1, x1_1, ax_b], fill=col_a1)
        draw.text((x0_1 - 2, y0_1 - 36), f"{t1['say_len']}", fill=col_a1, font=f_small)
        draw.text((x0_1 - 6, y0_1 - 18), short_stance.get(t1["stance"], t1["stance"]), fill=col_a1, font=f_small)

        x0_2 = int(cx + 4)
        x1_2 = x0_2 + bw
        y0_2 = ax_b - int(min(t2["say_len"], max_len) / max_len * (ax_b - ax_t))
        draw.rectangle([x0_2, y0_2, x1_2, ax_b], fill=col_a2)
        draw.text((x0_2 + 2, y0_2 - 36), f"{t2['say_len']}", fill=col_a2, font=f_small)
        draw.text((x0_2 - 2, y0_2 - 18), short_stance.get(t2["stance"], t2["stance"]), fill=col_a2, font=f_small)

        draw.text((int(cx - 14), ax_b + 14), f"T{i+1}", fill=(45, 55, 72), font=f_label)

    draw.rectangle([105, 142, 125, 158], fill=col_a1)
    draw.text((132, 140), "A1 低干涉/负偏移 (bias=-0.30, interv=0.10, brev=0.90) 均值 28.5字", fill=(45, 55, 72), font=f_small)
    draw.rectangle([105, 650, 125, 666], fill=col_a2)
    draw.text((132, 648), "A2 高干涉/正偏移 (bias=+0.25, interv=0.90, brev=0.10) 均值 93.2字 (3.27×)", fill=(45, 55, 72), font=f_small)

    draw_rounded_rect(draw, (840, 80, 1565, 720), radius=12, fill=(255, 255, 255), outline=(222, 226, 235), width=2)
    draw.text((865, 100), "(b) 语域升幅检测 (Δformality > 0.30) 与修复状态机双路径闭环", fill=(35, 43, 62), font=f_sub)

    bx_l, bx_t, bx_r, bx_b = 920, 190, 1520, 500
    for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y_px = bx_b - int(v * (bx_b - bx_t))
        draw.line([(bx_l, y_px), (bx_r, y_px)], fill=(235, 238, 244), width=1)
        draw.text((bx_l - 48, y_px - 10), f"{v:.2f}", fill=(90, 100, 115), font=f_small)
    draw.line([(bx_l, bx_t), (bx_l, bx_b)], fill=(120, 130, 145), width=2)
    draw.line([(bx_l, bx_b), (bx_r, bx_b)], fill=(120, 130, 145), width=2)

    user_formalities = [0.00, 0.80, 0.40, 0.40]
    base_formalities = [0.30, 0.255, 0.3368, 0.3463]

    xs = [bx_l + int(i / 3 * (bx_r - bx_l - 60)) + 30 for i in range(4)]
    for i, x_px in enumerate(xs):
        draw.line([(x_px, bx_t), (x_px, bx_b)], fill=(235, 238, 244), width=1)
        draw.text((x_px - 15, bx_b + 12), f"T{i+1}", fill=(45, 55, 72), font=f_label)

    y_base_t2 = bx_b - int(base_formalities[1] * (bx_b - bx_t))
    y_user_t2 = bx_b - int(user_formalities[1] * (bx_b - bx_t))
    draw.rectangle([xs[1] - 28, y_user_t2, xs[1] + 28, y_base_t2], fill=(253, 232, 232), outline=(214, 39, 40), width=1)
    draw.text((xs[1] + 36, y_user_t2 - 6), "Δ = +0.545 > 0.30\n触发 SUSPECTED_HURT", fill=(200, 35, 35), font=f_small)

    pts_u = [(xs[i], bx_b - int(user_formalities[i] * (bx_b - bx_t))) for i in range(4)]
    pts_b = [(xs[i], bx_b - int(base_formalities[i] * (bx_b - bx_t))) for i in range(4)]
    draw.line(pts_b, fill=(120, 130, 145), width=3)
    draw.line(pts_u, fill=(214, 39, 40), width=3)
    for (xu, yu), (xb, yb), uf, bf in zip(pts_u, pts_b, user_formalities, base_formalities):
        draw.ellipse([xb - 5, yb - 5, xb + 5, yb + 5], fill=(120, 130, 145))
        draw.ellipse([xu - 6, yu - 6, xu + 6, yu + 6], fill=(214, 39, 40))
        draw.text((xu - 18, yu - 24), f"{uf:.2f}", fill=(214, 39, 40), font=f_small)

    draw.line([(920, 148), (945, 148)], fill=(214, 39, 40), width=3)
    draw.text((952, 138), "当轮用户语域分 formality(u)", fill=(45, 55, 72), font=f_small)
    draw.line([(1210, 148), (1235, 148)], fill=(120, 130, 145), width=3)
    draw.text((1242, 138), "慢速语域基线 baseline (α=0.15)", fill=(45, 55, 72), font=f_small)

    states_b1 = ["T1: 无异常 (—)", "T2: 记账 (open)", "T3: 认领 (repairing)", "T4: 原谅 (healed)"]
    states_b2 = ["T1: 无异常 (—)", "T2: 记账 (open)", "T3: 认领 (repairing)", "T4: 叫停 (paused)"]
    draw.text((865, 555), "B1 接受路径：", fill=(44, 130, 60), font=f_small)
    draw.text((865, 615), "B2 叫停路径：", fill=(180, 95, 30), font=f_small)
    for i in range(4):
        bx0 = 975 + i * 142
        draw_rounded_rect(draw, (bx0, 545, bx0 + 134, 588), radius=6, fill=(232, 246, 235) if i == 3 else (240, 244, 250), outline=(160, 195, 170) if i == 3 else (190, 200, 215))
        draw.text((bx0 + 8, 557), states_b1[i], fill=(35, 50, 65), font=f_small)
        draw_rounded_rect(draw, (bx0, 605, bx0 + 134, 648), radius=6, fill=(254, 243, 230) if i == 3 else (240, 244, 250), outline=(220, 175, 130) if i == 3 else (190, 200, 215))
        draw.text((bx0 + 8, 617), states_b2[i], fill=(35, 50, 65), font=f_small)
    draw.text((865, 668), "注：B1 在 T4 销账并移出 unresolved；B2 在 T4 转为 paused 停止追问，且不升级为永久雷区 (sore_points=[])。", fill=(90, 100, 115), font=f_small)

    out = FIGURES_DIR / "fig4_ablation_and_mechanisms.png"
    img.save(out)
    print("Saved", out)


def main():
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    generate_fig1(data)
    generate_fig2(data)
    generate_fig3(data)
    generate_fig4(data)


if __name__ == "__main__":
    main()
