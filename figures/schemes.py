# -*- coding: utf-8 -*-
"""Схемы для диплома: шаблоны отмывания, общая схема метода, слой GNN. PNG 300 dpi, светлый фон для печати."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d6d0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
from pathlib import Path
OUT = str(Path(__file__).resolve().parent) + "/"


def arrow(ax, p, q, color=MUTED, lw=1.4, rad=0.0, shrink=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=lw, color=color,
                                 shrinkA=shrink, shrinkB=shrink, connectionstyle=f"arc3,rad={rad}"))


def node(ax, xy, color=BLUE, r=0.075):
    ax.add_patch(plt.Circle(xy, r, color=color, zorder=3))
    ax.add_patch(plt.Circle(xy, r, fill=False, color="white", lw=1.5, zorder=4))


# ---------------------------------------------------------------- 1. шаблоны отмывания
import math
pats = {}
pats["Веер наружу (fan-out)"] = ([(0, 0)] + [(1, y) for y in (-0.9, -0.3, 0.3, 0.9)], [(0, i) for i in range(1, 5)], {0})
pats["Веер внутрь (fan-in)"] = ([(1, 0)] + [(0, y) for y in (-0.9, -0.3, 0.3, 0.9)], [(i, 0) for i in range(1, 5)], {0})
pats["Сбор–рассылка"] = ([(0, y) for y in (-0.8, 0, 0.8)] + [(0.75, 0)] + [(1.5, y) for y in (-0.8, 0, 0.8)],
                         [(0, 3), (1, 3), (2, 3), (3, 4), (3, 5), (3, 6)], {3})
pats["Рассылка–сбор"] = ([(0, 0)] + [(0.75, y) for y in (-0.8, 0, 0.8)] + [(1.5, 0)],
                         [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)], {0, 4})
cyc = [(0.75 + 0.75 * math.cos(2 * math.pi * k / 5 + math.pi / 2), 0.75 * math.sin(2 * math.pi * k / 5 + math.pi / 2))
       for k in range(5)]
pats["Цикл"] = (cyc, [(k, (k + 1) % 5) for k in range(5)], {0})
pats["Двудольный граф"] = ([(0, y) for y in (-0.8, 0, 0.8)] + [(1.5, y) for y in (-0.5, 0.5)],
                           [(0, 3), (0, 4), (1, 3), (2, 4), (1, 4)], set())
pats["Стек"] = ([(0, y) for y in (-0.6, 0.6)] + [(0.75, y) for y in (-0.6, 0.6)] + [(1.5, y) for y in (-0.6, 0.6)],
                [(0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (2, 5), (3, 4), (3, 5)], set())
pats["Случайный обход"] = ([(0, -0.6), (0.5, 0.5), (1.0, -0.2), (1.5, 0.7), (1.6, -0.8)],
                           [(0, 1), (1, 2), (2, 3), (3, 4)], {0})
fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
for ax, (title, (pos, edges, hub)) in zip(axes.ravel(), pats.items()):
    for a, b in edges:
        arrow(ax, pos[a], pos[b], rad=0.0)
    for i, p in enumerate(pos):
        node(ax, p, ORANGE if i in hub else BLUE)
    ax.set_title(title, fontsize=10.5, color=INK, pad=6)
    ax.set_xlim(-0.35, 1.85); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
fig.text(0.5, 0.015, "Узлы — счета, стрелки — переводы. Оранжевым отмечены центральные счета шаблона.",
         ha="center", color=MUTED, fontsize=9.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT + "fig_patterns.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 2. общая схема метода


def box(ax, x, y, w, h, text, edge=BLUE, fill="#eef4fc", fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fill, ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


fig, ax = plt.subplots(figsize=(12, 3.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")
steps = [
    (0.1, "Переводы\n(отправитель,\nполучатель, сумма,\nвремя, валюта)", BLUE, "#eef4fc"),
    (2.05, "Граф счетов\nи переводов\n(протокол\nтрёх графов)", BLUE, "#eef4fc"),
    (4.0, "Модель\nXGBoost / GNN /\nгибрид\n→ p", BLUE, "#eef4fc"),
    (5.95, "Калибровка\nπ̂ = σ(a·logit p + b)", AQUA, "#e8f7f1"),
    (7.9, "Ожидаемый ущерб\nw = π̂ · L,\nL = min(A, C)", ORANGE, "#fdf0ea"),
    (9.85, "Топ-K по w\n→ аналитикам", ORANGE, "#fdf0ea"),
]
W, H, Y = 1.75, 1.75, 1.3
for i, (x, t, e, f) in enumerate(steps):
    box(ax, x, Y, W, H, t, edge=e, fill=f)
    if i:
        arrow(ax, (steps[i - 1][0] + W, Y + H / 2), (x, Y + H / 2), color=MUTED, shrink=2)
ax.text(2.05 + (4.0 + W - 2.05) / 2, 0.75, "Глава 3: оценка вероятности по графу", ha="center", color=BLUE, fontsize=10)
ax.text(5.95 + (9.85 + W - 5.95) / 2, 0.75, "Глава 2: экономически оптимальный отбор", ha="center", color=ORANGE, fontsize=10)
ax.plot([2.05, 4.0 + W], [1.05, 1.05], color=BLUE, lw=1.2)
ax.plot([5.95, 9.85 + W], [1.05, 1.05], color=ORANGE, lw=1.2)
ax.text(10.72, 0.3, "Метрика: доля предотвращённого ущерба D_K", ha="center", color=MUTED, fontsize=9)
fig.tight_layout()
fig.savefig(OUT + "fig_pipeline.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 3. слой GNN
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")
box(ax, 0.2, 3.7, 1.8, 0.9, "Отправитель s\nh_s", BLUE, "#eef4fc")
box(ax, 0.2, 2.15, 1.8, 0.9, "Перевод e\ng_e", ORANGE, "#fdf0ea")
box(ax, 0.2, 0.6, 1.8, 0.9, "Получатель d\nh_d", BLUE, "#eef4fc")
box(ax, 3.0, 3.55, 2.3, 1.15, "Сообщение вперёд\nm→ = φ→([h_s; g_e])", MUTED, "#f4f3f0", fs=9)
box(ax, 3.0, 0.5, 2.3, 1.15, "Сообщение назад\nm← = φ←([h_d; g_e])", MUTED, "#f4f3f0", fs=9)
box(ax, 6.2, 3.55, 2.2, 1.15, "Среднее по входящим\nу получателя: a_in", MUTED, "#f4f3f0", fs=9)
box(ax, 6.2, 0.5, 2.2, 1.15, "Среднее по исходящим\nу отправителя: a_out", MUTED, "#f4f3f0", fs=9)
box(ax, 9.0, 2.75, 1.85, 1.55, "Счёт:\nh ← LN(h + ψ([h;\na_in; a_out]))", BLUE, "#eef4fc", fs=9)
box(ax, 9.0, 0.6, 1.85, 1.55, "Перевод:\ng_e ← LN(g_e +\nχ([h_s; h_d; g_e]))", ORANGE, "#fdf0ea", fs=9)
arrow(ax, (2.0, 4.15), (3.0, 4.12), shrink=2); arrow(ax, (2.0, 2.6), (3.0, 3.9), shrink=2)
arrow(ax, (2.0, 1.05), (3.0, 1.08), shrink=2); arrow(ax, (2.0, 2.6), (3.0, 1.3), shrink=2)
arrow(ax, (5.3, 4.12), (6.2, 4.12), shrink=2); arrow(ax, (5.3, 1.08), (6.2, 1.08), shrink=2)
arrow(ax, (8.4, 4.12), (9.0, 3.8), shrink=2); arrow(ax, (8.4, 1.08), (9.0, 3.1), shrink=2)
arrow(ax, (9.92, 2.75), (9.92, 2.15), shrink=2)
ax.text(5.6, 2.62, "Разные сети для входящих и исходящих переводов\n(предложение 3.3); номера портов входят в g_e (предложение 3.4)",
        ha="center", va="center", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(OUT + "fig_gnn_layer.png", dpi=300, facecolor="white")
plt.close(fig)
print("ok")
