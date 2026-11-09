# -*- coding: utf-8 -*-
"""Графики по данным шага 15: кривые D_K, распределение сумм, n_eff от C, гетерофилия. PNG 300 dpi."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#b9b7b0",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, GRID, ORACLE = "#0b0b0b", "#52514e", "#e6e4de", "#9a988f"
COL = {"XGBoost": "#2a78d6", "GNN-гибрид": "#eb6834", "Ансамбль": "#1baf7a"}
from pathlib import Path
SRC = str(Path(__file__).resolve().parent.parent) + "/"   # CSV пишет 15_fig_data.py в корень проекта
OUT = str(Path(__file__).resolve().parent) + "/"


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


pct = mt.FuncFormatter(lambda v, _: f"{v:.0f}%")

# ---------------------------------------------------------------- 1. D_K от K
cur = pd.read_csv(SRC + "fig_curves.csv")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, ds in zip(axes, ["HI-Small", "LI-Small"]):
    c = cur[cur.dataset == ds]
    o = c[c.rule == "oracle"]
    ax.plot(o.K, o["mean"] * 100, color=ORACLE, lw=2, ls=(0, (1, 1.5)))
    ax.annotate("Оракул", (o.K.iloc[8], o["mean"].iloc[8] * 100), xytext=(-4, 6), textcoords="offset points",
                color=MUTED, fontsize=9, ha="right")
    for m, col in COL.items():
        for rule, ls in (("pL", "-"), ("p", (0, (5, 2.5)))):
            s = c[(c.model == m) & (c.rule == rule)]
            if not len(s):
                continue
            ax.plot(s.K, s["mean"] * 100, color=col, lw=2, ls=ls,
                    label=f"{m}, отбор по {'π̂·L' if rule == 'pL' else 'π̂'}")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", " ")))
    ax.set_xlim(10, 20000); ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(pct)
    ax.set_title(ds, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Бюджет проверок K (логарифмическая шкала)")
    style(ax)
axes[0].set_ylabel("Доля предотвращённого ущерба D_K")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(OUT + "fig_dk_curves.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 2. суммы и n_eff
am = pd.read_csv(SRC + "fig_amounts.csv")
ne = pd.read_csv(SRC + "fig_neff.csv")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
for cls, col, lab in ((0, "#2a78d6", "Честные переводы"), (1, "#eb6834", "Отмывочные переводы")):
    s = am[am.cls == cls]
    ax.step(s.bin_lo + 0.05, s.frac * 100, where="mid", color=col, lw=1.8, label=lab)
ax.set_xlim(0, 10)
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"10$^{{{int(v)}}}$"))
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("Сумма перевода, $ (логарифмическая шкала)")
ax.set_ylabel("Доля переводов класса")
ax.set_title("Распределение сумм, HI-Small", color=INK, fontsize=11, loc="left")
ax.axvline(6, color=MUTED, lw=1, ls=(0, (3, 2)))
ax.text(6.1, ax.get_ylim()[1] * 0.92, "C = 1 млн $", color=MUTED, fontsize=9)
ax.legend(frameon=False, fontsize=9)
style(ax)
ax = axes[1]
for ds, col in (("HI-Small", "#2a78d6"), ("LI-Small", "#eb6834")):
    s = ne[(ne.dataset == ds) & (ne.part == "тест")]
    ax.plot(s.C, s.n_eff, color=col, lw=2, label=f"{ds}, тест")
    ax.annotate(f"{int(s.n_eff.iloc[-1])}", (s.C.iloc[-1], s.n_eff.iloc[-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=9, color=MUTED)
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvline(1e6, color=MUTED, lw=1, ls=(0, (3, 2)))
ax.set_xlabel("Потолок C, $ (L = min(A, C))")
ax.set_ylabel("Эффективное число случаев n_eff")
ax.set_title("Надёжность денежной метрики", color=INK, fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9)
style(ax)
fig.tight_layout()
fig.savefig(OUT + "fig_amounts_neff.png", dpi=300, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 3. гетерофилия
he = pd.read_csv(SRC + "fig_hetero.csv")
fig, ax = plt.subplots(figsize=(8, 4.2))
for ds, col in (("HI-Small", "#2a78d6"), ("LI-Small", "#eb6834")):
    s = he[(he.dataset == ds) & (he.cls == 1)]
    xs = np.append(s.bin_lo.values, 1.0); ys = np.append(s.frac.values * 100, s.frac.values[-1] * 100)
    ax.step(xs, ys, where="post", color=col, lw=1.9, label=f"{ds}")
ax.set_xlim(0, 0.6)
ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("Доля отмывочных переводов среди соседей")
ax.set_ylabel("Доля среди отмывочных переводов")
ax.text(0.98, 0.95, "Средняя доля: 10,8% (HI-Small), 2,6% (LI-Small)\nу честных переводов: 0,09% и 0,05%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=MUTED)
ax.legend(frameon=False, fontsize=9, loc="center right")
style(ax)
fig.tight_layout()
fig.savefig(OUT + "fig_heterophily.png", dpi=300, facecolor="white")
plt.close(fig)
print("ok")
