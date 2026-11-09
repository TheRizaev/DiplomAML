# -*- coding: utf-8 -*-
"""
Шаг 15. Данные для рисунков диплома (только numpy/pandas; сами графики строятся отдельно).

Файлы (в папке проекта):
  fig_curves.csv   — доля предотвращённого ущерба D_K при K = 10…20000: модели × правило (p, p·L) × набор + оракул
  fig_amounts.csv  — гистограмма log10(сумма, $) для честных и отмывочных переводов (HI-Small)
  fig_neff.csv     — эффективное число случаев n_eff при L = min(A, C) для C от 10^3 до 10^9 (оба набора)
  fig_hetero.csv   — распределение доли отмывочных соседей у отмывочных и честных переводов (оба набора)
Запуск: python 15_fig_data.py (около 5 минут).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEEDS = [0, 1, 2]
CAP = 1_000_000.0
KS = np.unique(np.round(np.logspace(1, np.log10(20000), 40)).astype(int))
COLS = ["Timestamp", "From_Bank", "From_Acc", "To_Bank", "To_Acc", "Amount_Received",
        "Receiving_Currency", "Amount_Paid", "Payment_Currency", "Payment_Format", "Is_Laundering"]


def find(stem):
    for d in (HERE, HERE / "data", Path.home() / "Downloads"):
        p = d / f"{stem}.csv"
        if p.exists():
            return p
    raise SystemExit(f"Не найден {stem}.csv")


def lg(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_platt(z, t, iters=50):
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = sig(a * z + b)
        g = np.array([np.sum((q - t) * z), np.sum(q - t)])
        ww = q * (1 - q)
        H = np.array([[np.sum(ww * z * z), np.sum(ww * z)], [np.sum(ww * z), np.sum(ww)]]) + 1e-9 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(H, g)
    return a, b


def load_scores(path, idx_ref):
    arr = np.load(path)
    idx, sc = arr[0].astype(np.int64), arr[1]
    if not np.array_equal(idx, idx_ref):
        sc = sc[pd.Series(np.arange(len(idx)), index=idx).reindex(idx_ref).values]
    return sc


def curve(score, L, yt):
    o = np.argsort(-score, kind="stable")
    cum = np.cumsum((L * yt)[o])
    return cum[np.minimum(KS, len(cum)) - 1] / (L * yt).sum()


curves, neff_rows, het_rows = [], [], []
for DATA in ("HI-Small", "LI-Small"):
    OUTD = HERE if DATA == "HI-Small" else HERE / DATA
    df = pd.read_csv(find(f"{DATA}_Trans"), header=0, names=COLS,
                     dtype={"From_Bank": str, "To_Bank": str, "From_Acc": str, "To_Acc": str})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    n = len(df)
    y = df["Is_Laundering"].values.astype(np.float64)
    rates = {"US Dollar": 1.0}
    m = (df["Receiving_Currency"] == "US Dollar") & (df["Payment_Currency"] != "US Dollar") & (df["Amount_Paid"] > 0)
    rates.update((df.loc[m, "Amount_Received"] / df.loc[m, "Amount_Paid"]).groupby(df.loc[m, "Payment_Currency"]).median().to_dict())
    m2 = (df["Payment_Currency"] == "US Dollar") & (df["Receiving_Currency"] != "US Dollar") & (df["Amount_Received"] > 0)
    for k, v in (df.loc[m2, "Amount_Paid"] / df.loc[m2, "Amount_Received"]).groupby(df.loc[m2, "Receiving_Currency"]).median().items():
        rates.setdefault(k, v)
    usd = (df["Amount_Paid"] * df["Payment_Currency"].map(rates)).values
    codes, _ = pd.factorize(pd.concat([df["From_Bank"] + "_" + df["From_Acc"], df["To_Bank"] + "_" + df["To_Acc"]]))
    S, D = codes[:n], codes[n:]
    N = int(codes.max()) + 1
    order = np.argsort(df["Timestamp"].values, kind="stable")
    tr, va, te = order[:int(n * .6)], order[int(n * .6):int(n * .8)], order[int(n * .8):]
    yt, A = y[te], usd[te]
    L = np.minimum(A, CAP)
    V = float((y[tr] == 0).sum() / y[tr].sum()) ** 0.5
    print(f"{DATA}: загружено, тест {len(te):,}", flush=True)

    # ---- кривые D_K
    curves.append(pd.DataFrame({"dataset": DATA, "model": "Оракул", "rule": "oracle", "K": KS,
                                "mean": curve(L * yt, L, yt), "std": 0.0}))
    models = {}
    z = np.load(OUTD / "scores_12_xgb.npz")
    models["XGBoost"] = [z[f"pc{s}"] for s in SEEDS]
    models["GNN-гибрид"] = [sig(lg(load_scores(OUTD / f"test_scores_10_gnnhybrid_s{s}.npy", te)) - np.log(V))
                            for s in SEEDS]
    if all((OUTD / f"val_scores_10_gnnhybrid_s{s}_long.npy").exists() for s in SEEDS):
        ens = []
        for s in SEEDS:
            pc = []
            for nm in ("gnnhybrid", "xgb"):
                pv = load_scores(OUTD / f"val_scores_10_{nm}_s{s}_long.npy", va)
                pt = load_scores(OUTD / f"test_scores_10_{nm}_s{s}_long.npy", te)
                a, b = fit_platt(lg(pv), y[va])
                pc.append(sig(a * lg(pt) + b))
            ens.append((pc[0] + pc[1]) / 2)
        models["Ансамбль"] = ens
    for name, runs in models.items():
        for rule in ("p", "pL"):
            vals = np.array([curve(p if rule == "p" else p * L, L, yt) for p in runs])
            curves.append(pd.DataFrame({"dataset": DATA, "model": name, "rule": rule, "K": KS,
                                        "mean": vals.mean(0), "std": vals.std(0)}))
    print(f"{DATA}: кривые готовы ({', '.join(models)})", flush=True)

    # ---- n_eff от потолка C
    for C in np.logspace(3, 9, 25):
        for part, Lp in (("тест", np.minimum(A[yt == 1], C)), ("весь набор", np.minimum(usd[y == 1], C))):
            neff_rows.append({"dataset": DATA, "part": part, "C": C, "n_eff": Lp.sum() ** 2 / (Lp ** 2).sum()})

    # ---- гетерофилия: доля отмывочных соседей у ребра (соседи — переводы с общим счётом)
    deg = np.bincount(S, minlength=N) + np.bincount(D, minlength=N)
    lab = np.bincount(S, weights=y, minlength=N) + np.bincount(D, weights=y, minlength=N)
    nb = deg[S] + deg[D] - 2
    share = np.divide(lab[S] + lab[D] - 2 * y, nb, out=np.full(n, np.nan), where=nb > 0)
    bins = np.linspace(0, 1, 51)
    for cls in (1, 0):
        s_ = share[(y == cls) & ~np.isnan(share)]
        h, _ = np.histogram(s_, bins=bins)
        for lo, c in zip(bins[:-1], h):
            het_rows.append({"dataset": DATA, "cls": cls, "bin_lo": lo, "frac": c / len(s_)})
        print(f"{DATA}: класс {cls}: средняя доля отмывочных соседей {s_.mean():.4f}", flush=True)

    if DATA == "HI-Small":
        b = np.arange(0, 10.01, 0.1)
        la = np.log10(np.maximum(usd, 1e-2))
        rows = []
        for cls in (0, 1):
            h, _ = np.histogram(la[y == cls], bins=b)
            for lo, c in zip(b[:-1], h):
                rows.append({"cls": cls, "bin_lo": lo, "frac": c / (y == cls).sum()})
        pd.DataFrame(rows).to_csv(HERE / "fig_amounts.csv", index=False)

pd.concat(curves).to_csv(HERE / "fig_curves.csv", index=False)
pd.DataFrame(neff_rows).to_csv(HERE / "fig_neff.csv", index=False)
pd.DataFrame(het_rows).to_csv(HERE / "fig_hetero.csv", index=False)
print("Готово: fig_curves.csv, fig_amounts.csv, fig_neff.csv, fig_hetero.csv", flush=True)
