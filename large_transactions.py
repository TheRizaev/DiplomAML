import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = os.environ.get("AML_DATA", "HI-Small")
OUTD = HERE if DATA == "HI-Small" else HERE / DATA
OUT = OUTD / "results_13_large.txt"
SEEDS = [0, 1, 2]
CAP = 1_000_000.0
BINS = [(0, 1e4, "< 10 тыс."), (1e4, 1e5, "10–100 тыс."), (1e5, 1e6, "100 тыс.–1 млн"), (1e6, np.inf, "≥ 1 млн")]
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    OUT.write_text("\n".join(lines), encoding="utf-8")


def find(stem):
    for d in (HERE, HERE / "data", Path.home() / "Downloads"):
        p = d / f"{stem}.csv"
        if p.exists():
            return p
    raise SystemExit(f"Не найден {stem}.csv")


COLS = ["Timestamp", "From_Bank", "From_Acc", "To_Bank", "To_Acc", "Amount_Received",
        "Receiving_Currency", "Amount_Paid", "Payment_Currency", "Payment_Format", "Is_Laundering"]
df = pd.read_csv(find(f"{DATA}_Trans"), header=0, names=COLS,
                 usecols=["Timestamp", "Amount_Received", "Receiving_Currency", "Amount_Paid",
                          "Payment_Currency", "Is_Laundering"])
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
n = len(df)
y = df["Is_Laundering"].values.astype(np.float32)
rates = {"US Dollar": 1.0}
m = (df["Receiving_Currency"] == "US Dollar") & (df["Payment_Currency"] != "US Dollar") & (df["Amount_Paid"] > 0)
rates.update((df.loc[m, "Amount_Received"] / df.loc[m, "Amount_Paid"]).groupby(df.loc[m, "Payment_Currency"]).median().to_dict())
m2 = (df["Payment_Currency"] == "US Dollar") & (df["Receiving_Currency"] != "US Dollar") & (df["Amount_Received"] > 0)
for k, v in (df.loc[m2, "Amount_Paid"] / df.loc[m2, "Amount_Received"]).groupby(df.loc[m2, "Receiving_Currency"]).median().items():
    rates.setdefault(k, v)
usd = (df["Amount_Paid"] * df["Payment_Currency"].map(rates)).values
order = np.argsort(df["Timestamp"].values, kind="stable")
tr, te = order[:int(n * .6)], order[int(n * .8):]
y_tr, y_te, A = y[tr], y[te], usd[te]
L = np.minimum(A, CAP)
V = float((y_tr == 0).sum() / y_tr.sum()) ** 0.5


def lg(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def pr_auc(yt, sc):
    if yt.sum() == 0:
        return float("nan")
    o = np.argsort(-sc, kind="stable")
    yy = yt[o]
    tp = np.cumsum(yy)
    return float(np.sum(yy * tp / np.arange(1, len(yy) + 1)) / yy.sum())


def topk(sc, K):
    return np.argpartition(-sc, K - 1)[:K]


models = {}
z = np.load(OUTD / "scores_12_xgb.npz")
models["XGBoost"] = [z[f"pc{s}"] for s in SEEDS]
hyb = []
for s in SEEDS:
    arr = np.load(OUTD / f"test_scores_10_gnnhybrid_s{s}.npy")
    idx, sc = arr[0].astype(np.int64), arr[1]
    if not np.array_equal(idx, te):
        sc = sc[pd.Series(np.arange(len(idx)), index=idx).reindex(te).values]
    hyb.append(sig(lg(sc) - np.log(V)))
models["GNN-гибрид"] = hyb

log(f"Набор {DATA}. Тест: {len(te):,} переводов, отмывочных {int(y_te.sum())}. Поправка гибрида: log v = {np.log(V):.3f}")
log("\n1) PR-AUC внутри диапазона сумм (сравнение качества ранжирования среди переводов одного масштаба), %")
log(f"  {'диапазон':<16}{'переводов':>12}{'отмыв.':>8}" + "".join(f"{k:>20}" for k in models))
for lo, hi, lab in BINS:
    mask = (A >= lo) & (A < hi)
    row = f"  {lab:<16}{int(mask.sum()):>12,}{int(y_te[mask].sum()):>8}"
    for k, runs in models.items():
        v = np.array([pr_auc(y_te[mask], p[mask]) for p in runs]) * 100
        row += f"{v.mean():>12.1f} ± {v.std():4.1f}"
    log(row)

for K in (100, 1000):
    log(f"\n2) K = {K}, отбор по pi·L: сколько отмывочных переводов каждого диапазона попало в отбор "
        f"(из всех отмывочных этого диапазона), среднее по сидам")
    log(f"  {'диапазон':<16}{'всего отмыв.':>14}" + "".join(f"{k:>16}" for k in models))
    for lo, hi, lab in BINS:
        mask = (A >= lo) & (A < hi)
        row = f"  {lab:<16}{int(y_te[mask].sum()):>14}"
        for k, runs in models.items():
            c = np.mean([((y_te[topk(p * L, K)] == 1) & mask[topk(p * L, K)]).sum() for p in runs])
            row += f"{c:>16.1f}"
        log(row)
    row = f"  {'точность отбора':<16}{'':>14}"
    for k, runs in models.items():
        row += f"{np.mean([y_te[topk(p * L, K)].mean() for p in runs]) * 100:>15.1f}%"
    log(row)

log("\nГипотеза подтверждается, если преимущество гибрида по PR-AUC растёт с суммой "
    "и при K = 100 он отбирает больше крупных отмывочных переводов.")
