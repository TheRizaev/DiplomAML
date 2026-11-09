import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
OUT = HERE / "results_02.txt"
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)


def find(stem, exts):
    for d in (HERE, HERE / "data", DOWNLOADS):
        for e in exts:
            p = d / f"{stem}{e}"
            if p.exists():
                return p
    return None


t0 = time.time()
COLS = ["Timestamp", "From_Bank", "From_Acc", "To_Bank", "To_Acc", "Amount_Received",
        "Receiving_Currency", "Amount_Paid", "Payment_Currency", "Payment_Format", "Is_Laundering"]
df = pd.read_csv(find("HI-Small_Trans", [".csv", ".zip"]), header=0, names=COLS,
                 dtype={"From_Bank": str, "To_Bank": str, "From_Acc": str, "To_Acc": str})
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
df["src"] = df["From_Bank"] + "_" + df["From_Acc"]
df["dst"] = df["To_Bank"] + "_" + df["To_Acc"]
y = df["Is_Laundering"].values == 1
n = len(df)

rates = {"US Dollar": 1.0}
m = (df["Receiving_Currency"] == "US Dollar") & (df["Payment_Currency"] != "US Dollar") & (df["Amount_Paid"] > 0)
rates.update((df.loc[m, "Amount_Received"] / df.loc[m, "Amount_Paid"]).groupby(df.loc[m, "Payment_Currency"]).median().to_dict())
m2 = (df["Payment_Currency"] == "US Dollar") & (df["Receiving_Currency"] != "US Dollar") & (df["Amount_Received"] > 0)
for k, v in (df.loc[m2, "Amount_Paid"] / df.loc[m2, "Amount_Received"]).groupby(df.loc[m2, "Receiving_Currency"]).median().items():
    rates.setdefault(k, v)
df["usd"] = df["Amount_Paid"] * df["Payment_Currency"].map(rates)

log("=== 0. Стабильность курсов (разброс Amount_Received/Amount_Paid к USD) ===")
r = (df.loc[m, "Amount_Received"] / df.loc[m, "Amount_Paid"])
st = r.groupby(df.loc[m, "Payment_Currency"]).agg(["count", "min", "median", "max"])
for k, row in st.iterrows():
    log(f"  {k:<18} n={int(row['count']):>6}  min {row['min']:.6g}  med {row['median']:.6g}  max {row['max']:.6g}")

log("\n=== 1. Хвост в исходной валюте (без конвертации) ===")
for name, mask in [("честные", ~y), ("отмывание", y)]:
    s = df.loc[mask, "Amount_Paid"]
    log(f"  {name:<10} медиана {s.median():>12,.0f}  p99 {s.quantile(.99):>16,.0f}  max {s.max():>20,.0f}")

log("\n=== 2. Топ-30 отмывочных переводов по сумме в USD ===")
L = df[y].sort_values("usd", ascending=False)
top = L.head(30)
log(f"  {'дата':<17}{'сумма USD':>18}  {'валюта':<16}{'формат':<10}{'себе?':<6}{'src':<20}{'dst':<20}")
for _, rr in top.iterrows():
    log(f"  {str(rr['Timestamp']):<17}{rr['usd']:>18,.0f}  {rr['Payment_Currency']:<16}{rr['Payment_Format']:<10}"
        f"{'да' if rr['src'] == rr['dst'] else 'нет':<6}{rr['src']:<20}{rr['dst']:<20}")

k = max(1, len(L) // 100)
T = L.head(k)
log(f"\n  Топ-1% ({k} шт.) по валютам:  {T['Payment_Currency'].value_counts().to_dict()}")
log(f"  Топ-1% по форматам:          {T['Payment_Format'].value_counts().to_dict()}")
log(f"  Топ-1% переводы самому себе: {(T['src'] == T['dst']).mean():.2f}")
log(f"  Уникальных счетов-отправителей в топ-1%: {T['src'].nunique()}, получателей: {T['dst'].nunique()}")

amt_rep = L["usd"].round(-2).value_counts()
log(f"  Отмывочных сумм, повторяющихся (с точностью до 100 USD) 2+ раз: {(amt_rep > 1).sum()}  "
    f"(в них {amt_rep[amt_rep > 1].sum()} транзакций)")

log("\n=== 3. Сравнение хвостов ===")
for name, mask in [("честные", ~y), ("отмывание", y)]:
    s = df.loc[mask, "usd"].sort_values(ascending=False)
    k1 = max(1, len(s) // 100)
    log(f"  {name:<10} доля денег у топ-1%: {s.head(k1).sum() / s.sum():.3f};  "
        f"переводов > 1 млн USD: {(s > 1e6).sum():,} ({(s > 1e6).mean():.4f})")

order = np.argsort(df["Timestamp"].values, kind="stable")
split = np.empty(n, dtype="<U5")
split[order[:int(n * .6)]] = "train"
split[order[int(n * .6):int(n * .8)]] = "val"
split[order[int(n * .8):]] = "test"
df["split"] = split
log("\n=== 4. Отмывочные деньги по частям выборки ===")
g = df[y].groupby("split")["usd"].agg(["count", "sum", "median", "max"]).loc[["train", "val", "test"]]
tot = g["sum"].sum()
for k_, rr in g.iterrows():
    log(f"  {k_:<6} n={int(rr['count']):>5}  сумма {rr['sum']:>18,.0f} ({rr['sum'] / tot:.1%})  "
        f"медиана {rr['median']:>10,.0f}  max {rr['max']:>16,.0f}")

log("\n=== 5. Кандидаты для потерь L_i (только отмывочные переводы) ===")
a = df.loc[y, "usd"].values
cap = np.quantile(df.loc[~y, "usd"].values, 0.99)
cands = {
    "L = сумма": a,
    "L = sqrt(сумма)": np.sqrt(a),
    "L = log(1+сумма)": np.log1p(a),
    f"L = min(сумма, p99 честных = {cap:,.0f})": np.minimum(a, cap),
    "L = 1 (просто число случаев)": np.ones_like(a),
}


def gini(x):
    x = np.sort(x)
    nn = len(x)
    return float((2 * np.arange(1, nn + 1) - nn - 1).dot(x) / (nn * x.sum()))


log(f"  {'вариант':<44}{'доля топ-1%':>12}{'доля топ-10%':>14}{'Джини':>8}{'эфф. число':>12}")
for name, v in cands.items():
    s = np.sort(v)[::-1]
    k1, k10 = max(1, len(s) // 100), max(1, len(s) // 10)
    w = s / s.sum()
    neff = 1 / np.sum(w ** 2)
    log(f"  {name:<44}{s[:k1].sum() / s.sum():>12.3f}{s[:k10].sum() / s.sum():>14.3f}{gini(v):>8.3f}{neff:>12.0f}")
log("  (эфф. число = 1/Σw²: сколько транзакций реально определяют метрику; при L=1 это все отмывочные транзакции)")

log("\n=== 6. Чувствительность метрики «доля пойманных денег» ===")
s = np.sort(a)[::-1]
for miss in (1, 5, 10, 50):
    log(f"  Если модель ловит ВСЁ, кроме {miss:>2} самых крупных отмывочных переводов: доля денег = {1 - s[:miss].sum() / s.sum():.3f}")
log(f"  Если модель ловит ТОЛЬКО 10 самых крупных (0,2% случаев): доля денег = {s[:10].sum() / s.sum():.3f}")

log(f"\nГотово за {time.time() - t0:.0f} с.")
OUT.write_text("\n".join(lines), encoding="utf-8")
