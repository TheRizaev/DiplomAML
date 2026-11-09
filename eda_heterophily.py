import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import os as _os
DATA = _os.environ.get("AML_DATA", "HI-Small")
OUTD = HERE if DATA == "HI-Small" else HERE / DATA
OUTD.mkdir(exist_ok=True)

DOWNLOADS = Path.home() / "Downloads"
OUT_TXT = OUTD / "results_01.txt"
OUT_JSON = OUTD / "results_01.json"

lines = []
res = {}


def log(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)


def find_file(stem, exts):
    for d in (HERE, HERE / "data", DOWNLOADS):
        for e in exts:
            p = d / f"{stem}{e}"
            if p.exists():
                return p
    return None


t0 = time.time()
trans_path = find_file(f"{DATA}_Trans", [".csv", ".csv.zip", ".zip"])
pat_path = find_file(f"{DATA}_Patterns", [".txt", ".txt.zip", ".zip"])
if trans_path is None:
    sys.exit("Не нашёл HI-Small_Trans.csv / .zip ни в папке скрипта, ни в «Загрузках».")
log("Файл транзакций:", trans_path)
log("Файл паттернов:", pat_path)

COLS = ["Timestamp", "From_Bank", "From_Acc", "To_Bank", "To_Acc",
        "Amount_Received", "Receiving_Currency", "Amount_Paid",
        "Payment_Currency", "Payment_Format", "Is_Laundering"]
df = pd.read_csv(trans_path, header=0, names=COLS,
                 dtype={"From_Bank": str, "To_Bank": str, "From_Acc": str, "To_Acc": str})
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
df["src"] = df["From_Bank"] + "_" + df["From_Acc"]
df["dst"] = df["To_Bank"] + "_" + df["To_Acc"]
y = df["Is_Laundering"].astype(np.int8).values
n, n_l = len(df), int(y.sum())

log("\n=== 1. Общие числа ===")
log(f"Транзакций: {n:,}")
log(f"Отмывочных: {n_l:,}  (доля {n_l / n:.5f}, т.е. 1 на {n / n_l:.0f})")
log(f"Период: {df['Timestamp'].min()} — {df['Timestamp'].max()}")
acc_full = pd.unique(np.concatenate([df["src"].values, df["dst"].values]))
acc_short = pd.unique(np.concatenate([df["From_Acc"].values, df["To_Acc"].values]))
log(f"Счетов (банк+номер): {len(acc_full):,};  только по номеру: {len(acc_short):,}")
log(f"Переводов самому себе: {(df['src'] == df['dst']).mean():.4f}")
log(f"Переводов между разными валютами: {(df['Payment_Currency'] != df['Receiving_Currency']).mean():.4f}")
res["general"] = dict(n=n, n_launder=n_l, rate=n_l / n, accounts=len(acc_full),
                      accounts_short=len(acc_short),
                      t_min=str(df["Timestamp"].min()), t_max=str(df["Timestamp"].max()))

daily = df.groupby(df["Timestamp"].dt.date)["Is_Laundering"].agg(["count", "sum"])
daily["rate"] = daily["sum"] / daily["count"]
log("\nПо дням (транзакций, отмывочных, доля):")
for d, r in daily.iterrows():
    log(f"  {d}: {int(r['count']):>9,}  {int(r['sum']):>5}  {r['rate']:.5f}")
res["daily"] = {str(k): [int(v["count"]), int(v["sum"])] for k, v in daily.iterrows()}

log("\n=== 2. Форматы платежей ===")
pf = df.groupby("Payment_Format")["Is_Laundering"].agg(["count", "sum", "mean"]).sort_values("mean", ascending=False)
for k, r in pf.iterrows():
    log(f"  {k:<14} всего {int(r['count']):>9,}  отмыв. {int(r['sum']):>5}  доля {r['mean']:.5f}")
res["payment_format"] = {k: [int(r["count"]), int(r["sum"])] for k, r in pf.iterrows()}

log("\nВалюты платежа (топ):")
cur = df["Payment_Currency"].value_counts()
for k, v in cur.items():
    log(f"  {k:<20} {v:>9,}")

rates = {"US Dollar": 1.0}
m = (df["Receiving_Currency"] == "US Dollar") & (df["Payment_Currency"] != "US Dollar") & (df["Amount_Paid"] > 0)
r1 = (df.loc[m, "Amount_Received"] / df.loc[m, "Amount_Paid"]).groupby(df.loc[m, "Payment_Currency"]).median()
rates.update(r1.to_dict())
m2 = (df["Payment_Currency"] == "US Dollar") & (df["Receiving_Currency"] != "US Dollar") & (df["Amount_Received"] > 0)
r2 = (df.loc[m2, "Amount_Paid"] / df.loc[m2, "Amount_Received"]).groupby(df.loc[m2, "Receiving_Currency"]).median()
for k, v in r2.items():
    rates.setdefault(k, v)
missing = sorted(set(cur.index) - set(rates))
log("\nКурс к USD (медиана по самим данным):")
for k, v in sorted(rates.items()):
    log(f"  {k:<20} {v:.6g}")
if missing:
    log("  Нет курса для:", missing)
res["usd_rates"] = {k: float(v) for k, v in rates.items()}

df["amount_usd"] = df["Amount_Paid"] * df["Payment_Currency"].map(rates)

log("\n=== 3. Суммы (в USD) ===")
a = df["amount_usd"]
for name, mask in [("честные", y == 0), ("отмывание", y == 1)]:
    s = a[mask]
    log(f"  {name:<10} медиана {s.median():>12,.0f}   среднее {s.mean():>14,.0f}   "
        f"90%-квантиль {s.quantile(.9):>14,.0f}")
la = a[y == 1].sort_values(ascending=False)
top10 = la.head(max(1, len(la) // 10)).sum() / la.sum()
top1 = la.head(max(1, len(la) // 100)).sum() / la.sum()
log(f"  Отмывание: топ-10% самых крупных переводов несут {top10:.1%} всех отмывочных денег, топ-1% — {top1:.1%}")
log(f"  Сумма всех отмывочных переводов (с повторами по цепочкам): {la.sum():,.0f} USD")
res["amounts"] = dict(median_licit=float(a[y == 0].median()), median_launder=float(a[y == 1].median()),
                      mean_licit=float(a[y == 0].mean()), mean_launder=float(a[y == 1].mean()),
                      launder_top10_share=float(top10), launder_top1_share=float(top1),
                      launder_total_usd=float(la.sum()))

log("\n=== 4. Временной сплит 60/20/20 (как у Altman et al.) ===")
order = np.argsort(df["Timestamp"].values, kind="stable")
cut1, cut2 = int(n * .6), int(n * .8)
split = np.empty(n, dtype="<U5")
split[order[:cut1]] = "train"
split[order[cut1:cut2]] = "val"
split[order[cut2:]] = "test"
df["split"] = split
sp = df.groupby("split")["Is_Laundering"].agg(["count", "sum", "mean"]).loc[["train", "val", "test"]]
for k, r in sp.iterrows():
    log(f"  {k:<6} {int(r['count']):>9,}  отмыв. {int(r['sum']):>5}  доля {r['mean']:.5f}")
res["split"] = {k: [int(r["count"]), int(r["sum"])] for k, r in sp.iterrows()}

log("\n=== 5. Гомофилия / гетерофилия ===")
src_codes, uniq = pd.factorize(pd.concat([df["src"], df["dst"]]), sort=False)
s_idx, d_idx = src_codes[:n], src_codes[n:]
n_acc = len(uniq)
deg = np.bincount(s_idx, minlength=n_acc) + np.bincount(d_idx, minlength=n_acc)
lab = np.bincount(s_idx, weights=y, minlength=n_acc) + np.bincount(d_idx, weights=y, minlength=n_acc)
nb = deg[s_idx] + deg[d_idx] - 2
nb_l = lab[s_idx] + lab[d_idx] - 2 * y
with np.errstate(divide="ignore", invalid="ignore"):
    frac = np.where(nb > 0, nb_l / nb, np.nan)
h_l = np.nanmean(frac[y == 1])
h_c = np.nanmean(frac[y == 0])
base = n_l / n
log(f"  Для отмывочной транзакции: доля отмывочных среди соседних транзакций = {h_l:.4f}")
log(f"  Для честной транзакции:    доля отмывочных среди соседних транзакций = {h_c:.6f}")
log(f"  Базовая доля отмывания (случайный граф): {base:.6f}  -> «подъём» для отмывания x{h_l / base:.0f}")
log(f"  Гомофилия класса «отмывание» на рёбрах (доля своих среди соседей): {h_l:.4f}  (1 = только свои, 0 = только чужие)")
fl = frac[y == 1]
fl = fl[~np.isnan(fl)]
for thr in (0.05, 0.1, 0.25, 0.5):
    log(f"    доля отмывочных транзакций, у которых < {thr:.0%} соседей — отмывание: {(fl < thr).mean():.3f}")
res["edge_homophily"] = dict(launder=float(h_l), licit_launder_frac=float(h_c), base=float(base),
                             lt05=float((fl < .05).mean()), lt10=float((fl < .1).mean()),
                             lt25=float((fl < .25).mean()), lt50=float((fl < .5).mean()))

acc_bad = lab > 0
log(f"\n  Счетов-участников отмывания: {acc_bad.sum():,} из {n_acc:,} ({acc_bad.mean():.4f})")
share_bad_tx = lab[acc_bad] / deg[acc_bad]
log(f"  У счёта-участника доля отмывочных среди ВСЕХ его транзакций: медиана {np.median(share_bad_tx):.3f}, "
    f"среднее {share_bad_tx.mean():.3f}")
log("  (маленькое число = отмыватель «прячется» среди обычных операций, это и есть маскировка)")
nbr_bad = (np.bincount(s_idx, weights=acc_bad[d_idx], minlength=n_acc)
           + np.bincount(d_idx, weights=acc_bad[s_idx], minlength=n_acc))
with np.errstate(divide="ignore", invalid="ignore"):
    node_h = nbr_bad / deg
hb = np.nanmean(node_h[acc_bad])
hg = np.nanmean(1 - node_h[~acc_bad])
log(f"  Гомофилия счетов-участников (доля таких же среди соседей): {hb:.4f}")
log(f"  Гомофилия обычных счетов (доля обычных среди соседей):     {hg:.4f}")
log("  Для сравнения AMLgentex (другая синтетика): 0.52 у отмывателей и 0.98 у обычных")
res["node_homophily"] = dict(n_bad=int(acc_bad.sum()), n_acc=int(n_acc),
                             share_bad_tx_median=float(np.median(share_bad_tx)),
                             share_bad_tx_mean=float(share_bad_tx.mean()),
                             h_bad=float(hb), h_good=float(hg))

if pat_path is not None:
    log("\n=== 6. Файл паттернов ===")
    import zipfile
    if pat_path.suffix == ".zip":
        with zipfile.ZipFile(pat_path) as z:
            txt = z.read(z.namelist()[0]).decode("utf-8", "ignore").splitlines()
    else:
        txt = pat_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    kinds, cur_kind, in_pat = {}, None, 0
    pat_keys = set()
    for ln in txt:
        ln = ln.strip()
        if ln.startswith("BEGIN LAUNDERING ATTEMPT"):
            mm = re.search(r"-\s*([A-Z\-]+)", ln)
            cur_kind = mm.group(1).strip("-") if mm else "?"
            kinds[cur_kind] = kinds.get(cur_kind, 0) + 1
        elif ln.startswith("END LAUNDERING ATTEMPT"):
            cur_kind = None
        elif cur_kind and ln:
            p = ln.split(",")
            if len(p) >= 6:
                try:
                    pat_keys.add((p[0], p[1] + "_" + p[2], p[3] + "_" + p[4], round(float(p[5]), 2)))
                except ValueError:
                    pass
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        log(f"  {k:<16} попыток: {v}")
    lt = df[y == 1]
    key = list(zip(lt["Timestamp"].dt.strftime("%Y/%m/%d %H:%M"), lt["src"], lt["dst"],
                   lt["Amount_Received"].round(2)))
    in_p = np.array([k in pat_keys for k in key])
    log(f"  Транзакций в файле паттернов: {len(pat_keys):,}")
    log(f"  Отмывочных транзакций, попавших в какой-то паттерн: {in_p.mean():.3f} "
        f"(остальные — «одиночные» или продолжение цепочек)")
    res["patterns"] = dict(attempts=kinds, n_pattern_tx=len(pat_keys), share_launder_in_pattern=float(in_p.mean()))

log(f"\nГотово за {time.time() - t0:.0f} с.")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
