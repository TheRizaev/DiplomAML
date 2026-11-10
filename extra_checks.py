import argparse
import os
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--export", action="store_true", help="только собрать meta_16.npz")
ap.add_argument("--boot", type=int, default=2000, help="число бутстреп-повторов")
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
DATA = os.environ.get("AML_DATA", "HI-Small")
OUTD = HERE if DATA == "HI-Small" else HERE / DATA
OUTD.mkdir(exist_ok=True)
META = OUTD / "meta_16.npz"
OUT = OUTD / "results_16_extra.txt"
SEEDS = [0, 1, 2]
CAP = 1_000_000.0
KS = [100, 500, 1000, 2000]
lines = []
t0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    OUT.write_text("\n".join(lines), encoding="utf-8")


def export_meta():
    import pandas as pd

    def find(stem):
        for d in (HERE, HERE / "data", Path.home() / "Downloads"):
            for e in (".csv", ".zip"):
                p = d / f"{stem}{e}"
                if p.exists():
                    return p
        raise SystemExit(f"Не найден {stem}.csv")

    cols = ["Timestamp", "From_Bank", "From_Acc", "To_Bank", "To_Acc", "Amount_Received",
            "Receiving_Currency", "Amount_Paid", "Payment_Currency", "Payment_Format", "Is_Laundering"]
    df = pd.read_csv(find(f"{DATA}_Trans"), header=0, names=cols,
                     dtype={"From_Bank": str, "To_Bank": str, "From_Acc": str, "To_Acc": str})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    n = len(df)
    days = df["Timestamp"].dt.floor("D").value_counts().sort_index()
    small = days < 0.01 * days.median()
    start = next((d for i, d in enumerate(days.index) if small.iloc[i:].all()), days.index[-1] + pd.Timedelta(days=1))
    y = df["Is_Laundering"].values.astype(np.int8)

    rates = {"US Dollar": 1.0}
    m = (df["Receiving_Currency"] == "US Dollar") & (df["Payment_Currency"] != "US Dollar") & (df["Amount_Paid"] > 0)
    rates.update((df.loc[m, "Amount_Received"] / df.loc[m, "Amount_Paid"]).groupby(df.loc[m, "Payment_Currency"]).median().to_dict())
    m2 = (df["Payment_Currency"] == "US Dollar") & (df["Receiving_Currency"] != "US Dollar") & (df["Amount_Received"] > 0)
    for k, v in (df.loc[m2, "Amount_Paid"] / df.loc[m2, "Amount_Received"]).groupby(df.loc[m2, "Receiving_Currency"]).median().items():
        rates.setdefault(k, v)
    usd = (df["Amount_Paid"] * df["Payment_Currency"].map(rates)).values
    usd_r = (df["Amount_Received"] * df["Receiving_Currency"].map(rates)).values

    tmin = df["Timestamp"].values.astype("datetime64[m]").astype(np.int64)
    order = np.argsort(df["Timestamp"].values, kind="stable")
    tr, va, te = order[:int(n * .6)], order[int(n * .6):int(n * .8)], order[int(n * .8):]

    codes, _ = pd.factorize(pd.concat([df["From_Bank"] + "_" + df["From_Acc"], df["To_Bank"] + "_" + df["To_Acc"]]))
    S, D = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    banks, _ = pd.factorize(pd.concat([df["From_Bank"], df["To_Bank"]]))
    cur_p = pd.factorize(df["Payment_Currency"])[0]
    cur_r = pd.factorize(df["Receiving_Currency"])[0]
    e = pd.DataFrame({"s": S, "d": D, "t": tmin})
    first = e.groupby(["s", "d"], sort=False)["t"].min().reset_index()
    first["out_port"] = first.sort_values("t").groupby("s").cumcount()
    first["in_port"] = first.sort_values("t").groupby("d").cumcount()
    e = e.merge(first[["s", "d", "out_port", "in_port"]], on=["s", "d"], how="left")
    num = np.stack([
        np.log1p(usd), np.log1p(usd_r),
        (cur_p != cur_r).astype(np.float64), (S == D).astype(np.float64),
        (banks[:n] == banks[n:]).astype(np.float64),
        df["Timestamp"].dt.hour.values / 23.0,
        np.log1p(e["out_port"].values), np.log1p(e["in_port"].values),
    ], axis=1)

    y_tr = y[tr]
    np.savez_compressed(
        META, n=n, te=te, va=va, y_te=y[te], y_va=y[va], A_te=usd[te], A_va=usd[va],
        t_te=tmin[te], t_va=tmin[va],
        tail_start=np.datetime64(start).astype("datetime64[m]").astype(np.int64),
        V=float((y_tr == 0).sum() / y_tr.sum()) ** 0.5, n_pos_tr=int(y_tr.sum()),
        mu_all=num.mean(0), sd_all=num.std(0), mu_tr=num[tr].mean(0), sd_tr=num[tr].std(0),
        num_names=np.array(["log сумма отпр.", "log сумма получ.", "разные валюты", "самому себе",
                            "один банк", "час", "log вых. порт", "log вх. порт"]))


if not META.exists() or args.export:
    export_meta()
if args.export:
    raise SystemExit(0)

M = np.load(META, allow_pickle=False)
te, va = M["te"], M["va"]
y_te, y_va = M["y_te"].astype(float), M["y_va"].astype(float)
A = M["A_te"]
L = np.minimum(A, CAP)
V = float(M["V"])
day = M["t_te"] // 1440
tail_day = int(M["tail_start"]) // 1440
n_te = len(te)
TOT = float((L * y_te).sum())
log(f"Набор {DATA}. Тест: {n_te:,} переводов, отмывочных {int(y_te.sum())}; вес класса v = {V:.1f}")


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


def fit_isotonic(p, t):
    o = np.argsort(p, kind="stable")
    xs, ys = p[o], t[o]
    val, wt, right = [], [], []
    for x, v in zip(xs, ys):
        val.append(float(v)); wt.append(1.0); right.append(x)
        while len(val) > 1 and val[-2] >= val[-1]:
            w = wt[-2] + wt[-1]
            val[-2] = (val[-2] * wt[-2] + val[-1] * wt[-1]) / w
            wt[-2] = w; right[-2] = right[-1]
            val.pop(); wt.pop(); right.pop()
    right, val = np.array(right), np.array(val)

    def f(q):
        return val[np.minimum(np.searchsorted(right, q, side="left"), len(val) - 1)]
    return f


def load(path, idx):
    arr = np.load(path)
    i, sc = arr[0].astype(np.int64), arr[1]
    if not np.array_equal(i, idx):
        pos = np.full(int(max(i.max(), idx.max())) + 1, -1, dtype=np.int64)
        pos[i] = np.arange(len(i))
        sc = sc[pos[idx]]
    return sc


def topk(score, K):
    K = min(K, len(score))
    return np.argpartition(-score, K - 1)[:K]


def capture(score, K):
    t = topk(score, K)
    return float((L[t] * y_te[t]).sum() / TOT)


def oracle(K):
    v = np.sort(L[y_te == 1])[::-1]
    return float(v[:K].sum() / v.sum())


def ms(a):
    a = np.asarray(a, dtype=float) * 100
    return f"{a.mean():5.1f} ± {a.std():4.1f}" if len(a) > 1 else f"{a.mean():5.1f}       "


models = {}
z = OUTD / "scores_12_xgb.npz"
if z.exists():
    zz = np.load(z)
    models["XGBoost"] = [zz[f"pc{s}"] for s in SEEDS if f"pc{s}" in zz]
hyb = [sig(lg(load(p, te)) - np.log(V)) for p in (OUTD / f"test_scores_10_gnnhybrid_s{s}.npy" for s in SEEDS) if p.exists()]
if hyb:
    models["GNN-гибрид"] = hyb
log("Модели: " + ", ".join(f"{k} ({len(v)} сида)" for k, v in models.items())
    + ". XGBoost — калибровка Платта, гибрид — поправка logit p − log v.")

log("\n=== Б1. Правила отбора без модели и с моделью: доля предотвращённого ущерба, %, L = min(A, 1 млн) ===")


def by_amount_expected(K):
    vals, inv, cnt = np.unique(-L, return_inverse=True, return_counts=True)
    gain = np.bincount(inv, weights=L * y_te, minlength=len(vals))
    cum = np.cumsum(cnt)
    full = np.searchsorted(cum, K, side="right")
    got = gain[:full].sum()
    if full < len(vals):
        left = K - (cum[full - 1] if full > 0 else 0)
        got += gain[full] * left / cnt[full]
    return got / TOT


hdr = f"  {'правило':<34}" + "".join(f"{'K=' + str(K):>14}" for K in KS)
log(hdr)
log(f"  {'случайный отбор':<34}" + "".join(f"{K / n_te * 100:>14.2f}" for K in KS))
log(f"  {'по сумме (L, ожидание)':<34}" + "".join(f"{by_amount_expected(K) * 100:>14.2f}" for K in KS))
log(f"  {'по сумме без потолка (A)':<34}" + "".join(f"{capture(A, K) * 100:>14.2f}" for K in KS))
for name, runs in models.items():
    log(f"  {name + ': по pi^':<34}" + "".join(f"{ms([capture(p, K) for p in runs]):>14}" for K in KS))
    log(f"  {name + ': по pi^·L':<34}" + "".join(f"{ms([capture(p * L, K) for p in runs]):>14}" for K in KS))
log(f"  {'оракул':<34}" + "".join(f"{oracle(K) * 100:>14.1f}" for K in KS))
nt = day < tail_day
TOT_NT = float((L[nt] * y_te[nt]).sum())


def capture_nt(score, K):
    idx = np.flatnonzero(nt)
    t = idx[topk(score[idx], K)]
    return float((L[t] * y_te[t]).sum() / TOT_NT)


log(f"  без хвоста генератора ({int(nt.sum()):,} переводов, отмывочных {int(y_te[nt].sum())}):")
log(f"  {'по сумме без потолка (A)':<34}" + "".join(f"{capture_nt(A, K) * 100:>14.2f}" for K in KS))
for name, runs in models.items():
    log(f"  {name + ': по pi^':<34}" + "".join(f"{ms([capture_nt(p, K) for p in runs]):>14}" for K in KS))
    log(f"  {name + ': по pi^·L':<34}" + "".join(f"{ms([capture_nt(p * L, K) for p in runs]):>14}" for K in KS))
big = A >= CAP
log(f"  переводов от 1 млн $ в тесте: {int(big.sum()):,}, из них отмывочных {int(y_te[big].sum())} "
    f"({y_te[big].mean() * 100:.2f}%); отмывочных всего {int(y_te.sum())}")

log(f"\n=== Б2. Бутстреп по тестовой выборке ({args.boot} повторов, пуассоновские веса), отбор по pi^·L ===")
log("Статистика — среднее D_K по сидам; интервал — 2,5 и 97,5 перцентили. Разность считается на одних и тех же")
log("повторах (парное сравнение). Переводы считаются независимыми, поэтому интервалы — оценка снизу.")
MCAND = 8000
rng = np.random.default_rng(0)
pos_idx = np.flatnonzero(y_te == 1)
cand = {}
for name, runs in models.items():
    cand[name] = []
    for p in runs:
        sc = p * L
        c = np.argpartition(-sc, MCAND - 1)[:MCAND]
        cand[name].append(c[np.argsort(-sc[c], kind="stable")])
univ = np.unique(np.concatenate([pos_idx] + [c for v in cand.values() for c in v]))
where = np.full(n_te, -1, dtype=np.int64)
where[univ] = np.arange(len(univ))
G = rng.poisson(1.0, size=(args.boot, len(univ))).astype(np.float32)
den = G[:, where[pos_idx]] @ L[pos_idx].astype(np.float32)
boot = {}
for name in models:
    res = np.zeros((len(cand[name]), len(KS), args.boot))
    for si, c in enumerate(cand[name]):
        w = G[:, where[c]]
        cw = np.cumsum(w, axis=1)
        prev = cw - w
        gain = (L[c] * y_te[c]).astype(np.float32)
        for ki, K in enumerate(KS):
            take = np.clip(K - prev, 0, w)
            res[si, ki] = (take @ gain) / den
    boot[name] = res.mean(0)
log(f"  {'модель':<14}" + "".join(f"{'K=' + str(K):>24}" for K in KS))
for name in models:
    row = f"  {name:<14}"
    for ki in range(len(KS)):
        b = boot[name][ki] * 100
        row += f"{b.mean():>9.1f} [{np.percentile(b, 2.5):5.1f}; {np.percentile(b, 97.5):5.1f}]"
    log(row)
if len(models) == 2:
    a, b = list(models)
    row = f"  {'гибрид − XGB':<14}"
    row2 = f"  {'доля повторов, где гибрид лучше':<14}"
    for ki in range(len(KS)):
        d = (boot["GNN-гибрид"][ki] - boot["XGBoost"][ki]) * 100
        row += f"{d.mean():>+9.1f} [{np.percentile(d, 2.5):+5.1f}; {np.percentile(d, 97.5):+5.1f}]"
        row2 += f"  K={KS[ki]}: {(d > 0).mean() * 100:.1f}%"
    log(row)
    log(row2)

log("\n=== Б3. Бюджет в день: k проверок в каждые сутки теста (а не K на весь тест) ===")
ud, dinv, dcnt = np.unique(day, return_inverse=True, return_counts=True)
dpos = np.bincount(dinv, weights=y_te, minlength=len(ud))
dloss = np.bincount(dinv, weights=y_te * L, minlength=len(ud))
log("  сутки теста: дата, переводов, отмывочных, доля ущерба теста, %")
for i, d in enumerate(ud):
    date = str(np.datetime64(int(d), "D"))
    log(f"    {date}{' (хвост)' if d >= tail_day else '':<9} {dcnt[i]:>9,} {int(dpos[i]):>6} {dloss[i] / TOT * 100:>8.1f}")


def per_day(score, k):
    o = np.lexsort((-score, dinv))
    start = np.concatenate([[0], np.cumsum(dcnt)[:-1]])
    rank = np.arange(n_te) - np.repeat(start, dcnt)
    sel = o[rank < k]
    return float((L[sel] * y_te[sel]).sum() / TOT), len(sel)


KD = [10, 25, 50, 100, 250]
log(f"  {'правило':<34}" + "".join(f"{'k=' + str(k):>14}" for k in KD))
log(f"  {'всего проверок за тест':<34}" + "".join(f"{per_day(L, k)[1]:>14}" for k in KD))
log(f"  {'по сумме без потолка (A)':<34}" + "".join(f"{per_day(A, k)[0] * 100:>14.1f}" for k in KD))
for name, runs in models.items():
    log(f"  {name + ': по pi^':<34}" + "".join(f"{ms([per_day(p, k)[0] for p in runs]):>14}" for k in KD))
    log(f"  {name + ': по pi^·L':<34}" + "".join(f"{ms([per_day(p * L, k)[0] for p in runs]):>14}" for k in KD))
log(f"  {'оракул (в пределах суток)':<34}" + "".join(f"{per_day(y_te * L, k)[0] * 100:>14.1f}" for k in KD))
log("  для сравнения: тот же общий бюджет одним списком на весь тест (отбор по pi^·L)")
for name, runs in models.items():
    log(f"  {name + ': общий список':<34}"
        + "".join(f"{ms([capture(p * L, per_day(L, k)[1]) for p in runs]):>14}" for k in KD))

ntd = ud < tail_day
if ntd.sum() < len(ud):
    log(f"  то же без хвоста: только {int(ntd.sum())} суток до начала хвоста, доля от ущерба этих суток")


    def per_day_nt(score, k):
        o = np.lexsort((-score, dinv))
        start = np.concatenate([[0], np.cumsum(dcnt)[:-1]])
        rank = np.arange(n_te) - np.repeat(start, dcnt)
        sel = o[(rank < k) & nt[o]]
        return float((L[sel] * y_te[sel]).sum() / TOT_NT), len(sel)


    log(f"  {'всего проверок':<34}" + "".join(f"{per_day_nt(L, k)[1]:>14}" for k in KD))
    log(f"  {'по сумме без потолка (A)':<34}" + "".join(f"{per_day_nt(A, k)[0] * 100:>14.1f}" for k in KD))
    for name, runs in models.items():
        log(f"  {name + ': по pi^':<34}" + "".join(f"{ms([per_day_nt(p, k)[0] for p in runs]):>14}" for k in KD))
        log(f"  {name + ': по pi^·L':<34}" + "".join(f"{ms([per_day_nt(p * L, k)[0] for p in runs]):>14}" for k in KD))
    log(f"  {'оракул (в пределах суток)':<34}" + "".join(f"{per_day_nt(y_te * L, k)[0] * 100:>14.1f}" for k in KD))

log("\n=== Б4. Способы калибровки (модели, для которых сохранены прогнозы на валидации) ===")
log("Доля предотвращённого ущерба при отборе по pi^·L, %; сумма pi^ на тесте (факт: "
    f"{int(y_te.sum())}); логарифмические потери на тесте ×1000.")


def logloss(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y_te * np.log(p) + (1 - y_te) * np.log(1 - p)).mean() * 1000)


for name, pat in (("XGBoost", "{}_scores_10_xgb_s{}_long.npy"), ("GNN-гибрид (до 1500 эпох)", "{}_scores_10_gnnhybrid_s{}_long.npy")):
    pairs = [(OUTD / pat.format("val", s), OUTD / pat.format("test", s)) for s in SEEDS]
    pairs = [(a, b) for a, b in pairs if a.exists() and b.exists()]
    if not pairs:
        log(f"  {name}: прогнозов на валидации нет — пропуск")
        continue
    variants = {"без калибровки": [], "Платт": [], "изотонная": [], "поправка −log v": []}
    for pv_path, pt_path in pairs:
        pv, pt = load(pv_path, va), load(pt_path, te)
        a, b = fit_platt(lg(pv), y_va)
        iso = fit_isotonic(pv, y_va)
        variants["без калибровки"].append(pt)
        variants["Платт"].append(sig(a * lg(pt) + b))
        variants["изотонная"].append(iso(pt))
        variants["поправка −log v"].append(sig(lg(pt) - np.log(V)))
    log(f"  {name} ({len(pairs)} сида)")
    log(f"    {'калибровка':<20}" + "".join(f"{'K=' + str(K):>14}" for K in KS) + f"{'сумма pi^':>12}{'лог. потери':>13}")
    for vn, runs in variants.items():
        log(f"    {vn:<20}" + "".join(f"{ms([capture(p * L, K) for p in runs]):>14}" for K in KS)
            + f"{np.mean([p.sum() for p in runs]):>12.0f}{np.mean([logloss(p) for p in runs]):>13.2f}")

log("\n=== Б5. Стандартизация признаков перевода: обучающая часть против всех данных ===")
log(f"  {'признак':<20}{'среднее (все)':>15}{'среднее (обуч.)':>17}{'откл. (все)':>13}{'откл. (обуч.)':>15}{'сдвиг, откл.':>14}")
for i, nm in enumerate(M["num_names"]):
    sh = (M["mu_tr"][i] - M["mu_all"][i]) / (M["sd_all"][i] + 1e-12)
    log(f"  {str(nm):<20}{M['mu_all'][i]:>15.4f}{M['mu_tr'][i]:>17.4f}{M['sd_all'][i]:>13.4f}{M['sd_tr'][i]:>15.4f}{sh:>14.4f}")
log(f"  наибольший сдвиг среднего: {np.max(np.abs((M['mu_tr'] - M['mu_all']) / (M['sd_all'] + 1e-12))):.4f} станд. откл.; "
    f"наибольшее отличие станд. откл.: {np.max(np.abs(M['sd_tr'] / (M['sd_all'] + 1e-12) - 1)) * 100:.2f}%")

log(f"\nГотово за {time.time() - t0:.0f} с.")
