# -*- coding: utf-8 -*-
"""
Шаг 12. Проверка утверждений главы 2, которые ещё не проверены экспериментом.

  Э1. Чувствительность к потолку C и форме функции потерь (п. 2.6.3, предложение 2.8):
      C в {1e4, 1e5, 1e6, 1e7}, L = sqrt(A), L = log(1+A). Для каждой L — отбор по pi и по pi*L,
      доля оракула, эффективное число n_eff.
  Э2. Восстановление вероятностей без валидации (предложения 2.6, 2.7; схема В из п. 2.5):
      GNN с весами по классу:  logit pi = logit p - log v;
      GNN с весами по сумме:   logit pi = logit p - log v - log u(L).
  Э3. Задача о рюкзаке (п. 2.3, теорема 2.2, следствие 2.5): стоимость проверки
      c_i = 1 + kappa * log(1 + d_i), d_i — число переводов отправителя и получателя в графе.
      Сравнение: отбор по pi, по pi*L, жадный по плотности pi*L/c; верхняя граница Данцига
      и апостериорный сертификат (U - G)/G.

Модели:
  XGBoost (обучается здесь, 3 сида, калибровка Платта на валидации) — как в шаге 10;
  GNN class / GNN amount (шаг 7), GNN-гибрид (шаг 10) — берутся сохранённые прогнозы на тесте
  test_scores_07_*.npy и test_scores_10_gnnhybrid_*.npy; если файла нет, модель пропускается.

Запуск: python 12_theory_checks.py.  Результат: results_12_theory.txt
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
import os as _os
DATA = _os.environ.get("AML_DATA", "HI-Small")          # HI-Small (по умолчанию) или LI-Small
OUTD = HERE if DATA == "HI-Small" else HERE / DATA     # результаты второго набора — в отдельной папке
OUTD.mkdir(exist_ok=True)

DOWNLOADS = Path.home() / "Downloads"
OUT = OUTD / "results_12_theory.txt"
SEEDS = [0, 1, 2]
CAP = 1_000_000.0
MAX_W = 10.0           # как в шагах 7 и 10
U_MIN = 0.01           # нижний порог u(L) в схеме В
KS = [100, 1000]
lines = []
t0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)
    OUT.write_text("\n".join(lines), encoding="utf-8")


def find(stem, exts):
    for d in (HERE, HERE / "data", DOWNLOADS):
        for e in exts:
            p = d / f"{stem}{e}"
            if p.exists():
                return p
    return None


# ------------------------------------------------------------------ данные (как в шаге 10)
COLS = ["Timestamp", "From_Bank", "From_Acc", "To_Bank", "To_Acc", "Amount_Received",
        "Receiving_Currency", "Amount_Paid", "Payment_Currency", "Payment_Format", "Is_Laundering"]
df = pd.read_csv(find(f"{DATA}_Trans", [".csv", ".zip"]), header=0, names=COLS,
                 dtype={"From_Bank": str, "To_Bank": str, "From_Acc": str, "To_Acc": str})
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
usd_r = (df["Amount_Received"] * df["Receiving_Currency"].map(rates)).values

codes, _ = pd.factorize(pd.concat([df["From_Bank"] + "_" + df["From_Acc"], df["To_Bank"] + "_" + df["To_Acc"]]))
S, D = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
N = int(codes.max()) + 1
banks, _ = pd.factorize(pd.concat([df["From_Bank"], df["To_Bank"]]))
cur_p = pd.factorize(df["Payment_Currency"])[0]
cur_r = pd.factorize(df["Receiving_Currency"])[0]
fmt = pd.factorize(df["Payment_Format"])[0]

order = np.argsort(df["Timestamp"].values, kind="stable")
tr, va, te = order[:int(n * .6)], order[int(n * .6):int(n * .8)], order[int(n * .8):]
y_tr, y_va, y_te = y[tr], y[va], y[te]
A_te = usd[te]
log(f"Загрузка: {time.time() - t0:.0f} с. Тест: {len(te):,} переводов, отмывочных {int(y_te.sum())}")

# ------------------------------------------------------------------ вспомогательные функции


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


def topk(score, K):
    K = min(K, len(score))
    return np.argpartition(-score, K - 1)[:K]


def capture(score, L, yt, K):
    t = topk(score, K)
    return float((L[t] * yt[t]).sum() / (L * yt).sum())


def oracle(L, yt, K):
    v = np.sort(L[yt == 1])[::-1]
    return float(v[:K].sum() / v.sum())


def n_eff(L):
    return float(L.sum() ** 2 / (L ** 2).sum())


def ms(a):
    a = np.asarray(a, dtype=float) * 100
    return f"{a.mean():5.1f} ± {a.std():4.1f}" if len(a) > 1 else f"{a.mean():5.1f}       "


# ------------------------------------------------------------------ веса обучения (для теоретической поправки)
pos_w = float((y_tr == 0).sum() / y_tr.sum())
V = pos_w ** 0.5                                   # вес класса 1 в шагах 7 и 10
Ltr_pos = np.minimum(usd[tr][y_tr == 1], CAP)
mean_L_pos = Ltr_pos.mean()
lw_tr = np.minimum(Ltr_pos / mean_L_pos, MAX_W)
lw_mean = lw_tr.mean()


def u_of_L(L):
    """Множитель веса отмывочного перевода в варианте amount (шаг 7): u(L) = min(L/mean, MAX_W)/mean(...)."""
    # при u -> 0 поправка неустойчива (у таких переводов вес в обучении почти нулевой): нижний порог U_MIN
    return np.maximum(np.minimum(L / mean_L_pos, MAX_W) / lw_mean, U_MIN)


log(f"Вес класса v = {V:.1f}; средний L отмывочных в train = {mean_L_pos:,.0f} $; "
    f"u(L) от {u_of_L(np.array([0.0]))[0]:.3f} до {u_of_L(np.array([CAP]))[0]:.2f}")

# ------------------------------------------------------------------ XGBoost (как в шаге 10)
import xgboost as xgb

lusd = np.log1p(usd)
IS_ACH = (df["Payment_Format"].values == "ACH").astype(float)
PAIR = S * N + D
SB, DB = banks[:n], banks[n:]
NB = int(banks.max()) + 1


def sdiv(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def graph_feats(gidx):
    s_, d_ = S[gidx], D[gidx]
    od = np.bincount(s_, minlength=N).astype(float)
    idg = np.bincount(d_, minlength=N).astype(float)
    pu, pinv, pc = np.unique(PAIR[gidx], return_inverse=True, return_counts=True)
    uo = np.bincount((pu // N).astype(np.int64), minlength=N).astype(float)
    ui = np.bincount((pu % N).astype(np.int64), minlength=N).astype(float)
    l_ = lusd[gidx]
    mlo = sdiv(np.bincount(s_, weights=l_, minlength=N), od)
    mli = sdiv(np.bincount(d_, weights=l_, minlength=N), idg)
    ao = sdiv(np.bincount(s_, weights=IS_ACH[gidx], minlength=N), od)
    ai = sdiv(np.bincount(d_, weights=IS_ACH[gidx], minlength=N), idg)
    cpp = np.unique(s_ * 1000 + cur_p[gidx])
    nc = np.bincount((cpp // 1000).astype(np.int64), minlength=N).astype(float)
    bs = np.bincount(SB[gidx], minlength=NB).astype(float)
    F_ = np.stack([od[s_], idg[s_], od[d_], idg[d_], uo[s_], ui[s_], uo[d_], ui[d_], pc[pinv].astype(float),
                   mlo[s_], mli[d_], l_ - mlo[s_], l_ - mli[d_], ao[s_], ai[d_], nc[s_], bs[SB[gidx]]], 1)
    return F_.astype(np.float32), od, idg


TX_RAW = np.stack([lusd, np.log1p(usd_r), cur_p, cur_r, fmt, (cur_p != cur_r), (S == D), (SB == DB),
                   df["Timestamp"].dt.hour.values, df["Timestamp"].dt.hour.values * 60 + df["Timestamp"].dt.minute.values],
                  1).astype(np.float32)
GF_tr, _, _ = graph_feats(tr)
GF_va, _, _ = graph_feats(np.concatenate([tr, va]))
GF_te, OD_all, ID_all = graph_feats(order)
Xtr = np.hstack([TX_RAW[tr], GF_tr])
Xva = np.hstack([TX_RAW[va], GF_va[len(tr):]])
Xte = np.hstack([TX_RAW[te], GF_te[len(tr) + len(va):]])
del GF_tr, GF_va, GF_te
# степень концов перевода в полном графе — для стоимости проверки в задаче о рюкзаке
DEG_te = OD_all[S[te]] + ID_all[S[te]] + OD_all[D[te]] + ID_all[D[te]]

models = {}       # имя -> список (по сидам) калиброванных/исправленных вероятностей на тесте
raw = {}          # имя -> список сырых вероятностей на тесте
xgb_cache = OUTD / "scores_12_xgb.npz"
if xgb_cache.exists():
    z = np.load(xgb_cache)
    models["XGBoost"] = [z[f"pc{s}"] for s in SEEDS]
    raw["XGBoost"] = [z[f"p{s}"] for s in SEEDS]
    log("XGBoost: прогнозы взяты из scores_12_xgb.npz")
else:
    models["XGBoost"], raw["XGBoost"] = [], []
    store = {}
    for s in SEEDS:
        t1 = time.time()
        clf = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", scale_pos_weight=V,
                                eval_metric="aucpr", early_stopping_rounds=100, random_state=s, n_jobs=-1)
        clf.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
        pv, pt = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
        a, b = fit_platt(lg(pv), y_va.astype(float))
        pc = sig(a * lg(pt) + b)
        models["XGBoost"].append(pc)
        raw["XGBoost"].append(pt)
        store[f"pc{s}"], store[f"p{s}"] = pc, pt
        log(f"  XGBoost сид {s}: {time.time() - t1:.0f} с, Платт a = {a:.3f}, b = {b:.3f} "
            f"(теория для веса класса: a = 1, b = -log v = {-np.log(V):.3f})")
    np.savez_compressed(xgb_cache, **store)
del Xtr, Xva, Xte

# ------------------------------------------------------------------ сохранённые прогнозы GNN
L1 = np.minimum(A_te, CAP)


def load_scores(pattern):
    out = []
    for s in SEEDS:
        p = OUTD / pattern.format(s=s)
        if p.exists():
            arr = np.load(p)
            idx, sc = arr[0].astype(np.int64), arr[1]
            if not np.array_equal(idx, te):
                pos = pd.Series(np.arange(len(idx)), index=idx).reindex(te).values
                sc = sc[pos]
            out.append(sc)
    return out


gnn_specs = [("GNN class", "test_scores_07_class_s{s}.npy", "class"),
             ("GNN amount", "test_scores_07_amount_s{s}.npy", "amount"),
             ("GNN-гибрид", "test_scores_10_gnnhybrid_s{s}.npy", "class")]
for name, pat, kind in gnn_specs:
    sc = load_scores(pat)
    if not sc:
        log(f"{name}: файлов прогнозов нет — пропуск")
        continue
    raw[name] = sc
    if kind == "class":
        models[name] = [sig(lg(p) - np.log(V)) for p in sc]                       # предложение 2.6
    else:
        models[name] = [sig(lg(p) - np.log(V) - np.log(u_of_L(L1))) for p in sc]  # предложение 2.7, схема В
    log(f"{name}: прогнозов {len(sc)} (сиды), поправка: "
        + ("logit p - log v" if kind == "class" else "logit p - log v - log u(L)"))

# ================================================================== Э2. Схемы А / Б / В
log("\n=== Э2. Восстановление вероятностей без валидации (предложения 2.6, 2.7), L = min(A, 1 млн) ===")
log("Доля предотвращённого ущерба, %. Сырой p — выход модели; pi^ — после теоретической поправки.")
log("  ожидаемое число отмывочных среди теста: сумма pi^ должна быть близка к фактическому числу")
hdr = f"  {'модель':<12}{'K':>6}{'по p':>16}{'по p·L':>16}{'по pi^·L':>16}{'оракул':>9}"
log(hdr)
for name in models:
    for K in KS:
        r_p = [capture(p, L1, y_te, K) for p in raw[name]]
        r_pL = [capture(p * L1, L1, y_te, K) for p in raw[name]]
        r_cL = [capture(p * L1, L1, y_te, K) for p in models[name]]
        log(f"  {name:<12}{K:>6}{ms(r_p):>16}{ms(r_pL):>16}{ms(r_cL):>16}{oracle(L1, y_te, K) * 100:>8.1f}")
log("\n  калибровка в среднем: сумма вероятностей на тесте против числа отмывочных "
    f"({int(y_te.sum())})")
for name in models:
    log(f"  {name:<12} сырой p: {np.mean([p.sum() for p in raw[name]]):10.0f}   "
        f"pi^: {np.mean([p.sum() for p in models[name]]):10.0f}")

# ================================================================== Э1. Чувствительность к L
log("\n=== Э1. Чувствительность к функции потерь (п. 2.6.3) ===")
fns = [("min(A, " + f"{int(c):,}".replace(",", " ") + ")", (lambda a, c=c: np.minimum(a, c))) for c in (1e4, 1e5, 1e6, 1e7)]
fns += [("sqrt(A)", np.sqrt), ("log(1+A)", np.log1p), ("A", lambda a: a.astype(float))]
loss_forms = [(lab, f(A_te)) for lab, f in fns]
yall = y.astype(bool)
log("n_eff по отмывочным переводам (весь набор / тест):")
for (lab, f), (_, Lt) in zip(fns, loss_forms):
    log(f"  {lab:<18} {n_eff(f(usd[yall])):10.0f} / {n_eff(Lt[y_te == 1]):8.0f}")

for K in KS:
    log(f"\nK = {K}: доля предотвращённого ущерба, % (по pi^ -> по pi^·L; оракул)")
    head = f"  {'L':<18}" + "".join(f"{nm:>26}" for nm in models) + f"{'оракул':>9}"
    log(head)
    for lab, Lt in loss_forms:
        row = f"  {lab:<18}"
        for name in models:
            a = np.mean([capture(p, Lt, y_te, K) for p in models[name]]) * 100
            b = np.mean([capture(p * Lt, Lt, y_te, K) for p in models[name]]) * 100
            row += f"{a:>12.1f} -> {b:>5.1f}      "
        log(row + f"{oracle(Lt, y_te, K) * 100:>8.1f}")
log("Вывод устойчив, если правая цифра в каждой паре не меньше левой при всех L.")

# ================================================================== Э3. Рюкзак
log("\n=== Э3. Разная стоимость проверок: задача о рюкзаке (п. 2.3) ===")
log("c_i = 1 + kappa·log(1 + d_i), d_i — число переводов отправителя и получателя в полном графе.")
log("Бюджет B = K · (средняя стоимость). G — жадный по плотности (префикс до первого не влезшего),")
log("G+ — тот же порядок с дозаполнением, U — граница Данцига, серт. = (U - G+)/G+ (следствие 2.5).")
log("Ценность — ожидаемый предотвращённый ущерб Σ pi^·L (в $) и фактическая доля ущерба, %.")


def greedy_fill(key, c, B, limit=400_000):
    o = np.argsort(-key, kind="stable")[:limit]
    take, rest = [], B
    cmin = c.min()
    for i in o:
        if c[i] <= rest:
            take.append(i)
            rest -= c[i]
            if rest < cmin:
                break
    return np.array(take, dtype=np.int64)


def dantzig(w, c, B):
    o = np.argsort(-(w / c), kind="stable")
    cc, ww = np.cumsum(c[o]), np.cumsum(w[o])
    k = int(np.searchsorted(cc, B, side="right"))          # первый не влезший в порядке плотности
    G = ww[k - 1] if k > 0 else 0.0
    used = cc[k - 1] if k > 0 else 0.0
    U = G + (B - used) * w[o[k]] / c[o[k]] if k < len(o) else G
    return G, U, w[o[k]] if k < len(o) else 0.0


tot = float((L1 * y_te).sum())
name_main = "GNN-гибрид" if "GNN-гибрид" in models else "XGBoost"
for kappa in (0.5, 1.0, 2.0):
    c = 1 + kappa * np.log1p(DEG_te)
    log(f"\n  kappa = {kappa}: стоимость от {c.min():.1f} до {c.max():.1f}, средняя {c.mean():.2f}")
    log(f"  {'модель':<12}{'K':>6}{'по pi^':>14}{'по pi^·L':>14}{'по pi^·L/c':>14}"
        f"{'E[ущерб] G+':>16}{'серт., %':>10}{'w_k/G, %':>10}")
    for name in (["XGBoost", name_main] if name_main != "XGBoost" else ["XGBoost"]):
        for K in KS:
            B = K * c.mean()
            res = {"p": [], "w": [], "d": [], "ev": [], "cert": [], "wk": []}
            for p in models[name]:
                w = p * L1
                for key, lab in ((p, "p"), (w, "w"), (w / c, "d")):
                    sel = greedy_fill(key, c, B)
                    res[lab].append((L1[sel] * y_te[sel]).sum() / tot)
                    if lab == "d":
                        Gp = w[sel].sum()
                G, U, wk = dantzig(w, c, B)
                res["ev"].append(Gp)
                res["cert"].append((U - Gp) / Gp)
                res["wk"].append(wk / max(G, 1e-12))
            log(f"  {name:<12}{K:>6}{ms(res['p']):>14}{ms(res['w']):>14}{ms(res['d']):>14}"
                f"{np.mean(res['ev']):>16,.0f}{np.mean(res['cert']) * 100:>10.3f}{np.mean(res['wk']) * 100:>10.3f}")
log("Если серт. мал (доли процента), жадный отбор почти оптимален при данных pi^ — без решения рюкзака.")

log(f"\nГотово за {(time.time() - t0) / 60:.1f} мин.")
