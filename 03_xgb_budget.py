# -*- coding: utf-8 -*-
"""
Шаг 3. Первая модель (XGBoost) + отбор транзакций на проверку при бюджете K.

Что делает:
  1) строит признаки транзакции и простые графовые признаки счетов (метки в признаках НЕ используются);
  2) временной сплит 60/20/20, обучение на train, ранняя остановка на val;
  3) F1 по классу «отмывание» (порог подбирается на val) и PR-AUC — для сравнения с литературой;
  4) калибровка вероятностей на val (Platt);
  5) бюджет K проверок на тесте: ранжирование по p против ранжирования по p·L
     для нескольких вариантов потерь L (потолок, корень, логарифм).
Запуск: python 03_xgb_budget.py. Результаты: results_03.txt
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
OUT = HERE / "results_03.txt"
SEED = 42
CAP = 1_000_000          # потолок потерь C в USD (параметр штрафа; варьируется в анализе чувствительности)
BUDGETS = [100, 250, 500, 1000, 2000, 5000]
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
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
n = len(df)
y = df["Is_Laundering"].values.astype(np.int8)
log(f"Загружено {n:,} транзакций за {time.time() - t0:.0f} с")

# ---------------------------------------------------------------- курсы и суммы в USD
rates = {"US Dollar": 1.0}
m = (df["Receiving_Currency"] == "US Dollar") & (df["Payment_Currency"] != "US Dollar") & (df["Amount_Paid"] > 0)
rates.update((df.loc[m, "Amount_Received"] / df.loc[m, "Amount_Paid"]).groupby(df.loc[m, "Payment_Currency"]).median().to_dict())
m2 = (df["Payment_Currency"] == "US Dollar") & (df["Receiving_Currency"] != "US Dollar") & (df["Amount_Received"] > 0)
for k, v in (df.loc[m2, "Amount_Paid"] / df.loc[m2, "Amount_Received"]).groupby(df.loc[m2, "Receiving_Currency"]).median().items():
    rates.setdefault(k, v)
usd = (df["Amount_Paid"] * df["Payment_Currency"].map(rates)).values
usd_r = (df["Amount_Received"] * df["Receiving_Currency"].map(rates)).values
lusd = np.log1p(usd)

# ---------------------------------------------------------------- признаки
codes, _ = pd.factorize(pd.concat([df["From_Bank"] + "_" + df["From_Acc"], df["To_Bank"] + "_" + df["To_Acc"]]))
s, d = codes[:n], codes[n:]
A = codes.max() + 1
bank_codes, _ = pd.factorize(pd.concat([df["From_Bank"], df["To_Bank"]]))
sb, db = bank_codes[:n], bank_codes[n:]
cur_p = pd.factorize(df["Payment_Currency"])[0]
cur_r = pd.factorize(df["Receiving_Currency"])[0]
fmt = pd.factorize(df["Payment_Format"])[0]
is_ach = (df["Payment_Format"].values == "ACH").astype(float)

out_deg = np.bincount(s, minlength=A).astype(float)
in_deg = np.bincount(d, minlength=A).astype(float)
pair = s.astype(np.int64) * A + d
_, pinv, pcnt = np.unique(pair, return_inverse=True, return_counts=True)
upair = np.unique(pair)
u_out = np.bincount((upair // A).astype(np.int64), minlength=A).astype(float)   # число разных получателей
u_in = np.bincount((upair % A).astype(np.int64), minlength=A).astype(float)     # число разных отправителей
sum_l_out = np.bincount(s, weights=lusd, minlength=A)
sum_l_in = np.bincount(d, weights=lusd, minlength=A)
ach_out = np.bincount(s, weights=is_ach, minlength=A)
ach_in = np.bincount(d, weights=is_ach, minlength=A)
cur_pairs = np.unique(s.astype(np.int64) * 1000 + cur_p)
n_cur_out = np.bincount((cur_pairs // 1000).astype(np.int64), minlength=A).astype(float)


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


mean_l_out = safe_div(sum_l_out, out_deg)
mean_l_in = safe_div(sum_l_in, in_deg)
hour = df["Timestamp"].dt.hour.values
minute_of_day = hour * 60 + df["Timestamp"].dt.minute.values

X = pd.DataFrame({
    "log_usd_paid": lusd,
    "log_usd_recv": np.log1p(usd_r),
    "cur_paid": cur_p, "cur_recv": cur_r, "fmt": fmt,
    "diff_currency": (cur_p != cur_r).astype(np.int8),
    "self_transfer": (s == d).astype(np.int8),
    "same_bank": (sb == db).astype(np.int8),
    "hour": hour, "minute_of_day": minute_of_day,
    "src_out_deg": out_deg[s], "src_in_deg": in_deg[s],
    "dst_out_deg": out_deg[d], "dst_in_deg": in_deg[d],
    "src_fan_out": u_out[s], "src_fan_in": u_in[s],
    "dst_fan_out": u_out[d], "dst_fan_in": u_in[d],
    "pair_count": pcnt[pinv],
    "src_mean_log_out": mean_l_out[s], "dst_mean_log_in": mean_l_in[d],
    "amt_vs_src_mean": lusd - mean_l_out[s],
    "amt_vs_dst_mean": lusd - mean_l_in[d],
    "src_ach_share_out": safe_div(ach_out, out_deg)[s],
    "dst_ach_share_in": safe_div(ach_in, in_deg)[d],
    "src_n_currencies": n_cur_out[s],
    "src_bank_size": np.bincount(sb, minlength=bank_codes.max() + 1)[sb],
})
log(f"Признаков: {X.shape[1]}  (графовые считаются по структуре всего графа без меток, как в протоколе Altman et al.)")

# ---------------------------------------------------------------- временной сплит
order = np.argsort(df["Timestamp"].values, kind="stable")
tr, va, te = order[:int(n * .6)], order[int(n * .6):int(n * .8)], order[int(n * .8):]
log(f"train {len(tr):,} (отмыв. {y[tr].sum()}), val {len(va):,} ({y[va].sum()}), test {len(te):,} ({y[te].sum()})")

# ---------------------------------------------------------------- обучение
pos_w = float((y[tr] == 0).sum()) / max(1, y[tr].sum())
clf = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=8, min_child_weight=1,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    scale_pos_weight=pos_w ** 0.5, eval_metric="aucpr",
    early_stopping_rounds=100, random_state=SEED, n_jobs=-1,
)
t1 = time.time()
clf.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
log(f"XGBoost обучен за {time.time() - t1:.0f} с, лучшая итерация {clf.best_iteration}")
p_va = clf.predict_proba(X.iloc[va])[:, 1]
p_te = clf.predict_proba(X.iloc[te])[:, 1]


# ---------------------------------------------------------------- метрики классификации (без sklearn)
def pr_curve(yt, sc):
    o = np.argsort(-sc, kind="stable")
    yt = yt[o]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    prec = tp / (tp + fp)
    rec = tp / max(1, yt.sum())
    return prec, rec, sc[o]


def pr_auc(yt, sc):
    prec, rec, _ = pr_curve(yt, sc)
    r_prev = np.concatenate([[0], rec[:-1]])
    return float(np.sum((rec - r_prev) * prec))


prec_v, rec_v, thr_v = pr_curve(y[va], p_va)
f1_v = 2 * prec_v * rec_v / np.maximum(prec_v + rec_v, 1e-12)
best = int(np.argmax(f1_v))
thr = thr_v[best]
pred_te = (p_te >= thr).astype(int)
tp = int(((pred_te == 1) & (y[te] == 1)).sum())
fp = int(((pred_te == 1) & (y[te] == 0)).sum())
fn = int(((pred_te == 0) & (y[te] == 1)).sum())
f1_te = 2 * tp / max(1, 2 * tp + fp + fn)
log("\n=== Классификация (для сравнения с литературой) ===")
log(f"  Порог (подобран на val по F1): {thr:.4f};  F1 на val {f1_v[best]:.4f}")
log(f"  TEST: F1 = {f1_te * 100:.2f}%   precision = {tp / max(1, tp + fp):.3f}   recall = {tp / max(1, tp + fn):.3f}")
log(f"  TEST: PR-AUC = {pr_auc(y[te], p_te):.4f}  (случайная модель = {y[te].mean():.4f})")
log("  Ориентиры из литературы, HI-Small F1: GFP+XGBoost 63.2, Multi-PNA 68.2, Multi-FraudGT 76.1")

imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
log("  Важность признаков (топ-10, gain-share):")
for k, v in imp.head(10).items():
    log(f"    {k:<22} {v:.3f}")

# ---------------------------------------------------------------- калибровка (Platt на val)
def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_platt(z, t, iters=50):
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([np.sum((q - t) * z), np.sum(q - t)])
        w = q * (1 - q)
        H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-9 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(H, g)
    return a, b


a, b = fit_platt(logit(p_va), y[va].astype(float))
pc_te = 1 / (1 + np.exp(-(a * logit(p_te) + b)))
log("\n=== Калибровка ===")
for name, pp in [("сырые p", p_te), ("после Platt", pc_te)]:
    log(f"  {name:<12} сумма p на тесте = {pp.sum():8.1f}  при реальном числе отмывочных = {y[te].sum()}")

# ---------------------------------------------------------------- бюджет K
yt = y[te]
amt = usd[te]
losses = {
    "L=1 (случаи)": np.ones_like(amt),
    f"L=min(сумма,{CAP / 1e6:.0f} млн)": np.minimum(amt, CAP),
    "L=sqrt(сумма)": np.sqrt(amt),
    "L=log(1+сумма)": np.log1p(amt),
    "L=сумма": amt,
}
rng = np.random.default_rng(SEED)
strategies = {
    "по p (обычная модель)": pc_te,
    "по p·min(сумма,C)": pc_te * np.minimum(amt, CAP),
    "по p·sqrt(сумма)": pc_te * np.sqrt(amt),
    "по p·log(1+сумма)": pc_te * np.log1p(amt),
    "по p·сумма": pc_te * amt,
    "только по сумме (без модели)": amt,
    "случайно": rng.random(len(amt)),
}


def capture(score, L, K):
    K = min(K, len(score))
    top = np.argpartition(-score, K - 1)[:K]
    return float((L[top] * yt[top]).sum() / (L * yt).sum())


def oracle(L, K):
    v = np.sort((L * yt)[yt == 1])[::-1]
    return float(v[:K].sum() / v.sum())


log("\n=== Бюджет K проверок на тесте: какая доля потерь предотвращена ===")
log(f"  (в тесте {len(yt):,} транзакций, из них {yt.sum()} отмывочных)")
for Lname, L in losses.items():
    log(f"\n  Потери {Lname}:")
    log("    " + f"{'стратегия':<30}" + "".join(f"{'K=' + str(K):>9}" for K in BUDGETS))
    for sname, sc in strategies.items():
        log("    " + f"{sname:<30}" + "".join(f"{capture(sc, L, K):>9.3f}" for K in BUDGETS))
    log("    " + f"{'идеал (знаем все метки)':<30}" + "".join(f"{oracle(L, K):>9.3f}" for K in BUDGETS))

log("\n  Сырые vs калиброванные p (потери L=min(сумма,C)):")
Lc = losses[f"L=min(сумма,{CAP / 1e6:.0f} млн)"]
for name, pp in [("p·L по сырым p", p_te), ("p·L по калиброванным p", pc_te)]:
    log("    " + f"{name:<30}" + "".join(f"{capture(pp * np.minimum(amt, CAP), Lc, K):>9.3f}" for K in BUDGETS))

np.save(HERE / "test_scores_03.npy", np.vstack([te, p_te, pc_te]))
log(f"\nГотово за {time.time() - t0:.0f} с.")
OUT.write_text("\n".join(lines), encoding="utf-8")
