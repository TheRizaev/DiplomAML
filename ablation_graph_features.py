import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
OUT = HERE / "results_09_ablation.txt"
SEEDS = [0, 1, 2]
CAP = 1_000_000
BUDGETS = [100, 500, 1000, 2000]
TAIL_START = pd.Timestamp("2022-09-11")
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
n = len(df)
y = df["Is_Laundering"].values.astype(np.int8)

rates = {"US Dollar": 1.0}
m = (df["Receiving_Currency"] == "US Dollar") & (df["Payment_Currency"] != "US Dollar") & (df["Amount_Paid"] > 0)
rates.update((df.loc[m, "Amount_Received"] / df.loc[m, "Amount_Paid"]).groupby(df.loc[m, "Payment_Currency"]).median().to_dict())
m2 = (df["Payment_Currency"] == "US Dollar") & (df["Receiving_Currency"] != "US Dollar") & (df["Amount_Received"] > 0)
for k, v in (df.loc[m2, "Amount_Paid"] / df.loc[m2, "Amount_Received"]).groupby(df.loc[m2, "Receiving_Currency"]).median().items():
    rates.setdefault(k, v)
usd = (df["Amount_Paid"] * df["Payment_Currency"].map(rates)).values
usd_r = (df["Amount_Received"] * df["Receiving_Currency"].map(rates)).values
lusd = np.log1p(usd)

codes, _ = pd.factorize(pd.concat([df["From_Bank"] + "_" + df["From_Acc"], df["To_Bank"] + "_" + df["To_Acc"]]))
S, D = codes[:n], codes[n:]
A = int(codes.max()) + 1
bank_codes, _ = pd.factorize(pd.concat([df["From_Bank"], df["To_Bank"]]))
SB, DB = bank_codes[:n], bank_codes[n:]
NB = int(bank_codes.max()) + 1
CUR_P = pd.factorize(df["Payment_Currency"])[0]
CUR_R = pd.factorize(df["Receiving_Currency"])[0]
FMT = pd.factorize(df["Payment_Format"])[0]
IS_ACH = (df["Payment_Format"].values == "ACH").astype(float)
HOUR = df["Timestamp"].dt.hour.values
MOD = HOUR * 60 + df["Timestamp"].dt.minute.values
PAIR = S.astype(np.int64) * A + D


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def features(graph_idx, target_idx):
    s, d = S[graph_idx], D[graph_idx]
    out_deg = np.bincount(s, minlength=A).astype(float)
    in_deg = np.bincount(d, minlength=A).astype(float)
    up = np.unique(PAIR[graph_idx])
    u_out = np.bincount((up // A).astype(np.int64), minlength=A).astype(float)
    u_in = np.bincount((up % A).astype(np.int64), minlength=A).astype(float)
    pu, pc = np.unique(PAIR[graph_idx], return_counts=True)
    l = lusd[graph_idx]
    mean_l_out = safe_div(np.bincount(s, weights=l, minlength=A), out_deg)
    mean_l_in = safe_div(np.bincount(d, weights=l, minlength=A), in_deg)
    ach_out = safe_div(np.bincount(s, weights=IS_ACH[graph_idx], minlength=A), out_deg)
    ach_in = safe_div(np.bincount(d, weights=IS_ACH[graph_idx], minlength=A), in_deg)
    cp = np.unique(s.astype(np.int64) * 1000 + CUR_P[graph_idx])
    n_cur = np.bincount((cp // 1000).astype(np.int64), minlength=A).astype(float)
    bank_size = np.bincount(SB[graph_idx], minlength=NB)

    t = target_idx
    st, dt = S[t], D[t]
    pos = np.searchsorted(pu, PAIR[t])
    pos = np.clip(pos, 0, len(pu) - 1)
    pair_cnt = np.where(pu[pos] == PAIR[t], pc[pos], 0)
    return pd.DataFrame({
        "log_usd_paid": lusd[t], "log_usd_recv": np.log1p(usd_r[t]),
        "cur_paid": CUR_P[t], "cur_recv": CUR_R[t], "fmt": FMT[t],
        "diff_currency": (CUR_P[t] != CUR_R[t]).astype(np.int8),
        "self_transfer": (st == dt).astype(np.int8),
        "same_bank": (SB[t] == DB[t]).astype(np.int8),
        "hour": HOUR[t], "minute_of_day": MOD[t],
        "src_out_deg": out_deg[st], "src_in_deg": in_deg[st],
        "dst_out_deg": out_deg[dt], "dst_in_deg": in_deg[dt],
        "src_fan_out": u_out[st], "src_fan_in": u_in[st],
        "dst_fan_out": u_out[dt], "dst_fan_in": u_in[dt],
        "pair_count": pair_cnt,
        "src_mean_log_out": mean_l_out[st], "dst_mean_log_in": mean_l_in[dt],
        "amt_vs_src_mean": lusd[t] - mean_l_out[st], "amt_vs_dst_mean": lusd[t] - mean_l_in[dt],
        "src_ach_share_out": ach_out[st], "dst_ach_share_in": ach_in[dt],
        "src_n_currencies": n_cur[st], "src_bank_size": bank_size[SB[t]],
    })


order = np.argsort(df["Timestamp"].values, kind="stable")
tr, va, te = order[:int(n * .6)], order[int(n * .6):int(n * .8)], order[int(n * .8):]
t1 = time.time()
X_tr = features(tr, tr)
X_va = features(np.concatenate([tr, va]), va)
X_te = features(order, te)
log(f"Признаки по протоколу (train-граф / train+val / весь граф) за {time.time() - t1:.0f} с")
y_tr, y_va, y_te = y[tr], y[va], y[te]
no_tail = (df["Timestamp"].values[te] < np.datetime64(TAIL_START))
log(f"Тест: {len(te):,} транзакций, отмывочных {y_te.sum()};  без хвоста после 10.09: {no_tail.sum():,}, "
    f"отмывочных {y_te[no_tail].sum()}")


def pr_curve(yt, sc):
    o = np.argsort(-sc, kind="stable")
    yy = yt[o]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    return tp / (tp + fp), tp / max(1, yy.sum()), sc[o]


def pr_auc(yt, sc):
    p, r, _ = pr_curve(yt, sc)
    return float(np.sum((r - np.concatenate([[0], r[:-1]])) * p))


def f1_at(yt, sc, thr):
    pr = sc >= thr
    tp = int((pr & (yt == 1)).sum()); fp = int((pr & (yt == 0)).sum()); fn = int((~pr & (yt == 1)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


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


def capture(score, L, yt, K):
    K = min(K, len(score))
    top = np.argpartition(-score, K - 1)[:K]
    return float((L[top] * yt[top]).sum() / (L * yt).sum())


amt = usd[te]
Lcap = np.minimum(amt, CAP)
TX = ["log_usd_paid", "log_usd_recv", "cur_paid", "cur_recv", "fmt", "diff_currency",
      "self_transfer", "same_bank", "hour", "minute_of_day"]
SETS = {
    "только перевод (без графа)": TX,
    "только графовые признаки": [c for c in X_tr.columns if c not in TX],
    "перевод + граф (как в шаге 4)": list(X_tr.columns),
}
summary = []
for name, cols in SETS.items():
    log(f"\n=== {name}: {len(cols)} признаков ===")
    rs = []
    for seed in SEEDS:
        t2 = time.time()
        clf = xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", scale_pos_weight=(float((y_tr == 0).sum()) / y_tr.sum()) ** 0.5,
            eval_metric="aucpr", early_stopping_rounds=100, random_state=seed, n_jobs=-1)
        clf.fit(X_tr[cols], y_tr, eval_set=[(X_va[cols], y_va)], verbose=False)
        p_va = clf.predict_proba(X_va[cols])[:, 1]
        p_te = clf.predict_proba(X_te[cols])[:, 1]
        pv, rv, tv = pr_curve(y_va, p_va)
        thr = tv[int(np.argmax(2 * pv * rv / np.maximum(pv + rv, 1e-12)))]
        a, b = fit_platt(logit(p_va), y_va.astype(float))
        pc = 1 / (1 + np.exp(-(a * logit(p_te) + b)))
        r = {"f1": f1_at(y_te, p_te, thr), "f1_nt": f1_at(y_te[no_tail], p_te[no_tail], thr),
             "prauc": pr_auc(y_te, p_te)}
        for K in (100, 1000):
            r[f"p_K{K}"] = capture(pc, Lcap, y_te, K)
            r[f"pL_K{K}"] = capture(pc * Lcap, Lcap, y_te, K)
        rs.append(r)
        log(f"  сид {seed}: {time.time() - t2:.0f} с, F1 {r['f1'] * 100:.2f}%, без хвоста {r['f1_nt'] * 100:.2f}%")
    R = pd.DataFrame(rs)
    summary.append((name, R))

log("\n=== Итог (среднее ± std по 3 сидам), % ===")
cols_show = [("F1", "f1"), ("F1 без хвоста", "f1_nt"), ("PR-AUC", "prauc"),
             ("p·L K=100", "pL_K100"), ("p·L K=1000", "pL_K1000"), ("p K=1000", "p_K1000")]
log("  " + f"{'набор признаков':<32}" + "".join(f"{c[0]:>16}" for c in cols_show))
for name, R in summary:
    log("  " + f"{name:<32}" + "".join(f"{R[c].mean() * 100:>9.1f} ± {R[c].std() * 100:4.1f}" for _, c in cols_show))
log(f"\nГотово за {time.time() - t0:.0f} с.")
OUT.write_text("\n".join(lines), encoding="utf-8")
