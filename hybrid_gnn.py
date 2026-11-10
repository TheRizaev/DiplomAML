import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--loss", default="class", choices=["class", "amount"])
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--epochs", type=int, default=600)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--eval_every", type=int, default=20)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--max_w", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cap", type=float, default=1_000_000)
ap.add_argument("--tag", default="", help="суффикс имён файлов, например _long")
ap.add_argument("--cur_flag", default="codes", choices=["codes", "exact"],
                help="признак «разные валюты»: сравнение кодов валют (как в основной серии) или названий валют")
ap.add_argument("--norm", default="all", choices=["all", "train"],
                help="статистики стандартизации признаков перевода: по всем переводам или только по обучающим")
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
import os as _os
DATA = _os.environ.get("AML_DATA", "HI-Small")
OUTD = HERE if DATA == "HI-Small" else HERE / DATA
OUTD.mkdir(exist_ok=True)

DOWNLOADS = Path.home() / "Downloads"
TAG = f"s{args.seed}{args.tag}"
OUT = OUTD / f"results_10_hybrid_{TAG}.txt"
BUDGETS = [100, 500, 1000, 2000]
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(os.cpu_count() or 4)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    OUT.write_text("\n".join(lines), encoding="utf-8")


def find(stem, exts):
    for d in (HERE, HERE / "data", DOWNLOADS):
        for e in exts:
            p = d / f"{stem}{e}"
            if p.exists():
                return p
    return None


t0 = time.time()
log(f"Устройство: {DEV}, потоков: {torch.get_num_threads()}, вариант потерь: {args.loss}, "
    f"hidden={args.hidden}, layers={args.layers}")

COLS = ["Timestamp", "From_Bank", "From_Acc", "To_Bank", "To_Acc", "Amount_Received",
        "Receiving_Currency", "Amount_Paid", "Payment_Currency", "Payment_Format", "Is_Laundering"]
df = pd.read_csv(find(f"{DATA}_Trans", [".csv", ".zip"]), header=0, names=COLS,
                 dtype={"From_Bank": str, "To_Bank": str, "From_Acc": str, "To_Acc": str})
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
n = len(df)
_days = df["Timestamp"].dt.floor("D").value_counts().sort_index()
_small = _days < 0.01 * _days.median()
_start = next((d for i, d in enumerate(_days.index) if _small.iloc[i:].all()), _days.index[-1] + pd.Timedelta(days=1))
TAIL_START = np.datetime64(_start)
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
if args.cur_flag == "exact":
    DIFF_CUR = df["Payment_Currency"].values != df["Receiving_Currency"].values
else:
    DIFF_CUR = cur_p != cur_r
tmin = df["Timestamp"].values.astype("datetime64[m]").astype(np.int64)

e = pd.DataFrame({"s": S, "d": D, "t": tmin})
first = e.groupby(["s", "d"], sort=False)["t"].min().reset_index()
first["out_port"] = first.sort_values("t").groupby("s").cumcount()
first["in_port"] = first.sort_values("t").groupby("d").cumcount()
e = e.merge(first[["s", "d", "out_port", "in_port"]], on=["s", "d"], how="left")
out_port, in_port = e["out_port"].values, e["in_port"].values
del e, first

num = np.stack([
    np.log1p(usd), np.log1p(usd_r),
    DIFF_CUR.astype(np.float32), (S == D).astype(np.float32),
    (banks[:n] == banks[n:]).astype(np.float32),
    df["Timestamp"].dt.hour.values / 23.0,
    np.log1p(out_port), np.log1p(in_port),
], axis=1).astype(np.float32)
if args.norm == "train":
    _tr = np.argsort(df["Timestamp"].values, kind="stable")[:int(n * .6)]
    mu, sd = num[_tr].mean(0), num[_tr].std(0) + 1e-6
else:
    mu, sd = num.mean(0), num.std(0) + 1e-6
num = (num - mu) / sd
NUM = torch.tensor(num)
CP, CR, FM = (torch.tensor(a, dtype=torch.long) for a in (cur_p, cur_r, fmt))
ST, DT = torch.tensor(S), torch.tensor(D)
Y = torch.tensor(y)

order = np.argsort(df["Timestamp"].values, kind="stable")
tr, va, te = order[:int(n * .6)], order[int(n * .6):int(n * .8)], order[int(n * .8):]

lusd = np.log1p(usd)
IS_ACH = (df["Payment_Format"].values == "ACH").astype(float)
PAIR = S * N + D
SB, DB = banks[:n], banks[n:]
NB = int(banks.max()) + 1
GF_NAMES = ["src_out_deg", "src_in_deg", "dst_out_deg", "dst_in_deg", "src_fan_out", "src_fan_in",
            "dst_fan_out", "dst_fan_in", "pair_count", "src_mean_log_out", "dst_mean_log_in",
            "amt_vs_src_mean", "amt_vs_dst_mean", "src_ach_share_out", "dst_ach_share_in",
            "src_n_currencies", "src_bank_size"]
LOG_COLS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16}


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
    return F_.astype(np.float32)


TX_RAW = np.stack([lusd, np.log1p(usd_r), cur_p, cur_r, fmt, DIFF_CUR, (S == D), (SB == DB),
                   df["Timestamp"].dt.hour.values, df["Timestamp"].dt.hour.values * 60 + df["Timestamp"].dt.minute.values],
                  1).astype(np.float32)
GF = {}
for name, gidx in (("tr", tr), ("va", np.concatenate([tr, va])), ("te", order)):
    GF[name] = graph_feats(gidx)
GF_T = {k: v.copy() for k, v in GF.items()}
for k in GF_T:
    for c in LOG_COLS:
        GF_T[k][:, c] = np.log1p(GF_T[k][:, c])
gmu, gsd = GF_T["tr"].mean(0), GF_T["tr"].std(0) + 1e-6
for k in GF_T:
    GF_T[k] = torch.from_numpy(((GF_T[k] - gmu) / gsd).astype(np.float32))
NG = len(GF_NAMES)
log(f"Загрузка и признаки: {time.time() - t0:.0f} с.  Счетов {N:,}, транзакций {n:,}")


class Graph:

    def __init__(self, idx, gf):
        self.idx = torch.tensor(idx, dtype=torch.long)
        self.s, self.d = ST[self.idx], DT[self.idx]
        self.num, self.cp, self.cr, self.fm = (x[self.idx] for x in (NUM, CP, CR, FM))
        self.num = torch.cat([self.num, gf], 1)
        deg_in = torch.bincount(self.d, minlength=N).float()
        deg_out = torch.bincount(self.s, minlength=N).float()
        self.deg_in, self.deg_out = deg_in.clamp(min=1), deg_out.clamp(min=1)
        self.node_x = torch.stack([torch.log1p(deg_in), torch.log1p(deg_out)], 1)

    def to(self, dev):
        g = Graph.__new__(Graph)
        for k in ("s", "d", "num", "cp", "cr", "fm", "deg_in", "deg_out", "node_x"):
            setattr(g, k, getattr(self, k).to(dev))
        return g


G_tr = Graph(tr, GF_T["tr"]).to(DEV)
G_va = Graph(np.concatenate([tr, va]), GF_T["va"])
G_te = Graph(order, GF_T["te"])
pos_va = torch.arange(len(tr), len(tr) + len(va), device=DEV)
pos_te = torch.arange(len(tr) + len(va), n, device=DEV)


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class MultiGNN(nn.Module):
    def __init__(self, H, L):
        super().__init__()
        self.e_cp, self.e_cr, self.e_fm = nn.Embedding(32, 4), nn.Embedding(32, 4), nn.Embedding(16, 4)
        self.edge_in = nn.Linear(NUM.shape[1] + NG + 12, H)
        self.node_in = nn.Linear(2, H)
        self.msg_in = nn.ModuleList([mlp(2 * H, H, H) for _ in range(L)])
        self.msg_out = nn.ModuleList([mlp(2 * H, H, H) for _ in range(L)])
        self.upd = nn.ModuleList([mlp(3 * H, H, H) for _ in range(L)])
        self.eupd = nn.ModuleList([mlp(3 * H, H, H) for _ in range(L)])
        self.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self.enorm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self.head = mlp(3 * H, 1, H)

    def layer(self, k, h, ea, s, d, deg_in, deg_out):
        m_in = self.msg_in[k](torch.cat([h[s], ea], 1))
        m_out = self.msg_out[k](torch.cat([h[d], ea], 1))
        a_in = torch.zeros_like(h).index_add_(0, d, m_in) / deg_in[:, None]
        a_out = torch.zeros_like(h).index_add_(0, s, m_out) / deg_out[:, None]
        h = self.norm[k](h + self.upd[k](torch.cat([h, a_in, a_out], 1)))
        ea = self.enorm[k](ea + self.eupd[k](torch.cat([h[s], h[d], ea], 1)))
        return h, ea

    def forward(self, g):
        ea = F.relu(self.edge_in(torch.cat([g.num, self.e_cp(g.cp), self.e_cr(g.cr), self.e_fm(g.fm)], 1)))
        h = F.relu(self.node_in(g.node_x))
        for k in range(len(self.upd)):
            if self.training:
                h, ea = checkpoint(self.layer, k, h, ea, g.s, g.d, g.deg_in, g.deg_out, use_reentrant=False)
            else:
                h, ea = self.layer(k, h, ea, g.s, g.d, g.deg_in, g.deg_out)
        return self.head(torch.cat([h[g.s], h[g.d], ea], 1)).squeeze(1)


model = MultiGNN(args.hidden, args.layers).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
log(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

y_tr = Y[torch.tensor(tr)].to(DEV)
pos_w = float((y_tr == 0).sum() / y_tr.sum())
w = torch.ones_like(y_tr)
if args.loss == "class":
    w[y_tr == 1] = pos_w ** 0.5
else:
    Ltr = torch.tensor(np.minimum(usd[tr], args.cap), dtype=torch.float32, device=DEV)
    lw = (Ltr[y_tr == 1] / Ltr[y_tr == 1].mean()).clamp(max=args.max_w)
    lw = lw / lw.mean()
    w[y_tr == 1] = pos_w ** 0.5 * lw
log(f"Веса: класс 1 = {pos_w ** 0.5:.1f} в среднем; вариант {args.loss}")


def pr_auc(yt, sc):
    o = np.argsort(-sc, kind="stable")
    yy = yt[o]
    tp = np.cumsum(yy)
    p = tp / np.arange(1, len(yy) + 1)
    r = tp / max(1, yy.sum())
    return float(np.sum((r - np.concatenate([[0], r[:-1]])) * p))


@torch.no_grad()
def predict(g_cpu, pos):
    model.eval()
    g = g_cpu.to(DEV)
    out = torch.sigmoid(model(g)[pos]).cpu().numpy()
    del g
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    model.train()
    return out


best, best_state, bad = -1, None, 0
y_va = y[va]
for ep in range(1, args.epochs + 1):
    t1 = time.time()
    opt.zero_grad()
    logit = model(G_tr)
    loss = F.binary_cross_entropy_with_logits(logit, y_tr, weight=w)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    opt.step()
    if ep % args.eval_every == 0 or ep == 1:
        p_va = predict(G_va, pos_va)
        ap_va = pr_auc(y_va, p_va)
        log(f"  эпоха {ep:4d}  loss {loss.item():.4f}  PR-AUC val {ap_va:.4f}  ({time.time() - t1:.1f} с/эпоха, "
            f"всего {(time.time() - t0) / 60:.1f} мин)")
        if ap_va > best:
            best, bad = ap_va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log("  ранняя остановка")
                break

model.load_state_dict(best_state)
p_va = predict(G_va, pos_va)
p_te = predict(G_te, pos_te)
y_te = y[te]
np.save(OUTD / f"test_scores_10_gnnhybrid_{TAG}.npy", np.vstack([te, p_te]))
np.save(OUTD / f"val_scores_10_gnnhybrid_{TAG}.npy", np.vstack([va, p_va]))


def pr_curve(yt, sc):
    o = np.argsort(-sc, kind="stable")
    yy = yt[o]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    return tp / (tp + fp), tp / max(1, yy.sum()), sc[o]


def f1_at(yt, sc, thr):
    pr = sc >= thr
    tp = int((pr & (yt == 1)).sum()); fp = int((pr & (yt == 0)).sum()); fn = int((~pr & (yt == 1)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def lg(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_platt(z, t, iters=50):
    a, b = 1.0, 0.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(a * z + b)))
        g = np.array([np.sum((q - t) * z), np.sum(q - t)])
        ww = q * (1 - q)
        H = np.array([[np.sum(ww * z * z), np.sum(ww * z)], [np.sum(ww * z), np.sum(ww)]]) + 1e-9 * np.eye(2)
        a, b = np.array([a, b]) - np.linalg.solve(H, g)
    return a, b


def capture(score, L, yt, K):
    K = min(K, len(score))
    top = np.argpartition(-score, K - 1)[:K]
    return float((L[top] * yt[top]).sum() / (L * yt).sum())


import json
import xgboost as xgb

no_tail = df["Timestamp"].values[te] < TAIL_START
amt = usd[te]
Lc = np.minimum(amt, args.cap)
ones = np.ones_like(amt)


def evaluate(name, pva, pte):
    pv, rv, tv = pr_curve(y_va, pva)
    thr = tv[int(np.argmax(2 * pv * rv / np.maximum(pv + rv, 1e-12)))]
    a, b = fit_platt(lg(pva), y_va.astype(float))
    pc = 1 / (1 + np.exp(-(a * lg(pte) + b)))
    r = {"model": name, "seed": args.seed, "f1": f1_at(y_te, pte, thr), "prauc": pr_auc(y_te, pte),
         "f1_nt": f1_at(y_te[no_tail], pte[no_tail], thr), "prauc_nt": pr_auc(y_te[no_tail], pte[no_tail])}
    for K in BUDGETS:
        r[f"p_K{K}"] = capture(pc, Lc, y_te, K)
        r[f"pL_K{K}"] = capture(pc * Lc, Lc, y_te, K)
        r[f"nt_pL_K{K}"] = capture(pc[no_tail] * Lc[no_tail], Lc[no_tail], y_te[no_tail], K)
        r[f"case_pL_K{K}"] = capture(pc * Lc, ones, y_te, K)
    (OUTD / f"metrics_10_{name}_{TAG}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    log(f"  {name:<12} F1 {r['f1'] * 100:6.2f}  без хвоста {r['f1_nt'] * 100:6.2f}  PR-AUC {r['prauc']:.4f}  "
        f"p·L K=100 {r['pL_K100'] * 100:5.1f}  K=1000 {r['pL_K1000'] * 100:5.1f}")
    return pc


t2 = time.time()
Xtr = np.hstack([TX_RAW[tr], GF["tr"]])
Xva = np.hstack([TX_RAW[va], GF["va"][len(tr):]])
Xte = np.hstack([TX_RAW[te], GF["te"][len(tr) + len(va):]])
ytr_np = y[tr]
clf = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", scale_pos_weight=(float((ytr_np == 0).sum()) / ytr_np.sum()) ** 0.5,
                        eval_metric="aucpr", early_stopping_rounds=100, random_state=args.seed, n_jobs=-1)
clf.fit(Xtr, ytr_np, eval_set=[(Xva, y_va)], verbose=False)
x_va, x_te = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
log(f"\nXGBoost обучен за {time.time() - t2:.0f} с")

log("\n=== Тест ===")
pc_g = evaluate("gnn_hybrid", p_va, p_te)
pc_x = evaluate("xgb", x_va, x_te)
np.save(OUTD / f"val_scores_10_xgb_{TAG}.npy", np.vstack([va, x_va]))
np.save(OUTD / f"test_scores_10_xgb_{TAG}.npy", np.vstack([te, x_te]))
e_va = (pd.Series(p_va).rank().values / len(p_va) + pd.Series(x_va).rank().values / len(x_va)) / 2
e_te = (pd.Series(p_te).rank().values / len(p_te) + pd.Series(x_te).rank().values / len(x_te)) / 2
evaluate("ensemble", np.clip(e_va, 1e-6, 1 - 1e-6), np.clip(e_te, 1e-6, 1 - 1e-6))
def platt_both(pv, pt):
    a, b = fit_platt(lg(pv), y_va.astype(float))
    return 1 / (1 + np.exp(-(a * lg(pv) + b))), 1 / (1 + np.exp(-(a * lg(pt) + b)))
gv, gt = platt_both(p_va, p_te)
xv, xt = platt_both(x_va, x_te)
evaluate("ensemble_cal", np.clip((gv + xv) / 2, 1e-7, 1 - 1e-7), np.clip((gt + xt) / 2, 1e-7, 1 - 1e-7))
log(f"\nГотово за {(time.time() - t0) / 60:.1f} мин.")
