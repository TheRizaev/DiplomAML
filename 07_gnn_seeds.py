# -*- coding: utf-8 -*-
"""
Шаг 7. Графовая нейросеть (стабильная версия: lr 1e-3, ограничение градиента, ограничение весов, несколько сидов) для классификации транзакций (рёбер) — собственная реализация
в духе Multi-GNN (Egressy et al., AAAI 2024) на чистом PyTorch, без torch_geometric.

Что взято из Multi-GNN:
  * обратная передача сообщений: входящие и исходящие соседи агрегируются РАЗНЫМИ сетями;
  * номера портов: у каждого перевода есть номер получателя среди всех получателей отправителя
    (по времени первого контакта) и номер отправителя у получателя;
  * обновление рёбер (EU): признаки перевода обновляются на каждом слое по концам ребра.
Не реализовано: Ego ID (нужна выборка подграфов; на AML-задаче в статье почти не помогает).

Протокол как у Altman et al.: train-граф = train-рёбра, val-граф = train+val, test-граф = все рёбра.
Обучение full-batch (весь граф за шаг) на CPU или GPU.

Варианты функции потерь (аргумент --loss):
  class  — взвешенная по классу кросс-энтропия (как в литературе);
  amount — веса отмывочных переводов пропорциональны потерям L = min(сумма, CAP) (идея диплома).

Запуск: python 07_gnn_seeds.py --loss class --seed 0   (сиды 0, 1, 2; --loss class или amount)
Результаты: results_07_<loss>_s<seed>.txt, metrics_07_<loss>_s<seed>.json
"""
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
args = ap.parse_args()

HERE = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
TAG = f"{args.loss}_s{args.seed}"
OUT = HERE / f"results_07_{TAG}.txt"
BUDGETS = [100, 500, 1000, 2000]
TAIL_START = np.datetime64("2022-09-11")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(os.cpu_count() or 4)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lines = []


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


t0 = time.time()
log(f"Устройство: {DEV}, потоков: {torch.get_num_threads()}, вариант потерь: {args.loss}, "
    f"hidden={args.hidden}, layers={args.layers}")

# ------------------------------------------------------------------ данные
COLS = ["Timestamp", "From_Bank", "From_Acc", "To_Bank", "To_Acc", "Amount_Received",
        "Receiving_Currency", "Amount_Paid", "Payment_Currency", "Payment_Format", "Is_Laundering"]
df = pd.read_csv(find("HI-Small_Trans", [".csv", ".zip"]), header=0, names=COLS,
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
tmin = df["Timestamp"].values.astype("datetime64[m]").astype(np.int64)

# номера портов: порядок первого контакта с контрагентом
e = pd.DataFrame({"s": S, "d": D, "t": tmin})
first = e.groupby(["s", "d"], sort=False)["t"].min().reset_index()
first["out_port"] = first.sort_values("t").groupby("s").cumcount()
first["in_port"] = first.sort_values("t").groupby("d").cumcount()
e = e.merge(first[["s", "d", "out_port", "in_port"]], on=["s", "d"], how="left")
out_port, in_port = e["out_port"].values, e["in_port"].values
del e, first

num = np.stack([
    np.log1p(usd), np.log1p(usd_r),
    (cur_p != cur_r).astype(np.float32), (S == D).astype(np.float32),
    (banks[:n] == banks[n:]).astype(np.float32),
    df["Timestamp"].dt.hour.values / 23.0,
    np.log1p(out_port), np.log1p(in_port),
], axis=1).astype(np.float32)
mu, sd = num.mean(0), num.std(0) + 1e-6
num = (num - mu) / sd
NUM = torch.tensor(num)
CP, CR, FM = (torch.tensor(a, dtype=torch.long) for a in (cur_p, cur_r, fmt))
ST, DT = torch.tensor(S), torch.tensor(D)
Y = torch.tensor(y)

order = np.argsort(df["Timestamp"].values, kind="stable")
tr, va, te = order[:int(n * .6)], order[int(n * .6):int(n * .8)], order[int(n * .8):]
log(f"Загрузка и признаки: {time.time() - t0:.0f} с.  Счетов {N:,}, транзакций {n:,}")


class Graph:
    """Набор рёбер (индексы транзакций) + степени узлов в этом графе. Хранится в ОЗУ, на GPU — по требованию."""

    def __init__(self, idx):
        self.idx = torch.tensor(idx, dtype=torch.long)
        self.s, self.d = ST[self.idx], DT[self.idx]
        self.num, self.cp, self.cr, self.fm = (x[self.idx] for x in (NUM, CP, CR, FM))
        deg_in = torch.bincount(self.d, minlength=N).float()
        deg_out = torch.bincount(self.s, minlength=N).float()
        self.deg_in, self.deg_out = deg_in.clamp(min=1), deg_out.clamp(min=1)
        self.node_x = torch.stack([torch.log1p(deg_in), torch.log1p(deg_out)], 1)

    def to(self, dev):
        g = Graph.__new__(Graph)
        for k in ("s", "d", "num", "cp", "cr", "fm", "deg_in", "deg_out", "node_x"):
            setattr(g, k, getattr(self, k).to(dev))
        return g


G_tr = Graph(tr).to(DEV)
G_va = Graph(np.concatenate([tr, va]))
G_te = Graph(order)
pos_va = torch.arange(len(tr), len(tr) + len(va), device=DEV)    # позиции val-рёбер внутри G_va
# позиции test-рёбер внутри G_te (order отсортирован по времени, test — последние 20%)
pos_te = torch.arange(len(tr) + len(va), n, device=DEV)


def mlp(i, o, h):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class MultiGNN(nn.Module):
    def __init__(self, H, L):
        super().__init__()
        self.e_cp, self.e_cr, self.e_fm = nn.Embedding(32, 4), nn.Embedding(32, 4), nn.Embedding(16, 4)
        self.edge_in = nn.Linear(NUM.shape[1] + 12, H)
        self.node_in = nn.Linear(2, H)
        self.msg_in = nn.ModuleList([mlp(2 * H, H, H) for _ in range(L)])    # от отправителей (входящие)
        self.msg_out = nn.ModuleList([mlp(2 * H, H, H) for _ in range(L)])   # от получателей (обратная передача)
        self.upd = nn.ModuleList([mlp(3 * H, H, H) for _ in range(L)])
        self.eupd = nn.ModuleList([mlp(3 * H, H, H) for _ in range(L)])      # обновление рёбер
        self.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self.enorm = nn.ModuleList([nn.LayerNorm(H) for _ in range(L)])
        self.head = mlp(3 * H, 1, H)

    def layer(self, k, h, ea, s, d, deg_in, deg_out):
        m_in = self.msg_in[k](torch.cat([h[s], ea], 1))     # сообщение отправитель -> получатель
        m_out = self.msg_out[k](torch.cat([h[d], ea], 1))   # обратное сообщение получатель -> отправитель
        a_in = torch.zeros_like(h).index_add_(0, d, m_in) / deg_in[:, None]
        a_out = torch.zeros_like(h).index_add_(0, s, m_out) / deg_out[:, None]
        h = self.norm[k](h + self.upd[k](torch.cat([h, a_in, a_out], 1)))
        ea = self.enorm[k](ea + self.eupd[k](torch.cat([h[s], h[d], ea], 1)))
        return h, ea

    def forward(self, g):
        ea = F.relu(self.edge_in(torch.cat([g.num, self.e_cp(g.cp), self.e_cr(g.cr), self.e_fm(g.fm)], 1)))
        h = F.relu(self.node_in(g.node_x))
        for k in range(len(self.upd)):
            if self.training:   # экономия памяти видеокарты: слой пересчитывается при обратном проходе
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
    lw = (Ltr[y_tr == 1] / Ltr[y_tr == 1].mean()).clamp(max=args.max_w)   # ограничение веса одного перевода
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
np.save(HERE / f"test_scores_07_{TAG}.npy", np.vstack([te, p_te]))


# ------------------------------------------------------------------ метрики (как в шаге 4)
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


pv, rv, tv = pr_curve(y_va, p_va)
thr = tv[int(np.argmax(2 * pv * rv / np.maximum(pv + rv, 1e-12)))]
no_tail = df["Timestamp"].values[te] < TAIL_START
a, b = fit_platt(lg(p_va), y_va.astype(float))
pc = 1 / (1 + np.exp(-(a * lg(p_te) + b)))
amt = usd[te]
Lc = np.minimum(amt, args.cap)

log("\n=== Классификация (тест) ===")
log(f"  F1 весь тест:      {f1_at(y_te, p_te, thr) * 100:.2f} %    PR-AUC {pr_auc(y_te, p_te):.4f}")
log(f"  F1 без хвоста:     {f1_at(y_te[no_tail], p_te[no_tail], thr) * 100:.2f} %    "
    f"PR-AUC {pr_auc(y_te[no_tail], p_te[no_tail]):.4f}")
log("  Сравни: наш XGBoost 70.68 / 59.79;  Multi-PNA+EU (статья) 68.16")
log(f"\n=== Бюджет K: доля потерь L=min(сумма,{args.cap / 1e6:.0f} млн) ===")
log("  " + f"{'':<26}" + "".join(f"{'K=' + str(K):>9}" for K in BUDGETS))
for lab, sc, mask in [("по p (весь)", pc, None), ("по p·L (весь)", pc * Lc, None),
                      ("по p (без хвоста)", pc, no_tail), ("по p·L (без хвоста)", pc * Lc, no_tail)]:
    if mask is None:
        vals = [capture(sc, Lc, y_te, K) for K in BUDGETS]
    else:
        vals = [capture(sc[mask], Lc[mask], y_te[mask], K) for K in BUDGETS]
    log("  " + f"{lab:<26}" + "".join(f"{v:>9.3f}" for v in vals))
log("  Доля пойманных случаев:")
for lab, sc in [("по p", pc), ("по p·L", pc * Lc)]:
    log("  " + f"{lab:<26}" + "".join(f"{capture(sc, np.ones_like(amt), y_te, K):>9.3f}" for K in BUDGETS))
import json
res = {"loss": args.loss, "seed": args.seed, "best_val_prauc": best,
       "f1": f1_at(y_te, p_te, thr), "prauc": pr_auc(y_te, p_te),
       "f1_nt": f1_at(y_te[no_tail], p_te[no_tail], thr), "prauc_nt": pr_auc(y_te[no_tail], p_te[no_tail])}
for K in BUDGETS:
    res[f"p_K{K}"] = capture(pc, Lc, y_te, K)
    res[f"pL_K{K}"] = capture(pc * Lc, Lc, y_te, K)
    res[f"nt_p_K{K}"] = capture(pc[no_tail], Lc[no_tail], y_te[no_tail], K)
    res[f"nt_pL_K{K}"] = capture(pc[no_tail] * Lc[no_tail], Lc[no_tail], y_te[no_tail], K)
    res[f"case_p_K{K}"] = capture(pc, np.ones_like(amt), y_te, K)
    res[f"case_pL_K{K}"] = capture(pc * Lc, np.ones_like(amt), y_te, K)
(HERE / f"metrics_07_{TAG}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log(f"\nГотово за {(time.time() - t0) / 60:.1f} мин.")
