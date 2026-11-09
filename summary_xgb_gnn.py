import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
rows = [json.loads(p.read_text(encoding="utf-8")) | {"model": "GNN " + json.loads(p.read_text(encoding="utf-8"))["loss"]}
        for p in sorted(HERE.glob("metrics_07_*.json"))]
G = pd.DataFrame(rows)
X = pd.read_csv(HERE / "results_04_seeds.csv").rename(columns={"f1_notail": "f1_nt", "prauc_notail": "prauc_nt"})
X["model"] = "XGBoost"
A = pd.concat([X, G], ignore_index=True)

lines = []


def log(s=""):
    lines.append(s)


def cell(sub, col):
    v = sub[col].dropna() * 100
    return f"{v.mean():5.1f} ± {v.std():3.1f}" if len(v) > 1 else (f"{v.mean():5.1f}" if len(v) else "   н/д")


models = ["XGBoost", "GNN class", "GNN amount"]
log("Сводка (среднее ± std по сидам; число сидов: " +
    ", ".join(f"{m} {int((A['model'] == m).sum())}" for m in models) + ")\n")
blocks = [
    ("Классификация, %", [("F1 весь тест", "f1"), ("F1 без хвоста", "f1_nt"),
                          ("PR-AUC весь тест", "prauc"), ("PR-AUC без хвоста", "prauc_nt")]),
    ("Доля предотвращённых потерь L=min(сумма,1 млн), %",
     [(f"{s} K={K}", f"{c}_K{K}") for K in (100, 500, 1000, 2000)
      for s, c in (("по p   ", "p"), ("по p·L ", "pL"))]),
    ("То же без хвоста, %",
     [(f"{s} K={K}", f"nt_{c}_K{K}") for K in (100, 1000) for s, c in (("по p   ", "p"), ("по p·L ", "pL"))]),
    ("Доля пойманных случаев, %",
     [(f"{s} K={K}", f"case_{c}_K{K}") for K in (100, 1000) for s, c in (("по p   ", "p"), ("по p·L ", "pL"))]),
]
for title, items in blocks:
    log(title)
    log("  " + f"{'':<22}" + "".join(f"{m:>16}" for m in models))
    for lab, col in items:
        log("  " + f"{lab:<22}" + "".join(f"{cell(A[A['model'] == m], col):>16}" for m in models))
    log()
(HERE / "results_08_summary.txt").write_text("\n".join(lines), encoding="utf-8")
