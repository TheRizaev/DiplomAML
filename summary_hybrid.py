import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
import os as _os
DATA = _os.environ.get("AML_DATA", "HI-Small")
OUTD = HERE if DATA == "HI-Small" else HERE / DATA
OUTD.mkdir(exist_ok=True)

import re as _re
TAGF = _os.environ.get("SUMMARY_TAG", "")
rows = [json.loads(p.read_text(encoding="utf-8")) for p in OUTD.glob("metrics_10_*.json")
        if _re.search(r"_s\d+" + _re.escape(TAGF) + r"$", p.stem)]
for p in ([] if TAGF else OUTD.glob("metrics_07_class_*.json")):
    r = json.loads(p.read_text(encoding="utf-8")); r["model"] = "gnn_plain"; rows.append(r)
A = pd.DataFrame(rows)
models = [("xgb", "XGBoost"), ("gnn_plain", "GNN (шаг 7)"), ("gnn_hybrid", "GNN-гибрид"), ("ensemble", "Ансамбль"), ("ensemble_cal", "Ансамбль кал.")]
lines = []
def log(s=""):
    lines.append(s)
def cell(m, c):
    v = A.loc[A["model"] == m, c].dropna() * 100
    return f"{v.mean():5.1f} ± {v.std():3.1f}" if len(v) > 1 else (f"{v.mean():5.1f}" if len(v) else "н/д")
log("Сидов: " + ", ".join(f"{t} {int((A['model'] == m).sum())}" for m, t in models) + "\n")
items = [("F1", "f1"), ("F1 без хвоста", "f1_nt"), ("PR-AUC", "prauc"), ("PR-AUC без хвоста", "prauc_nt"),
         ("p·L K=100", "pL_K100"), ("p·L K=500", "pL_K500"), ("p·L K=1000", "pL_K1000"), ("p·L K=2000", "pL_K2000"),
         ("p·L K=1000 без хвоста", "nt_pL_K1000"), ("по p K=1000", "p_K1000"), ("случаи p·L K=1000", "case_pL_K1000")]
log("  " + f"{'метрика, %':<24}" + "".join(f"{t:>15}" for _, t in models))
for lab, c in items:
    log("  " + f"{lab:<24}" + "".join(f"{cell(m, c):>15}" for m, _ in models))
(OUTD / f"results_11_summary{TAGF}.txt").write_text("\n".join(lines), encoding="utf-8")
