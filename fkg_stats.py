#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisis estadistico de results_raw.csv.

Produce:
  - media +/- desviacion estandar por modelo y por lambda
  - prueba de significancia PAREADA sobre las 30 semillas (Wilcoxon signed-rank)
  - correccion por comparaciones multiples (Holm-Bonferroni)
  - tamaño de efecto (Cohen's d pareado)
  - AUC / AP desglosados por tipo de relacion
  - tabla lista para pegar en el manuscrito (markdown y LaTeX)

Uso:
    python fkg_stats.py results_raw.csv --reference rgcn --lam 0.3
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["auc", "ap", "delta_muscle", "retrieval_muscle",
           "delta_type", "retrieval_type", "silhouette_type"]

PRETTY = {"auc": "AUC", "ap": "AP", "delta_muscle": "Δ (muscle)",
          "retrieval_muscle": "Retrieval@5 (muscle)", "delta_type": "Δ (type)",
          "retrieval_type": "Retrieval@5 (type)", "silhouette_type": "Silhouette"}


def paired_cohens_d(a, b):
    d = np.asarray(a) - np.asarray(b)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def holm(pvals):
    """Holm-Bonferroni. Devuelve los p ajustados en el orden original."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adj = np.full_like(p, np.nan)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def summary_table(df, lam):
    sub = df[(df["lam"] == lam) | (df["model"] == "jaccard")]
    rows = []
    for model, g in sub.groupby("model", sort=False):
        row = {"model": model, "label": g["label"].iloc[0],
               "runs": int(g["seed"].nunique()), "params": int(g["params"].iloc[0])}
        for m in METRICS:
            row[m] = f"{g[m].mean():.4f} ± {g[m].std(ddof=1):.4f}" if g[m].notna().any() else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def significance(df, reference, lam):
    ref = df[(df["model"] == reference) & (df["lam"] == lam)].sort_values("seed")
    out = []
    for model, g in df[(df["lam"] == lam) & (df["model"] != reference)].groupby("model"):
        g = g.sort_values("seed")
        common = sorted(set(ref["seed"]) & set(g["seed"]))
        a = ref[ref["seed"].isin(common)]
        b = g[g["seed"].isin(common)]
        for m in METRICS:
            x, y = a[m].values, b[m].values
            if np.isnan(x).any() or np.isnan(y).any() or np.allclose(x, y):
                p = np.nan
            else:
                p = wilcoxon(x, y).pvalue
            out.append(dict(metric=m, reference=reference, model=model, n=len(common),
                            mean_ref=x.mean(), mean_model=y.mean(),
                            diff=x.mean() - y.mean(), p=p, d=paired_cohens_d(x, y)))
    res = pd.DataFrame(out)
    if not res.empty:
        res["p_holm"] = holm(res["p"].values)
        res["sig"] = np.where(res["p_holm"] < 0.05, "*", "")
    return res


def lambda_ablation(df):
    """Efecto de quitar la regularizacion funcional (lambda = 0)."""
    out = []
    for model, g in df[df["model"] != "jaccard"].groupby("model"):
        lams = sorted(g["lam"].dropna().unique())
        if len(lams) < 2:
            continue
        lo, hi = lams[0], lams[-1]
        a = g[g["lam"] == hi].sort_values("seed")
        b = g[g["lam"] == lo].sort_values("seed")
        common = sorted(set(a["seed"]) & set(b["seed"]))
        a, b = a[a["seed"].isin(common)], b[b["seed"].isin(common)]
        for m in METRICS:
            x, y = a[m].values, b[m].values
            if np.isnan(x).any() or np.isnan(y).any() or np.allclose(x, y):
                p = np.nan
            else:
                p = wilcoxon(x, y).pvalue
            out.append(dict(model=model, metric=m, mean_lam_hi=x.mean(),
                            mean_lam_0=y.mean(), diff=x.mean() - y.mean(),
                            p=p, d=paired_cohens_d(x, y)))
    return pd.DataFrame(out)


def per_relation_table(df, lam):
    rows = []
    for model, g in df[(df["lam"] == lam) & (df["model"] != "jaccard")].groupby("model"):
        acc = {}
        for blob in g["per_relation"]:
            for rel, vals in json.loads(blob).items():
                acc.setdefault(rel, {"auc": [], "ap": [], "neg_fail": [], "n": []})
                acc[rel]["auc"].append(vals["auc"])
                acc[rel]["ap"].append(vals["ap"])
                acc[rel]["neg_fail"].append(vals["neg_fail"])
                acc[rel]["n"].append(vals["n_test"])
        for rel, v in acc.items():
            rows.append(dict(model=model, relation=rel, n_test=int(np.mean(v["n"])),
                             auc=f"{np.nanmean(v['auc']):.4f} ± {np.nanstd(v['auc'], ddof=1):.4f}",
                             ap=f"{np.nanmean(v['ap']):.4f} ± {np.nanstd(v['ap'], ddof=1):.4f}",
                             neg_impossible=f"{np.mean(v['neg_fail']):.1%}"))
    return pd.DataFrame(rows)


def to_latex(tab):
    cols = ["label"] + METRICS
    t = tab[cols].rename(columns={"label": "Model", **PRETTY})
    return t.to_latex(index=False, escape=False, column_format="l" + "c" * len(METRICS))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("csv")
    ap_.add_argument("--reference", default="rgcn")
    ap_.add_argument("--lam", type=float, default=0.3)
    ap_.add_argument("--prefix", default="tabla")
    a = ap_.parse_args()

    df = pd.read_csv(a.csv)

    tab = summary_table(df, a.lam)
    sig = significance(df, a.reference, a.lam)
    abl = lambda_ablation(df)
    rel = per_relation_table(df, a.lam)

    print("\n=== Tabla principal (lambda = %s) ===" % a.lam)
    print(tab.to_markdown(index=False))
    print("\n=== Wilcoxon pareado vs %s (Holm) ===" % a.reference)
    print(sig.to_markdown(index=False, floatfmt=".4f"))
    print("\n=== Ablacion lambda = 0 ===")
    print(abl.to_markdown(index=False, floatfmt=".4f"))
    print("\n=== Por tipo de relacion ===")
    print(rel.to_markdown(index=False))

    tab.to_csv(f"{a.prefix}_principal.csv", index=False)
    sig.to_csv(f"{a.prefix}_significancia.csv", index=False)
    abl.to_csv(f"{a.prefix}_ablacion_lambda.csv", index=False)
    rel.to_csv(f"{a.prefix}_por_relacion.csv", index=False)
    with open(f"{a.prefix}_principal.tex", "w", encoding="utf-8") as f:
        f.write(to_latex(tab))
    print(f"\nArchivos escritos con prefijo '{a.prefix}_'")


if __name__ == "__main__":
    main()